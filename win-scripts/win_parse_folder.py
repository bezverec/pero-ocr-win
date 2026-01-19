#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Windows-friendly PERO OCR folder parser.

This is a CLI utility script; some pylint complexity rules are relaxed locally
to keep the code practical and readable.
"""

from __future__ import annotations

import argparse
import configparser
import logging
import os
import re
import sys
import time
import traceback
from multiprocessing import Pool
from typing import List, Optional, Set

import cv2
import numpy as np
import torch

# PERO imports must be at top-level (pylint: wrong-import-position).
# NOTE: utils import triggers side-effects on import in upstream PERO.
from pero_ocr import utils as _pero_utils  # noqa: F401  pylint: disable=unused-import
from pero_ocr.core.layout import PageLayout
from pero_ocr.document_ocr.page_parser import PageParser

# Optional dependencies (avoid import-outside-toplevel)
try:
    import lmdb  # type: ignore
except ImportError:  # pragma: no cover
    lmdb = None  # type: ignore

# --- Windows-safe optional import of safe_gpu (POSIX-only) --------------------
SAFE_GPU_AVAILABLE = False
SAFE_GPU = None  # will hold module reference if available

try:
    # safe_gpu uses fcntl (POSIX), so it will fail on Windows.
    if os.name != "nt":
        from safe_gpu import safe_gpu as _safe_gpu  # type: ignore
        SAFE_GPU = _safe_gpu
        SAFE_GPU_AVAILABLE = True
except ImportError:  # pragma: no cover
    SAFE_GPU_AVAILABLE = False
    SAFE_GPU = None


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", required=True, help="Path to input config file."
    )
    parser.add_argument(
        "-s",
        "--skip-processed",
        action="store_true",
        required=False,
        help="If set, already processed files are skipped.",
    )
    parser.add_argument("-i", "--input-image-path", help="")
    parser.add_argument("-x", "--input-xml-path", help="")
    parser.add_argument("--input-logit-path", help="")
    parser.add_argument("--output-xml-path", help="")
    parser.add_argument("--output-render-path", help="")
    parser.add_argument("--output-line-path", help="")
    parser.add_argument("--output-logit-path", help="")
    parser.add_argument("--output-alto-path", help="")
    parser.add_argument("--output-transcriptions-file-path", help="")
    parser.add_argument(
        "--skipp-missing-xml",
        action="store_true",
        help="Skip images which have missing xml.",
    )

    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=None,
        help=(
            "If set, the computation runs on the specified GPU. "
            "If not set and --device=gpu, safe-gpu is used on POSIX "
            "(if available); on Windows it falls back to default CUDA."
        ),
    )

    parser.add_argument(
        "--process-count",
        type=int,
        default=1,
        help="Number of parallel processes (works mostly only for line cropping).",
    )
    return parser.parse_args()


def setup_logging(config):
    """Configure logging based on config."""
    level = config.get("LOGGING_LEVEL", fallback="WARNING")
    level = logging.getLevelName(level)

    logging.basicConfig(
        format="[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
        level=level,
    )

    logger = logging.getLogger("pero_ocr")
    logger.setLevel(level)


def get_value_or_none(config, section, key):
    """Get value from config or None if not present."""
    if config.has_option(section, key):
        return config[section][key]
    return None


def create_dir_if_not_exists(path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def load_already_processed_files_in_directory(directory: Optional[str]) -> Set[str]:
    """Load set of already processed files from directory."""
    already_processed: Set[str] = set()

    if directory is not None:
        regex = re.compile(r"(.+?)(\.logits|\.xml|\.jpg)")
        for file in os.listdir(directory):
            matched = regex.match(file)
            if matched:
                already_processed.add(matched.groups()[0])

    return already_processed


def load_already_processed_files(directories: List[Optional[str]]) -> Set[str]:
    """Load intersection of already processed files from multiple directories."""
    already_processed: Set[str] = set()
    first = True

    for directory in directories:
        if directory is None:
            continue

        files = load_already_processed_files_in_directory(directory)
        if first:
            already_processed = files
            first = False
        else:
            already_processed = already_processed.intersection(files)

    return already_processed


def get_device(
    device: str,
    gpu_index: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
):
    """
    Windows-safe device selection.

    - If device == "cpu": always CPU.
    - If gpu_index is provided: use that CUDA device (cuda:<id>).
    - Else if device == "gpu":
        - If CUDA not available: warn and fallback to CPU.
        - If safe_gpu is available (POSIX only): claim a GPU and use default 'cuda'.
        - Else (Windows or safe_gpu missing): use default 'cuda'.
    """
    if device != "gpu":
        return torch.device("cpu")

    if not torch.cuda.is_available():
        if logger:
            logger.warning("CUDA not available. Falling back to CPU.")
        return torch.device("cpu")

    if gpu_index is not None:
        return torch.device(f"cuda:{gpu_index}")

    # No explicit GPU index: try safe_gpu on POSIX if available
    if SAFE_GPU_AVAILABLE and SAFE_GPU is not None:
        try:
            SAFE_GPU.claim_gpus(logger=logger)
        except Exception as exc:  # pylint: disable=broad-except
            if logger:
                logger.warning(
                    "safe_gpu.claim_gpus failed (%s); using default CUDA device.", exc
                )
    else:
        if logger:
            logger.info("safe_gpu not available (or Windows). Using default CUDA device.")

    return torch.device("cuda")


class LmdbWriter:
    """Writer for LMDB database storage."""
    # pylint: disable=too-few-public-methods

    def __init__(self, path):
        if lmdb is None:
            raise RuntimeError(
                "LMDB support requested but 'lmdb' is not installed. "
                "Install it or avoid --output-line-path pointing to lmdb."
            )
        gb100 = 100_000_000_000
        self.env_out = lmdb.open(path, map_size=gb100)

    def __call__(self, page_layout: PageLayout, file_id):
        """Write page layout to LMDB."""
        all_lines = sorted(list(page_layout.lines_iterator()), key=lambda x: x.id)
        records_to_write = {}

        for line in all_lines:
            if not line.transcription:
                continue
            key = f"{file_id}-{line.id}.jpg"
            img = cv2.imencode(
                ".jpg",
                line.crop.astype(np.uint8),
                [int(cv2.IMWRITE_JPEG_QUALITY), 95],
            )[1].tobytes()
            records_to_write[key] = img

        with self.env_out.begin(write=True) as txn_out:
            cursor = txn_out.cursor()
            for key, value in records_to_write.items():
                cursor.put(key.encode(), value)


class Computator:
    """Main computation class for processing pages."""
    # This is a CLI worker with many parameters by design.
    # pylint: disable=too-many-instance-attributes,too-many-arguments,too-many-positional-arguments

    def __init__(
        self,
        page_parser,
        input_image_path,
        input_xml_path,
        input_logit_path,
        output_render_path,
        output_logit_path,
        output_alto_path,
        output_xml_path,
        output_line_path,
    ):
        self.page_parser = page_parser
        self.input_image_path = input_image_path
        self.input_xml_path = input_xml_path
        self.input_logit_path = input_logit_path
        self.output_render_path = output_render_path
        self.output_logit_path = output_logit_path
        self.output_alto_path = output_alto_path
        self.output_xml_path = output_xml_path
        self.output_line_path = output_line_path

    def _process_outputs(self, page_layout, file_id, image):
        """Process and save all outputs."""
        if self.output_xml_path is not None:
            page_layout.to_pagexml(os.path.join(self.output_xml_path, f"{file_id}.xml"))

        if self.output_render_path is not None and image is not None:
            page_layout.render_to_image(image)
            cv2.imwrite(
                os.path.join(self.output_render_path, f"{file_id}.jpg"),
                image,
                [int(cv2.IMWRITE_JPEG_QUALITY), 70],
            )

        if self.output_logit_path is not None:
            page_layout.save_logits(os.path.join(self.output_logit_path, f"{file_id}.logits"))

        if self.output_alto_path is not None:
            page_layout.to_altoxml(os.path.join(self.output_alto_path, f"{file_id}.xml"))

        if self.output_line_path is not None:
            self._save_line_images(page_layout, file_id)

    def _save_line_images(self, page_layout, file_id):
        """Save individual line images."""
        if "lmdb" in self.output_line_path:
            LmdbWriter(self.output_line_path)(page_layout, file_id)
            return

        for region in page_layout.regions:
            for line in region.lines:
                line_path = os.path.join(self.output_line_path, f"{file_id}-{line.id}.jpg")
                cv2.imwrite(
                    line_path,
                    line.crop.astype(np.uint8),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 98],
                )

    @staticmethod
    def _extract_annotations(page_layout, file_id):
        """Extract transcription annotations."""
        all_lines = sorted(list(page_layout.lines_iterator()), key=lambda x: x.id)
        return [
            f"{file_id}-{line.id}.jpg {line.transcription}"
            for line in all_lines
            if line.transcription
        ]

    def __call__(self, image_file_name, file_id, index, ids_count):
        """Process a single file."""
        print(f"Processing {file_id}")
        start_time = time.time()

        try:
            if self.input_image_path is not None:
                img_path = os.path.join(self.input_image_path, image_file_name)
                image = cv2.imread(img_path, 1)
                if image is None:
                    raise RuntimeError(f'Unable to read image "{img_path}"')
            else:
                image = None

            if self.input_xml_path:
                page_layout = PageLayout(file=os.path.join(self.input_xml_path, f"{file_id}.xml"))
            else:
                page_layout = PageLayout(id=file_id, page_size=(image.shape[0], image.shape[1]))

            if self.input_logit_path is not None:
                page_layout.load_logits(os.path.join(self.input_logit_path, f"{file_id}.logits"))

            page_layout = self.page_parser.process_page(image, page_layout)
            self._process_outputs(page_layout, file_id, image)
            annotations = self._extract_annotations(page_layout, file_id)

        except KeyboardInterrupt:
            traceback.print_exc()
            print("Terminated by user.")
            sys.exit()
        except Exception as exc:  # pylint: disable=broad-except
            print(f"ERROR: Failed to process file {file_id}.")
            print(exc)
            traceback.print_exc()
            annotations = []

        elapsed = time.time() - start_time
        current = index + 1
        percentage = current / ids_count * 100
        print(
            f"DONE {current}/{ids_count} ({percentage:.2f} %) "
            f"[id: {file_id}] Time:{elapsed:.2f}"
        )
        return annotations


def _prepare_file_lists(input_image_path, input_xml_path, logger):
    """Prepare lists of files to process."""
    if input_image_path is not None:
        logger.info("Reading images from %s.", input_image_path)
        ignored_extensions = {"", ".xml", ".logits"}
        images_to_process = sorted(
            f
            for f in os.listdir(input_image_path)
            if os.path.splitext(f)[1].lower() not in ignored_extensions
        )
        ids_to_process = [os.path.splitext(os.path.basename(f))[0] for f in images_to_process]
        return ids_to_process, images_to_process

    if input_xml_path is not None:
        logger.info("Reading page xml from %s", input_xml_path)
        xml_to_process = [f for f in os.listdir(input_xml_path) if os.path.splitext(f)[1] == ".xml"]
        images_to_process = [None] * len(xml_to_process)
        ids_to_process = [os.path.splitext(os.path.basename(f))[0] for f in xml_to_process]
        return ids_to_process, images_to_process

    raise RuntimeError("Either INPUT_IMAGE_PATH or INPUT_XML_PATH has to be specified.")


def _filter_missing_xml(input_xml_path, ids_to_process, images_to_process):
    """Filter files where XML is missing."""
    filtered_ids = []
    filtered_images = []
    for file_id, image_file_name in zip(ids_to_process, images_to_process):
        if os.path.exists(os.path.join(input_xml_path, f"{file_id}.xml")):
            filtered_ids.append(file_id)
            filtered_images.append(image_file_name)
    return filtered_ids, filtered_images


def main():
    """Main entry point."""
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    args = parse_arguments()
    config_path = args.config

    if not os.path.isfile(config_path):
        print(f'ERROR: Config file does not exist: "{config_path}".')
        sys.exit(-1)

    config = configparser.ConfigParser()
    config.read(config_path)

    if "PARSE_FOLDER" not in config:
        config.add_section("PARSE_FOLDER")

    arg_mapping = {
        "INPUT_IMAGE_PATH": args.input_image_path,
        "INPUT_XML_PATH": args.input_xml_path,
        "INPUT_LOGIT_PATH": args.input_logit_path,
        "OUTPUT_XML_PATH": args.output_xml_path,
        "OUTPUT_RENDER_PATH": args.output_render_path,
        "OUTPUT_LINE_PATH": args.output_line_path,
        "OUTPUT_LOGIT_PATH": args.output_logit_path,
        "OUTPUT_ALTO_PATH": args.output_alto_path,
    }
    for key, value in arg_mapping.items():
        if value is not None:
            config["PARSE_FOLDER"][key] = value

    setup_logging(config["PARSE_FOLDER"])
    logger = logging.getLogger()

    device = get_device(args.device, args.gpu_id, logger)
    logger.info("Using device: %s", device)

    page_parser = PageParser(config, config_path=os.path.dirname(config_path), device=device)

    input_image_path = get_value_or_none(config, "PARSE_FOLDER", "INPUT_IMAGE_PATH")
    input_xml_path = get_value_or_none(config, "PARSE_FOLDER", "INPUT_XML_PATH")
    input_logit_path = get_value_or_none(config, "PARSE_FOLDER", "INPUT_LOGIT_PATH")
    output_render_path = get_value_or_none(config, "PARSE_FOLDER", "OUTPUT_RENDER_PATH")
    output_line_path = get_value_or_none(config, "PARSE_FOLDER", "OUTPUT_LINE_PATH")
    output_xml_path = get_value_or_none(config, "PARSE_FOLDER", "OUTPUT_XML_PATH")
    output_logit_path = get_value_or_none(config, "PARSE_FOLDER", "OUTPUT_LOGIT_PATH")
    output_alto_path = get_value_or_none(config, "PARSE_FOLDER", "OUTPUT_ALTO_PATH")

    if not page_parser.provides_ctc_logits:
        if not input_logit_path and output_alto_path:
            logger.error(
                "Cannot create ALTO with current PageParser (transformer outputs are incompatible)"
            )
            sys.exit(2)
        if output_logit_path:
            logger.error(
                "Cannot store logits with current PageParser (transformer outputs are incompatible)"
            )
            sys.exit(2)

    for directory in [output_render_path, output_line_path, output_xml_path, output_logit_path, output_alto_path]:
        if directory:
            create_dir_if_not_exists(directory)

    if input_logit_path is not None and input_xml_path is None:
        input_logit_path = None
        logger.warning(
            "Logit path specified and Page XML path not specified. Logits will be ignored."
        )

    ids_to_process, images_to_process = _prepare_file_lists(input_image_path, input_xml_path, logger)

    if args.skip_processed:
        already_processed = load_already_processed_files([output_xml_path, output_logit_path, output_render_path])
        if already_processed:
            logger.info("Already processed %s file(s).", len(already_processed))
            filtered = [(i, im) for i, im in zip(ids_to_process, images_to_process) if i not in already_processed]
            if filtered:
                ids_to_process, images_to_process = zip(*filtered)
                ids_to_process, images_to_process = list(ids_to_process), list(images_to_process)
            else:
                ids_to_process, images_to_process = [], []

    if input_xml_path and args.skipp_missing_xml:
        ids_to_process, images_to_process = _filter_missing_xml(input_xml_path, ids_to_process, images_to_process)

    computator = Computator(
        page_parser,
        input_image_path,
        input_xml_path,
        input_logit_path,
        output_render_path,
        output_logit_path,
        output_alto_path,
        output_xml_path,
        output_line_path,
    )

    start_time = time.time()
    results = []

    if args.process_count > 1 and ids_to_process:
        with Pool(processes=args.process_count) as pool:
            tasks = [(img, file_id, idx, len(ids_to_process)) for idx, (file_id, img) in enumerate(zip(ids_to_process, images_to_process))]
            results = pool.starmap(computator, tasks)
    else:
        for idx, (file_id, img_file) in enumerate(zip(ids_to_process, images_to_process)):
            results.append(computator(img_file, file_id, idx, len(ids_to_process)))

    if args.output_transcriptions_file_path is not None:
        with open(args.output_transcriptions_file_path, "w", encoding="utf-8") as handle:
            for page_lines in results:
                if page_lines:
                    handle.write("\n".join(page_lines) + "\n")

    if page_parser.decoder:
        logger.info(page_parser.decoder.decoding_summary())

    if ids_to_process:
        avg_time = (time.time() - start_time) / len(ids_to_process)
        logger.info("AVERAGE PROCESSING TIME %s", avg_time)


if __name__ == "__main__":
    main()

