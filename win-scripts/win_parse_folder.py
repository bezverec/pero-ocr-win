#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script for processing OCR documents with pero_ocr.
"""

import argparse
import configparser
import logging
import logging.handlers
import os
import re
import sys
import time
import traceback
from multiprocessing import Pool
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import torch

from pero_ocr.core.layout import PageLayout
from pero_ocr.document_ocr.page_parser import PageParser

# --- Windows-safe optional import of safe_gpu (POSIX-only) --------------------
_SAFE_GPU_AVAILABLE = False
SAFE_GPU = None  # will hold module reference if available

try:
    # safe_gpu uses fcntl (POSIX), so it will fail on Windows.
    if os.name != "nt":
        from safe_gpu import safe_gpu as _safe_gpu  # type: ignore
        SAFE_GPU = _safe_gpu
        _SAFE_GPU_AVAILABLE = True
except ImportError:
    _SAFE_GPU_AVAILABLE = False
    SAFE_GPU = None
# ------------------------------------------------------------------------------


class ArgumentParser:
    """Parser for command line arguments."""

    @staticmethod
    def parse() -> argparse.Namespace:
        """Parse and return command line arguments."""
        parser = argparse.ArgumentParser()
        parser.add_argument('-c', '--config', required=True,
                            help='Path to input config file.')
        parser.add_argument('-s', '--skip-processed', action='store_true',
                            help='Skip already processed files.')
        parser.add_argument('-i', '--input-image-path', help='Image path')
        parser.add_argument('-x', '--input-xml-path', help='XML path')
        parser.add_argument('--input-logit-path', help='Logit path')
        parser.add_argument('--output-xml-path', help='Output XML path')
        parser.add_argument('--output-render-path', help='Render path')
        parser.add_argument('--output-line-path', help='Line image path')
        parser.add_argument('--output-logit-path', help='Output logit path')
        parser.add_argument('--output-alto-path', help='ALTO output path')
        parser.add_argument('--output-transcriptions-file-path',
                            help='Transcriptions file path')
        parser.add_argument('--skipp-missing-xml', action='store_true',
                            help='Skip images with missing XML.')

        parser.add_argument('--device', choices=["gpu", "cpu"], default="gpu")
        parser.add_argument(
            '--gpu-id',
            type=int,
            default=None,
            help='GPU ID to use (if not set, uses safe-gpu on POSIX)'
        )

        parser.add_argument('--process-count', type=int, default=1,
                            help='Number of parallel processes.')
        return parser.parse_args()


class ConfigManager:
    """Manages configuration loading and updates."""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = configparser.ConfigParser()
        self._load_config()

    def _load_config(self):
        """Load configuration from file."""
        if not os.path.isfile(self.config_path):
            raise FileNotFoundError(
                f'Config file does not exist: "{self.config_path}".'
            )
        self.config.read(self.config_path)

        if 'PARSE_FOLDER' not in self.config:
            self.config.add_section('PARSE_FOLDER')

    def update_from_args(self, args: argparse.Namespace):
        """Update configuration with command line arguments."""
        arg_mapping = {
            'INPUT_IMAGE_PATH': args.input_image_path,
            'INPUT_XML_PATH': args.input_xml_path,
            'INPUT_LOGIT_PATH': args.input_logit_path,
            'OUTPUT_XML_PATH': args.output_xml_path,
            'OUTPUT_RENDER_PATH': args.output_render_path,
            'OUTPUT_LINE_PATH': args.output_line_path,
            'OUTPUT_LOGIT_PATH': args.output_logit_path,
            'OUTPUT_ALTO_PATH': args.output_alto_path,
        }

        for key, value in arg_mapping.items():
            if value is not None:
                self.config['PARSE_FOLDER'][key] = value

    def get_value(self, section: str, key: str) -> Optional[str]:
        """Get value from config or None if not present."""
        if self.config.has_option(section, key):
            return self.config[section][key]
        return None


class LoggingConfig:
    """Configures logging."""

    @staticmethod
    def setup(config: configparser.SectionProxy):
        """Setup logging based on config."""
        level = config.get('LOGGING_LEVEL', fallback='WARNING')
        level = logging.getLevelName(level)

        logging.basicConfig(
            format='[%(levelname)s] %(asctime)s - %(name)s - %(message)s',
            level=level
        )

        logger = logging.getLogger('pero_ocr')
        logger.setLevel(level)


class DeviceManager:
    """Manages device selection (CPU/GPU)."""

    @staticmethod
    def get_device(device: str, gpu_index: Optional[int] = None,
                   logger: Optional[logging.Logger] = None) -> torch.device:
        """
        Get appropriate device for computation.
        """
        if device != "gpu":
            return torch.device("cpu")

        if not torch.cuda.is_available():
            if logger:
                logger.warning("CUDA not available. Falling back to CPU.")
            return torch.device("cpu")

        if gpu_index is not None:
            return torch.device(f"cuda:{gpu_index}")

        if _SAFE_GPU_AVAILABLE and SAFE_GPU is not None:
            try:
                SAFE_GPU.claim_gpus(logger=logger)
            except Exception as e:  # pylint: disable=broad-except
                if logger:
                    logger.warning("safe_gpu.claim_gpus failed: %s", e)
        elif logger:
            logger.info("safe_gpu not available. Using default CUDA device.")

        return torch.device("cuda")


class FileProcessor:
    """Processes individual files."""

    def __init__(self, config: ConfigManager):
        self.config = config
        self.page_parser = None
        self._paths = {}

    def initialize(self, args: argparse.Namespace) -> None:
        """Initialize the processor."""
        logger = logging.getLogger()
        device = DeviceManager.get_device(args.device, args.gpu_id, logger)
        logger.info("Using device: %s", device)

        self.page_parser = PageParser(
            self.config.config,
            config_path=os.path.dirname(self.config.config_path),
            device=device
        )
        self._load_paths()

    def _load_paths(self) -> None:
        """Load all paths from config."""
        self._paths = {
            'input_image': self.config.get_value('PARSE_FOLDER',
                                                 'INPUT_IMAGE_PATH'),
            'input_xml': self.config.get_value('PARSE_FOLDER',
                                               'INPUT_XML_PATH'),
            'input_logit': self.config.get_value('PARSE_FOLDER',
                                                 'INPUT_LOGIT_PATH'),
            'output_render': self.config.get_value('PARSE_FOLDER',
                                                   'OUTPUT_RENDER_PATH'),
            'output_line': self.config.get_value('PARSE_FOLDER',
                                                 'OUTPUT_LINE_PATH'),
            'output_xml': self.config.get_value('PARSE_FOLDER',
                                                'OUTPUT_XML_PATH'),
            'output_logit': self.config.get_value('PARSE_FOLDER',
                                                  'OUTPUT_LOGIT_PATH'),
            'output_alto': self.config.get_value('PARSE_FOLDER',
                                                 'OUTPUT_ALTO_PATH'),
        }

    def _check_compatibility(self) -> None:
        """Check parser compatibility with requested outputs."""
        logger = logging.getLogger()

        if not self.page_parser.provides_ctc_logits:
            if (not self._paths['input_logit'] and
                    self._paths['output_alto']):
                logger.error(
                    'Cannot create ALTO with current PageParser '
                    '(transformer outputs are incompatible)'
                )
                sys.exit(2)

            if self._paths['output_logit']:
                logger.error(
                    'Cannot store logits with current PageParser '
                    '(transformer outputs are incompatible)'
                )
                sys.exit(2)

    def _create_output_dirs(self) -> None:
        """Create output directories if they don't exist."""
        paths = [
            self._paths['output_render'],
            self._paths['output_line'],
            self._paths['output_xml'],
            self._paths['output_logit'],
            self._paths['output_alto'],
        ]

        for path in paths:
            if path and not os.path.exists(path):
                os.makedirs(path)

    def _handle_logit_warning(self) -> None:
        """Handle logit path warnings."""
        logger = logging.getLogger()
        if (self._paths['input_logit'] is not None and
                self._paths['input_xml'] is None):
            self._paths['input_logit'] = None
            logger.warning(
                'Logit path specified without XML path. Logits will be ignored.'
            )


class LmdbWriter:
    """Writer for LMDB database storage."""

    def __init__(self, path: str):
        import lmdb
        gb100 = 100_000_000_000
        self.env_out = lmdb.open(path, map_size=gb100)

    def __call__(self, page_layout: PageLayout, file_id: str) -> None:
        """Write page layout to LMDB."""
        all_lines = list(page_layout.lines_iterator())
        all_lines.sort(key=lambda x: x.id)
        records = {}

        for line in all_lines:
            if line.transcription:
                key = f'{file_id}-{line.id}.jpg'
                img = cv2.imencode(
                    '.jpg',
                    line.crop.astype(np.uint8),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 95]
                )[1].tobytes()
                records[key] = img

        with self.env_out.begin(write=True) as txn:
            cursor = txn.cursor()
            for key, value in records.items():
                cursor.put(key.encode(), value)


class PageComputator:
    """Computes and processes individual pages."""

    def __init__(self, page_parser: PageParser, paths: Dict[str, Optional[str]]):
        self.page_parser = page_parser
        self.paths = paths

    def process(self, image_file_name: Optional[str], file_id: str,
                index: int, total_count: int) -> List[str]:
        """Process a single file."""
        print(f"Processing {file_id}")
        start_time = time.time()

        try:
            page_layout = self._load_page(image_file_name, file_id)
            page_layout = self.page_parser.process_page(
                self._load_image(image_file_name),
                page_layout
            )
            self._save_outputs(page_layout, file_id)
            annotations = self._extract_annotations(page_layout, file_id)

        except KeyboardInterrupt:
            traceback.print_exc()
            print('Terminated by user.')
            sys.exit()
        except Exception as error:  # pylint: disable=broad-except
            print(f'ERROR: Failed to process file {file_id}.')
            print(error)
            traceback.print_exc()
            annotations = []

        self._print_progress(index, total_count, file_id, start_time)
        return annotations

    def _load_image(self, image_file_name: Optional[str]) -> Optional[np.ndarray]:
        """Load image from file."""
        if (image_file_name is None or
                self.paths['input_image'] is None):
            return None

        image_path = os.path.join(self.paths['input_image'], image_file_name)
        image = cv2.imread(image_path, 1)
        if image is None:
            raise RuntimeError(f'Unable to read image "{image_path}"')
        return image

    def _load_page(self, image_file_name: Optional[str],
                   file_id: str) -> PageLayout:
        """Load or create page layout."""
        if self.paths['input_xml']:
            xml_path = os.path.join(self.paths['input_xml'], file_id + '.xml')
            page_layout = PageLayout(file=xml_path)
        else:
            image = self._load_image(image_file_name)
            page_size = image.shape[:2] if image is not None else (0, 0)
            page_layout = PageLayout(id=file_id, page_size=page_size)

        if self.paths['input_logit']:
            logits_path = os.path.join(
                self.paths['input_logit'], file_id + '.logits'
            )
            page_layout.load_logits(logits_path)

        return page_layout

    def _save_outputs(self, page_layout: PageLayout, file_id: str) -> None:
        """Save all requested outputs."""
        # Save XML
        if self.paths['output_xml']:
            xml_path = os.path.join(self.paths['output_xml'], file_id + '.xml')
            page_layout.to_pagexml(xml_path)

        # Save render
        if self.paths['output_render']:
            image = self._load_image(None)  # Get current image
            if image is not None:
                page_layout.render_to_image(image)
                render_path = os.path.join(
                    self.paths['output_render'], file_id + '.jpg'
                )
                cv2.imwrite(
                    render_path,
                    image,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 70]
                )

        # Save logits
        if self.paths['output_logit']:
            logits_path = os.path.join(
                self.paths['output_logit'], file_id + '.logits'
            )
            page_layout.save_logits(logits_path)

        # Save ALTO
        if self.paths['output_alto']:
            alto_path = os.path.join(self.paths['output_alto'], file_id + '.xml')
            page_layout.to_altoxml(alto_path)

        # Save line images
        self._save_line_images(page_layout, file_id)

    def _save_line_images(self, page_layout: PageLayout, file_id: str) -> None:
        """Save individual line images."""
        if self.paths['output_line'] is None:
            return

        if 'lmdb' in self.paths['output_line']:
            writer = LmdbWriter(self.paths['output_line'])
            writer(page_layout, file_id)
        else:
            for region in page_layout.regions:
                for line in region.lines:
                    line_path = os.path.join(
                        self.paths['output_line'],
                        f'{file_id}-{line.id}.jpg'
                    )
                    cv2.imwrite(
                        line_path,
                        line.crop.astype(np.uint8),
                        [int(cv2.IMWRITE_JPEG_QUALITY), 98]
                    )

    def _extract_annotations(self, page_layout: PageLayout,
                             file_id: str) -> List[str]:
        """Extract transcription annotations."""
        all_lines = list(page_layout.lines_iterator())
        all_lines.sort(key=lambda x: x.id)
        annotations = []

        for line in all_lines:
            if line.transcription:
                key = f'{file_id}-{line.id}.jpg'
                annotations.append(f"{key} {line.transcription}")

        return annotations

    def _print_progress(self, index: int, total: int,
                        file_id: str, start_time: float) -> None:
        """Print processing progress."""
        elapsed = time.time() - start_time
        current = index + 1
        percentage = current / total * 100
        print(f"DONE {current}/{total} ({percentage:.2f} %) "
              f"[id: {file_id}] Time:{elapsed:.2f}")


class FileListManager:
    """Manages lists of files to process."""

    @staticmethod
    def prepare(input_image_path: Optional[str],
                input_xml_path: Optional[str],
                logger: logging.Logger) -> Tuple[List[str], List[Optional[str]]]:
        """Prepare lists of files to process."""
        if input_image_path is not None:
            logger.info('Reading images from %s.', input_image_path)
            return FileListManager._get_image_files(input_image_path)
        if input_xml_path is not None:
            logger.info('Reading page xml from %s', input_xml_path)
            return FileListManager._get_xml_files(input_xml_path)

        raise RuntimeError(
            'Either INPUT_IMAGE_PATH or INPUT_XML_PATH must be specified.'
        )

    @staticmethod
    def _get_image_files(path: str) -> Tuple[List[str], List[Optional[str]]]:
        """Get image files from directory."""
        ignored_extensions = {'', '.xml', '.logits'}
        images = [
            f for f in os.listdir(path)
            if os.path.splitext(f)[1].lower() not in ignored_extensions
        ]
        images.sort()
        ids = [os.path.splitext(os.path.basename(f))[0] for f in images]
        return ids, images

    @staticmethod
    def _get_xml_files(path: str) -> Tuple[List[str], List[Optional[str]]]:
        """Get XML files from directory."""
        xml_files = [
            f for f in os.listdir(path)
            if os.path.splitext(f)[1] == '.xml'
        ]
        ids = [os.path.splitext(os.path.basename(f))[0] for f in xml_files]
        return ids, [None] * len(ids)

    @staticmethod
    def filter_missing_xml(input_xml_path: str, ids: List[str],
                           images: List[Optional[str]]
                           ) -> Tuple[List[str], List[Optional[str]]]:
        """Filter files where XML is missing."""
        filtered_ids = []
        filtered_images = []

        for file_id, image_file in zip(ids, images):
            file_path = os.path.join(input_xml_path, file_id + '.xml')
            if os.path.exists(file_path):
                filtered_ids.append(file_id)
                filtered_images.append(image_file)

        return filtered_ids, filtered_images


class ProcessedFilesManager:
    """Manages already processed files."""

    @staticmethod
    def load(directories: List[Optional[str]]) -> Set[str]:
        """Load set of already processed files."""
        if not directories:
            return set()

        processed = None
        for directory in directories:
            if directory is None:
                continue

            files = ProcessedFilesManager._load_from_dir(directory)
            if processed is None:
                processed = files
            else:
                processed = processed.intersection(files)

        return processed or set()

    @staticmethod
    def _load_from_dir(directory: str) -> Set[str]:
        """Load processed files from a single directory."""
        pattern = re.compile(r"(.+?)(\.logits|\.xml|\.jpg)")
        processed = set()

        for file in os.listdir(directory):
            match = pattern.match(file)
            if match:
                processed.add(match.group(1))

        return processed


class MainProcessor:
    """Main processor orchestrating the entire workflow."""

    def __init__(self):
        self.args = None
        self.config = None
        self.file_processor = None
        self.ids_to_process = []
        self.images_to_process = []

    def run(self) -> None:
        """Run the main processing pipeline."""
        self.args = ArgumentParser.parse()
        self.config = ConfigManager(self.args.config)
        self.config.update_from_args(self.args)

        LoggingConfig.setup(self.config.config['PARSE_FOLDER'])
        logger = logging.getLogger()

        self.file_processor = FileProcessor(self.config)
        self.file_processor.initialize(self.args)

        self._prepare_processing()
        self._filter_files()
        self._process_files()

        self._print_summary(logger)

    def _prepare_processing(self) -> None:
        """Prepare for processing."""
        self.file_processor._check_compatibility()
        self.file_processor._create_output_dirs()
        self.file_processor._handle_logit_warning()

        paths = self.file_processor._paths
        logger = logging.getLogger()
        self.ids_to_process, self.images_to_process = (
            FileListManager.prepare(
                paths['input_image'],
                paths['input_xml'],
                logger
            )
        )

    def _filter_files(self) -> None:
        """Filter files based on various criteria."""
        paths = self.file_processor._paths
        logger = logging.getLogger()

        # Filter already processed
        if self.args.skip_processed:
            output_dirs = [
                paths['output_xml'],
                paths['output_logit'],
                paths['output_render'],
            ]
            processed = ProcessedFilesManager.load(output_dirs)
            if processed:
                logger.info("Already processed %s file(s).", len(processed))
                self._apply_filter(processed)

        # Filter missing XML
        if paths['input_xml'] and self.args.skipp_missing_xml:
            self.ids_to_process, self.images_to_process = (
                FileListManager.filter_missing_xml(
                    paths['input_xml'],
                    self.ids_to_process,
                    self.images_to_process
                )
            )

    def _apply_filter(self, exclude_set: Set[str]) -> None:
        """Apply filter to exclude files."""
        filtered = [
            (id_, img) for id_, img in zip(self.ids_to_process,
                                           self.images_to_process)
            if id_ not in exclude_set
        ]
        if filtered:
            self.ids_to_process, self.images_to_process = zip(*filtered)
        else:
            self.ids_to_process, self.images_to_process = [], []

    def _process_files(self) -> None:
        """Process all files."""
        computator = PageComputator(
            self.file_processor.page_parser,
            self.file_processor._paths
        )

        start_time = time.time()
        results = []

        if self.args.process_count > 1:
            results = self._process_parallel(computator)
        else:
            results = self._process_sequential(computator)

        self._save_transcriptions(results)
        self._print_processing_time(start_time)

    def _process_parallel(self, computator: PageComputator) -> List[List[str]]:
        """Process files in parallel."""
        with Pool(processes=self.args.process_count) as pool:
            tasks = [
                (img, file_id, idx, len(self.ids_to_process))
                for idx, (file_id, img) in enumerate(
                    zip(self.ids_to_process, self.images_to_process)
                )
            ]
            return pool.starmap(computator.process, tasks)

    def _process_sequential(self, computator: PageComputator) -> List[List[str]]:
        """Process files sequentially."""
        results = []
        for idx, (file_id, img) in enumerate(
            zip(self.ids_to_process, self.images_to_process)
        ):
            result = computator.process(
                img, file_id, idx, len(self.ids_to_process)
            )
            results.append(result)
        return results

    def _save_transcriptions(self, results: List[List[str]]) -> None:
        """Save transcription results to file."""
        if self.args.output_transcriptions_file_path:
            with open(self.args.output_transcriptions_file_path, 'w',
                      encoding='utf-8') as file:
                for page_lines in results:
                    if page_lines:
                        file.write('\n'.join(page_lines) + '\n')

    def _print_processing_time(self, start_time: float) -> None:
        """Print average processing time."""
        if self.ids_to_process:
            avg_time = (time.time() - start_time) / len(self.ids_to_process)
            logger = logging.getLogger()
            logger.info('AVERAGE PROCESSING TIME %s', avg_time)

    def _print_summary(self, logger: logging.Logger) -> None:
        """Print final summary."""
        if self.file_processor.page_parser.decoder:
            logger.info(self.file_processor.page_parser.decoder.decoding_summary())


def main() -> None:
    """Main entry point."""
    try:
        processor = MainProcessor()
        processor.run()
    except FileNotFoundError as error:
        print(f'ERROR: {error}')
        sys.exit(-1)
    except Exception as error:  # pylint: disable=broad-except
        print(f'ERROR: {error}')
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
