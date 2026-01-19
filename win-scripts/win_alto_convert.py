from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import requests
from lxml import etree


@dataclass(frozen=True)
class AltoSpec:
    key: str                 # e.g. "v4.4"
    ns: str                  # namespace URI
    xsd_url: str             # LoC schema URL
    schemaversion: Optional[str]  # SCHEMAVERSION value (v3/v4), None for v2


NS_V2 = "http://www.loc.gov/standards/alto/ns-v2#"
NS_V3 = "http://www.loc.gov/standards/alto/ns-v3#"
NS_V4 = "http://www.loc.gov/standards/alto/ns-v4#"

# Explicit, versioned LoC XSDs (stable targets)
SPECS: Dict[str, AltoSpec] = {
    "v2.0": AltoSpec("v2.0", NS_V2, "https://www.loc.gov/standards/alto/v2/alto-2-0.xsd", None),
    "v2.1": AltoSpec("v2.1", NS_V2, "https://www.loc.gov/standards/alto/v2/alto-2-1.xsd", None),

    "v3.0": AltoSpec("v3.0", NS_V3, "https://www.loc.gov/standards/alto/v3/alto-3-0.xsd", "3.0"),
    "v3.1": AltoSpec("v3.1", NS_V3, "https://www.loc.gov/standards/alto/v3/alto-3-1.xsd", "3.1"),

    "v4.0": AltoSpec("v4.0", NS_V4, "https://www.loc.gov/standards/alto/v4/alto-4-0.xsd", "4.0"),
    "v4.1": AltoSpec("v4.1", NS_V4, "https://www.loc.gov/standards/alto/v4/alto-4-1.xsd", "4.1"),
    "v4.2": AltoSpec("v4.2", NS_V4, "https://www.loc.gov/standards/alto/v4/alto-4-2.xsd", "4.2"),
    "v4.3": AltoSpec("v4.3", NS_V4, "https://www.loc.gov/standards/alto/v4/alto-4-3.xsd", "4.3"),
    "v4.4": AltoSpec("v4.4", NS_V4, "https://www.loc.gov/standards/alto/v4/alto-4-4.xsd", "4.4"),
}

XSI = "http://www.w3.org/2001/XMLSchema-instance"


def _cache_dir() -> str:
    # Windows-friendly cache location
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    d = os.path.join(base, "alto_xsd_cache")
    os.makedirs(d, exist_ok=True)
    return d


def _cache_path_for(url: str) -> str:
    # simple deterministic filename
    safe = url.replace("://", "_").replace("/", "_").replace("?", "_")
    return os.path.join(_cache_dir(), safe)


def download_xsd(url: str) -> bytes:
    p = _cache_path_for(url)
    if os.path.exists(p) and os.path.getsize(p) > 0:
        with open(p, "rb") as f:
            return f.read()

    r = requests.get(url, timeout=60)
    r.raise_for_status()
    data = r.content

    with open(p, "wb") as f:
        f.write(data)
    return data


def build_schema(xsd_bytes: bytes) -> etree.XMLSchema:
    # parse XSD (bytes -> element)
    xsd_doc = etree.XML(xsd_bytes)
    return etree.XMLSchema(xsd_doc)


def validate(tree: etree._ElementTree, schema: etree.XMLSchema) -> Tuple[bool, str]:
    ok = schema.validate(tree)
    if ok:
        return True, ""
    return False, str(schema.error_log)


def rewrite_namespace_inplace(tree: etree._ElementTree, target_ns: str) -> None:
    """
    Rewrite ALL element namespaces to target_ns (keeps localnames).
    Then cleanup namespaces so we don't end up with ns0/ns1 garbage.
    """
    root = tree.getroot()

    for el in root.iter():
        if not isinstance(el.tag, str):
            continue
        local = etree.QName(el).localname
        el.tag = f"{{{target_ns}}}{local}"

    # remove old xmlns declarations & re-normalize
    etree.cleanup_namespaces(root)


def ensure_schema_location(root: etree._Element, spec: AltoSpec) -> None:
    # xsi:schemaLocation="<ns> <xsd_url>"
    root.set(etree.QName(XSI, "schemaLocation"), f"{spec.ns} {spec.xsd_url}")


def ensure_schemaversion(root: etree._Element, spec: AltoSpec) -> None:
    # v3/v4 commonly carry SCHEMAVERSION on root; v2 does not.
    if spec.schemaversion is None:
        if "SCHEMAVERSION" in root.attrib:
            del root.attrib["SCHEMAVERSION"]
        return
    root.set("SCHEMAVERSION", spec.schemaversion)


def detect_ns(tree: etree._ElementTree) -> str:
    root = tree.getroot()
    if isinstance(root.tag, str) and root.tag.startswith("{"):
        return root.tag.split("}")[0][1:]
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_xml", help="Input ALTO XML")
    ap.add_argument("output_xml",
                    help="Output ALTO XML (use NUL on Windows for validate-only)")
    ap.add_argument("--to", required=True, choices=list(SPECS.keys()),
                    help="Target ALTO version")
    ap.add_argument("--validate-only", action="store_true",
                    help="Only validate (no conversion)")
    ap.add_argument("--pretty", action="store_true", help="Pretty-print XML")
    ap.add_argument("--no-cache", action="store_true",
                    help="Do not use cached XSDs (force re-download)")
    args = ap.parse_args()

    spec = SPECS[args.to]

    if args.no_cache:
        # wipe only this xsd from cache (optional)
        p = _cache_path_for(spec.xsd_url)
        if os.path.exists(p):
            os.remove(p)

    parser = etree.XMLParser(remove_blank_text=False, huge_tree=True)
    tree = etree.parse(args.input_xml, parser)
    root = tree.getroot()

    src_ns = detect_ns(tree)
    if not src_ns:
        print("[WARN] Input has no namespace; converter will still try to rewrite.")

    if not args.validate_only:
        # 1) rewrite element namespaces cleanly
        rewrite_namespace_inplace(tree, spec.ns)

        # 2) set schemaLocation + SCHEMAVERSION rules
        root = tree.getroot()
        ensure_schema_location(root, spec)
        ensure_schemaversion(root, spec)

        # 3) cleanup again after attribute changes
        etree.cleanup_namespaces(root)

    # validate against target XSD
    xsd_bytes = download_xsd(spec.xsd_url)
    schema = build_schema(xsd_bytes)
    ok, msg = validate(tree, schema)

    if not ok:
        print(f"[INVALID] target={spec.key}")
        print(msg)
        raise SystemExit(2)

    print(f"[VALID] target={spec.key}")

    if not args.validate_only:
        tree.write(
            args.output_xml,
            encoding="UTF-8",
            xml_declaration=True,
            pretty_print=args.pretty,
        )
        print(f"Wrote: {args.output_xml}")


if __name__ == "__main__":
    main()
