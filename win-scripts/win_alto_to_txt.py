import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


def detect_alto_ns(root: ET.Element) -> str:
    # Root tag looks like "{namespace}alto"
    if root.tag.startswith("{") and "}" in root.tag:
        return root.tag.split("}")[0][1:]
    return ""  # no namespace (unlikely for ALTO)


def extract_text(alto_path: Path) -> str:
    tree = ET.parse(alto_path)
    root = tree.getroot()
    ns_uri = detect_alto_ns(root)

    ns = {"a": ns_uri} if ns_uri else {}
    def xp(tag: str) -> str:
        return f"a:{tag}" if ns_uri else tag

    lines_out = []

    # Blocks in reading order as present in XML
    for block in root.findall(f".//{xp('TextBlock')}", ns):
        block_lines = []
        for tl in block.findall(f"./{xp('TextLine')}", ns):
            parts = []
            # ALTO commonly uses <String/> for words + <SP/> for spaces
            for child in list(tl):
                local = child.tag.split("}")[-1]  # strip ns
                if local == "String":
                    txt = child.attrib.get("CONTENT", "")
                    if txt:
                        parts.append(txt)
                elif local == "SP":
                    # optional: explicit space
                    if parts and not parts[-1].endswith(" "):
                        parts.append(" ")
            # Normalize: join and collapse repeated spaces
            line = "".join(parts).strip()
            if line:
                block_lines.append(line)

        if block_lines:
            lines_out.extend(block_lines)
            lines_out.append("")  # blank line between blocks

    # Remove trailing blank lines
    while lines_out and lines_out[-1] == "":
        lines_out.pop()

    return "\n".join(lines_out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("alto_xml", help="Path to ALTO XML")
    ap.add_argument("-o", "--out", help="Output TXT path (default: same name .txt)")
    args = ap.parse_args()

    in_path = Path(args.alto_xml)
    out_path = Path(args.out) if args.out else in_path.with_suffix(".txt")

    text = extract_text(in_path)
    out_path.write_text(text, encoding="utf-8")
    print(f"OK: wrote {out_path} ({len(text)} chars)")


if __name__ == "__main__":
    main()
