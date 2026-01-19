import argparse
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Optional, Tuple

WS_RE = re.compile(r"\s+")


def norm_ws(s: str) -> str:
    return WS_RE.sub(" ", s).strip()


def local_name(tag: str) -> str:
    # "{namespace}TextRegion" -> "TextRegion"
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def parse_points(points: Optional[str]) -> List[Tuple[int, int]]:
    # "x,y x,y ..." -> [(x,y),...]
    if not points:
        return []
    out = []
    for pair in points.strip().split():
        if "," not in pair:
            continue
        x, y = pair.split(",", 1)
        try:
            out.append((int(float(x)), int(float(y))))
        except (ValueError, TypeError):
            pass
    return out


def bbox_from_points(pts: List[Tuple[int, int]]) -> Optional[Tuple[int, int, int, int]]:
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))  # (x0,y0,x1,y1)


def find_child(el: ET.Element, wanted_local: str) -> Optional[ET.Element]:
    for ch in list(el):
        if local_name(ch.tag) == wanted_local:
            return ch
    return None


def iter_desc(el: ET.Element, wanted_local: str):
    for n in el.iter():
        if local_name(n.tag) == wanted_local:
            yield n


def get_textline_text(tl: ET.Element) -> str:
    # Prefer TextEquiv/Unicode (PAGE)
    # Some PAGE files have multiple TextEquiv; take the first with Unicode content.
    for te in iter_desc(tl, "TextEquiv"):
        uni = find_child(te, "Unicode")
        if uni is not None and (uni.text or "").strip():
            return norm_ws(uni.text or "")
    return ""


def get_line_y(tl: ET.Element) -> int:
    # Prefer Baseline y (more stable), fallback to Coords bbox top
    bl = find_child(tl, "Baseline")
    if bl is not None:
        pts = parse_points(bl.attrib.get("points"))
        bb = bbox_from_points(pts)
        if bb:
            return bb[1]  # top y
    coords = find_child(tl, "Coords")
    if coords is not None:
        pts = parse_points(coords.attrib.get("points"))
        bb = bbox_from_points(pts)
        if bb:
            return bb[1]
    return 0


def get_line_x(tl: ET.Element) -> int:
    coords = find_child(tl, "Coords")
    if coords is not None:
        pts = parse_points(coords.attrib.get("points"))
        bb = bbox_from_points(pts)
        if bb:
            return bb[0]
    return 0


@dataclass
class Line:
    y: int
    x: int
    text: str


@dataclass
class Region:
    id: str
    x0: int
    y0: int
    x1: int
    y1: int
    lines: List[Line]


def _compute_region_bbox(tr: ET.Element) -> Tuple[int, int, int, int]:
    """Compute bounding box for a TextRegion element."""
    coords = find_child(tr, "Coords")
    if coords is not None:
        bb = bbox_from_points(parse_points(coords.attrib.get("points")))
        if bb:
            return bb

    # fallback bbox from lines if region coords missing
    xs, ys, xe, ye = [], [], [], []
    for tl in iter_desc(tr, "TextLine"):
        c = find_child(tl, "Coords")
        if c is None:
            continue
        b = bbox_from_points(parse_points(c.attrib.get("points")))
        if not b:
            continue
        xs.append(b[0])
        ys.append(b[1])
        xe.append(b[2])
        ye.append(b[3])

    if xs:
        return (min(xs), min(ys), max(xe), max(ye))
    return (0, 0, 0, 0)


def _extract_lines_from_region(tr: ET.Element) -> List[Line]:
    """Extract all text lines from a TextRegion element."""
    lines: List[Line] = []
    for tl in iter_desc(tr, "TextLine"):
        txt = get_textline_text(tl)
        if not txt:
            continue
        lines.append(Line(y=get_line_y(tl), x=get_line_x(tl), text=txt))
    return lines


def extract_regions(root: ET.Element) -> List[Region]:
    # PAGE root: PcGts/Page
    page = None
    for n in root.iter():
        if local_name(n.tag) == "Page":
            page = n
            break
    if page is None:
        raise RuntimeError("Nenašel jsem element <Page> v PAGE XML.")

    regions: List[Region] = []
    for tr in iter_desc(page, "TextRegion"):
        rid = tr.attrib.get("id", "")
        x0, y0, x1, y1 = _compute_region_bbox(tr)
        lines = _extract_lines_from_region(tr)

        if lines:
            regions.append(Region(id=rid, x0=x0, y0=y0, x1=x1, y1=y1, lines=lines))

    return regions


def sort_regions_reading_order(regions: List[Region]) -> List[Region]:
    """
    Heuristika pro noviny:
    - regiony seskupíme do "řádků" podle y0 s tolerancí (~0.6 * výška regionu)
    - uvnitř řádku left-to-right podle x0
    - řádky top-to-bottom
    """
    if not regions:
        return []

    regs = sorted(regions, key=lambda r: (r.y0, r.x0))
    rows: List[List[Region]] = []
    for r in regs:
        placed = False
        for row in rows:
            ref = row[0]
            ref_h = max(1, ref.y1 - ref.y0)
            r_h = max(1, r.y1 - r.y0)
            tol = max(15, int(0.6 * max(ref_h, r_h)))
            if abs(r.y0 - ref.y0) <= tol:
                row.append(r)
                placed = True
                break
        if not placed:
            rows.append([r])

    for row in rows:
        row.sort(key=lambda r: r.x0)
    rows.sort(key=lambda row: min(r.y0 for r in row))

    out: List[Region] = []
    for row in rows:
        out.extend(row)
    return out


def render_text(regions: List[Region], blank_line_between_regions: bool = True) -> str:
    out_lines: List[str] = []
    for reg in regions:
        # řádky v regionu shora dolů, druhotně zleva
        reg.lines.sort(key=lambda ln: (ln.y, ln.x))
        for ln in reg.lines:
            out_lines.append(ln.text)
        if blank_line_between_regions:
            out_lines.append("")  # prázdný řádek mezi bloky
    while out_lines and out_lines[-1] == "":
        out_lines.pop()
    return "\n".join(out_lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pagexml", help="Cesta k PAGE XML (např. 1.xml)")
    ap.add_argument("-o", "--out",
                    help="Výstupní TXT (default: stejné jméno .txt vedle XML)")
    ap.add_argument("--no-blank-lines", action="store_true",
                    help="Nevkládat prázdný řádek mezi regiony")
    args = ap.parse_args()

    in_path = args.pagexml
    out_path = args.out or (os.path.splitext(in_path)[0] + ".txt")

    tree = ET.parse(in_path)
    root = tree.getroot()

    regions = extract_regions(root)
    regions = sort_regions_reading_order(regions)

    text = render_text(regions, blank_line_between_regions=not args.no_blank_lines)

    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)

    print(f"OK: {out_path} (regions={len(regions)})")


if __name__ == "__main__":
    main()
