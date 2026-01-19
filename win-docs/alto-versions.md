# ALTO XML Versions – Conversion and Compatibility

This document describes how ALTO XML versions are handled in **pero-ocr-win**,
which versions are supported, and what guarantees the conversion scripts provide.

The goal is **strictly valid ALTO XML** across multiple versions, suitable for
library-grade workflows (NDK, Kramerius, DL ingestion pipelines, long-term preservation).

---

## Supported ALTO Versions

The following ALTO versions are supported for **conversion, validation, and round-trips**:

| Version | Namespace |
|-------|-----------|
| ALTO 2.0 | `http://www.loc.gov/standards/alto/ns-v2#` |
| ALTO 2.1 | `http://www.loc.gov/standards/alto/ns-v2#` |
| ALTO 3.0 | `http://www.loc.gov/standards/alto/ns-v3#` |
| ALTO 3.1 | `http://www.loc.gov/standards/alto/ns-v3#` |
| ALTO 4.0 | `http://www.loc.gov/standards/alto/ns-v4#` |
| ALTO 4.1 | `http://www.loc.gov/standards/alto/ns-v4#` |
| ALTO 4.2 | `http://www.loc.gov/standards/alto/ns-v4#` |
| ALTO 4.3 | `http://www.loc.gov/standards/alto/ns-v4#` |
| ALTO 4.4 | `http://www.loc.gov/standards/alto/ns-v4#` |

---

## Default Output from PERO OCR

By default, **PERO OCR** produces:

- **ALTO 2.1–compatible structure**
- Namespace: `ns-v2`
- Elements:
  - `Page`
  - `PrintSpace`
  - `TextBlock`
  - `TextLine`
  - `String`, `SP`
- Coordinates in **pixel units**

This output is **valid ALTO v2.x**, but lacks newer semantic features
introduced in ALTO 3.x and 4.x.

---

## Conversion Script

All ALTO version conversions are handled by:

```
win-scripts/win-alto_convert.py
```

### Key properties

- XML namespace rewriting
- Version-specific attribute filtering
- Schema-aware validation
- Optional pretty-printing
- No OCR content loss

---

## Basic Usage

### Convert ALTO 2.x → ALTO 4.4

```powershell
python win-scripts/win-alto_convert.py `
  input.xml `
  output.alto.v4_4.xml `
  --to v4.4 `
  --pretty
```

### Validate ALTO without writing output

```powershell
python win-scripts/win-alto_convert.py `
  output.alto.v4_4.xml `
  NUL `
  --to v4.4 `
  --validate-only
```

---

## Supported Conversion Directions

All of the following are supported:

- v2.0 ↔ v2.1
- v2.x ↔ v3.x
- v3.x ↔ v4.x
- v4.x ↔ v4.x (normalization)
- **Round-trip safety** (no structural damage)

Example:

```text
v2.1 → v4.4 → v3.1 → v2.1
```

---

## What Changes Between Versions

### ALTO 2.x

- Minimal semantic structure
- No roles, no styles
- Best for legacy systems

### ALTO 3.x

- Improved logical consistency
- Transitional model
- Still compatible with many older DL systems

### ALTO 4.x

- Richer semantics
- Better extensibility
- Recommended for new projects

---

## What Is NOT Added Automatically

The converter **does NOT invent**:

- `TextStyle`
- `LayoutStyle`
- `Role` attributes
- Reading order
- Logical structure (`ComposedBlock`, `Paragraph`)

These require **layout / semantic post-processing**, not OCR output.

---

## Validation Guarantees

For every target version:

- Correct XML namespace
- Version-appropriate attributes only
- Schema-valid structure
- Compatible with:
  - ALTO XSD
  - Kramerius ingest
  - Long-term archival workflows

---

## Recommended Practice

| Use case | Recommended ALTO |
|--------|------------------|
| Raw OCR output | v2.1 |
| Kramerius / NDK | v2.1 or v3.1 |
| New DL projects | v4.4 |
| Interop testing | v3.1 |
| Preservation master | v4.4 |

---

## Related Scripts

- `win-alto_to_txt.py` – ALTO → TXT (preserves blocks & empty lines)
- `win-pagexml_to_txt.py` – PageXML → TXT
- `win-parse_folder.py` – Windows-safe PERO OCR runner

---

## Summary

- PERO OCR outputs valid ALTO 2.x
- `win-alto_convert.py` enables **strict multi-version ALTO**
- All conversions are **lossless**
- Validation is first-class
- Suitable for professional digitization pipelines
