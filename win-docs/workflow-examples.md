# Workflow Examples (pero-ocr-win)

This document shows **practical end-to-end workflows** using PERO OCR on Windows
together with the **win-scripts** provided in this repository.

The focus is on:
- reproducibility
- ALTO correctness
- block / region preservation
- conversion to text

---

## Directory Convention

Example working directory:

```text
C:\temp\pero\
├─ in\                 # input images
│  └─ 0001.jpg
├─ out\                # OCR outputs
├─ models\
│  └─ pero_eu_cz_print_newspapers_2022-09-26\
│     └─ config.ini
└─ pero-ocr-win\
   ├─ win-scripts\
   └─ win-requirements.txt
```

---

## Workflow 1: Image → ALTO (Default PERO Output)

This is the **baseline OCR workflow**.

### Command

```powershell
python win-scripts\win-parse_folder.py `
  --config C:\temp\pero\models\pero_eu_cz_print_newspapers_2022-09-26\config.ini `
  --input-image-path C:\temp\pero\in `
  --output-xml-path C:\temp\pero\out `
  --device gpu
```

### Output

```text
out\
└─ 0001.xml        # ALTO XML (default, PERO-native)
```

Notes:
- This ALTO is usually **v2.x**
- Structure: Page → PrintSpace → TextBlock → TextLine → String
- Regions and reading order are preserved

---

## Workflow 2: ALTO Version Conversion

Convert PERO-generated ALTO into a **specific, strictly valid ALTO version**.

### Example: v2.1 → v4.4

```powershell
python win-scripts\win-alto_convert.py `
  C:\temp\pero\out\0001.xml `
  C:\temp\pero\out\0001.alto.v4_4.xml `
  --to v4.4 `
  --pretty
```

### Validation Only

```powershell
python win-scripts\win-alto_convert.py `
  C:\temp\pero\out\0001.alto.v4_4.xml `
  NUL `
  --to v4.4 `
  --validate-only
```

Supported targets:
- v2.0
- v2.1
- v3.0
- v3.1
- v4.0
- v4.1
- v4.2
- v4.3
- v4.4

---

## Workflow 3: ALTO → TXT (Block-Aware)

Extract text from ALTO **while respecting layout blocks**.

### Command

```powershell
python win-scripts\win-alto_to_txt.py `
  C:\temp\pero\out\0001.alto.v4_4.xml `
  C:\temp\pero\out\0001.txt `
  --paragraphs
```

### Result

```text
Paragraph 1 text line 1
Paragraph 1 text line 2

Paragraph 2 text line 1
Paragraph 2 text line 2
```

Rules:
- Empty line = block / region break
- Line order preserved
- Suitable for further NLP / indexing

---

## Workflow 4: PageXML → TXT

If PERO is configured to emit PageXML:

```powershell
python win-scripts\win-pagexml_to_txt.py `
  C:\temp\pero\out\0001.page.xml `
  C:\temp\pero\out\0001.txt `
  --paragraphs
```

Use when:
- Working with PAGE-based pipelines
- Integrating with Transkribus-like tools

---

## Workflow 5: Batch Processing

### Convert all ALTO files to v4.4

```powershell
for %f in (C:\temp\pero\out\*.xml) do (
  python win-scripts\win-alto_convert.py "%f" "%f.v4_4.xml" --to v4.4
)
```

### Convert all ALTO v4.4 to TXT

```powershell
for %f in (C:\temp\pero\out\*.v4_4.xml) do (
  python win-scripts\win-alto_to_txt.py "%f" "%f.txt" --paragraphs
)
```

---

## Recommended Archival Workflow

1. OCR → **native PERO ALTO**
2. Convert → **ALTO v4.4 (strict)**
3. Validate
4. Derive TXT / derivatives
5. Preserve original + converted ALTO

---

## Notes

- Always keep the **original PERO ALTO**
- Conversion is deterministic and reversible
- Scripts are Windows-safe (no fcntl, no safe_gpu)

---

Next document:
**README-upstream.md**
