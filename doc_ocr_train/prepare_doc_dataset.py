import argparse
import csv
import html
import json
import random
import re
import zipfile
from pathlib import Path


IMAGE_RE = re.compile(r"[A-Za-z]:\\.*?encarch_198\\files\\.*?\.jpg_?", re.IGNORECASE)
TEXT_RE = re.compile(r"<w:t[^>]*>(.*?)</w:t>", re.DOTALL)
TAG_RE = re.compile(r"<[^>]+>")


def clean_text(text: str) -> str:
    text = html.unescape(text)
    text = text.replace("\u00a0", " ")
    text = TAG_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_docx(docx_path: Path) -> dict:
    with zipfile.ZipFile(docx_path) as zf:
        xml = zf.read("word/document.xml").decode("utf-8", errors="ignore")

    chunks = TEXT_RE.findall(xml)
    full_text = clean_text(" ".join(chunks))
    image_match = IMAGE_RE.search(full_text)

    sample_id = docx_path.stem.replace("_table", "")
    record = {
        "sample_id": sample_id,
        "source_doc": str(docx_path),
        "image_path_raw": image_match.group(0).rstrip("_") if image_match else "",
        "text_raw": full_text,
    }
    return record


def split_records(records, seed: int, train_ratio: float, val_ratio: float):
    rng = random.Random(seed)
    items = list(records)
    rng.shuffle(items)

    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = items[:n_train]
    val = items[n_train:n_train + n_val]
    test = items[n_train + n_val:]
    return train, val, test


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare OCR dataset from DOCX table files."
    )
    parser.add_argument("--input-dir", required=True, help="Folder with *_table.docx files")
    parser.add_argument("--output-dir", required=True, help="Output folder for metadata/splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--min-chars", type=int, default=30)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    docx_files = sorted(input_dir.glob("*.docx"))
    if not docx_files:
        raise FileNotFoundError(f"No DOCX files found in: {input_dir}")

    rows = []
    for fp in docx_files:
        row = parse_docx(fp)
        if len(row["text_raw"]) >= args.min_chars:
            rows.append(row)

    train, val, test = split_records(rows, args.seed, args.train_ratio, args.val_ratio)

    write_jsonl(output_dir / "all.jsonl", rows)
    write_jsonl(output_dir / "train.jsonl", train)
    write_jsonl(output_dir / "val.jsonl", val)
    write_jsonl(output_dir / "test.jsonl", test)

    write_csv(output_dir / "all.csv", rows)

    summary = {
        "total_docx": len(docx_files),
        "kept_samples": len(rows),
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "seed": args.seed,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
