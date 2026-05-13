#!/usr/bin/env python3
"""
Сравнение с другими *лёгкими* моделями TrOCR (тот же класс, что train_hf_docx_ocr.py:
TrOCRProcessor + VisionEncoderDecoderModel, порядок памяти как у microsoft/trocr-small-*).

Не включает microsoft/trocr-small-printed и microsoft/trocr-small-handwritten —
их запускайте через train_hf_docx_ocr.py по умолчанию или --models вручную.

Типы (для ориентира на русском датасете):
  OCR  — заточены под печать / строки (латиница, цифры, и т.д.).
  HTR  — рукопись (часто под другой язык; метрики на русском могут быть хуже).
  base — промежуточный претрейн Microsoft (не финальный SROIE/IAM).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# id -> (OCR | HTR | BASE, краткий комментарий)
LIGHT_MODEL_INFO: dict[str, tuple[str, str]] = {
    "microsoft/trocr-small-stage1": (
        "BASE",
        "Промежуточный этап TrOCR small (до fine-tune на печать/рукопись IAM/SROIE)",
    ),
    "vukpetar/trocr-small-photomath": (
        "OCR",
        "Печать / математические строки (Photomath)",
    ),
    "qantev/trocr-small-spanish": (
        "OCR",
        "Печать, испанский (малый TrOCR)",
    ),
}

DEFAULT_MODEL_ORDER = list(LIGHT_MODEL_INFO.keys())


def main() -> None:
    root = Path(__file__).resolve().parent
    trainer = root / "train_hf_docx_ocr.py"

    parser = argparse.ArgumentParser(
        description="Запуск train_hf_docx_ocr.py на других лёгких TrOCR-моделях (не small-printed/handwritten)."
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODEL_ORDER,
        help="Список HF model id (по умолчанию — LIGHT_MODEL_INFO).",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="Только показать команду, не запускать.")
    args, passthrough = parser.parse_known_args()

    print("Модели для прогона (тип — ориентир OCR/HTR/BASE):\n")
    for mid in args.models:
        kind, note = LIGHT_MODEL_INFO.get(mid, ("?", "вне встроенного справочника — проверьте карточку на HF"))
        print(f"  [{kind:4}] {mid}\n       {note}\n")

    cmd: list[str] = [
        sys.executable,
        str(trainer),
        "--input-dir",
        args.input_dir,
        "--output-dir",
        args.output_dir,
        "--models",
        *args.models,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        *passthrough,
    ]

    print("Команда:\n ", " ".join(cmd), "\n")

    if args.dry_run:
        return

    subprocess.run(cmd, check=True, cwd=str(root))


if __name__ == "__main__":
    main()
