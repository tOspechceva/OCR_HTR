import argparse
import inspect
import json
import random
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

import evaluate
import numpy as np
import torch
from PIL import Image
from datasets import Dataset, DatasetDict
from docx import Document
from docx.oxml.ns import qn
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    default_data_collator,
)


DEFAULT_MODELS = [
    "microsoft/trocr-small-printed",
    "philschmid/trocr-base-printed",
    "kazars24/trocr-base-handwritten-ru",
    "cyrillic-trocr/trocr-handwritten-cyrillic",
    "taiga75/ru-trocr-1700s",
]


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def guess_ext(blob: bytes) -> str:
    if blob.startswith(b"\x89PNG"):
        return ".png"
    if blob.startswith(b"\xff\xd8"):
        return ".jpg"
    if blob.startswith(b"GIF8"):
        return ".gif"
    return ".png"


def extract_first_image_blob(cell) -> Optional[bytes]:
    drawing_elems = cell._tc.xpath(".//w:drawing")
    if not drawing_elems:
        return None

    blips = drawing_elems[0].xpath(".//a:blip")
    if not blips:
        return None

    rel_id = blips[0].get(qn("r:embed"))
    if not rel_id:
        return None

    image_part = cell.part.related_parts.get(rel_id)
    if image_part is None:
        return None

    return image_part.blob


def parse_docx_tables(docx_path: Path, image_out_dir: Path) -> List[Dict]:
    doc = Document(str(docx_path))
    records = []

    for table_idx, table in enumerate(doc.tables):
        for row_idx, row in enumerate(table.rows):
            cells = row.cells
            if len(cells) < 3:
                continue

            header_probe = normalize_text(cells[1].text).lower()
            if "название файла" in header_probe:
                continue

            translation = normalize_text(cells[2].text) if len(cells) >= 3 else ""
            seg_error = normalize_text(cells[3].text) if len(cells) >= 4 else ""
            script_type = normalize_text(cells[4].text) if len(cells) >= 5 else ""
            source_file_name = normalize_text(cells[1].text) if len(cells) >= 2 else ""

            if not translation:
                continue

            blob = extract_first_image_blob(cells[0])
            if blob is None:
                continue

            ext = guess_ext(blob)
            out_name = f"{docx_path.stem}_t{table_idx}_r{row_idx}{ext}"
            out_path = image_out_dir / out_name

            try:
                img = Image.open(BytesIO(blob)).convert("RGB")
                img.save(out_path)
            except Exception:
                continue

            records.append(
                {
                    "image_path": str(out_path),
                    "text": translation,
                    "docx_file": str(docx_path),
                    "source_file_name": source_file_name,
                    "segmentation_error": seg_error,
                    "script_type": script_type,
                }
            )

    return records


def build_dataset_from_docx(input_dir: Path, work_dir: Path, min_chars: int) -> Dataset:
    image_out_dir = work_dir / "extracted_images"
    image_out_dir.mkdir(parents=True, exist_ok=True)

    all_records: List[Dict] = []
    docx_files = sorted(input_dir.glob("*.docx"))
    if not docx_files:
        raise FileNotFoundError(f"No .docx files found in {input_dir}")

    for docx_path in docx_files:
        all_records.extend(parse_docx_tables(docx_path, image_out_dir))

    filtered = []
    for r in all_records:
        txt = normalize_text(r["text"])
        if len(txt) < min_chars:
            continue
        r["text"] = txt
        filtered.append(r)

    if not filtered:
        raise RuntimeError("No usable records after parsing/filtering.")

    manifest_path = work_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for r in filtered:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return Dataset.from_list(filtered)


def split_dataset(
    dataset: Dataset, test_size: float, val_size_within_train: float, seed: int
) -> DatasetDict:
    train_test = dataset.train_test_split(test_size=test_size, seed=seed)
    train_val = train_test["train"].train_test_split(test_size=val_size_within_train, seed=seed)
    return DatasetDict(
        {
            "train": train_val["train"],
            "validation": train_val["test"],
            "test": train_test["test"],
        }
    )


@dataclass
class OCRRunner:
    model_name: str
    output_root: Path
    max_target_length: int
    batch_size: int
    num_train_epochs: int
    learning_rate: float
    fp16: bool

    def train_and_eval(self, dataset_dict: DatasetDict) -> Dict:
        run_dir = self.output_root / self.model_name.replace("/", "__")
        run_dir.mkdir(parents=True, exist_ok=True)

        processor = TrOCRProcessor.from_pretrained(self.model_name)
        model = VisionEncoderDecoderModel.from_pretrained(self.model_name)

        decoder_start_id = (
            processor.tokenizer.cls_token_id
            if processor.tokenizer.cls_token_id is not None
            else processor.tokenizer.bos_token_id
        )
        eos_id = (
            processor.tokenizer.sep_token_id
            if processor.tokenizer.sep_token_id is not None
            else processor.tokenizer.eos_token_id
        )
        pad_id = (
            processor.tokenizer.pad_token_id
            if processor.tokenizer.pad_token_id is not None
            else processor.tokenizer.eos_token_id
        )
        if decoder_start_id is None or eos_id is None or pad_id is None:
            raise ValueError(
                f"Tokenizer for {self.model_name} misses required special tokens "
                "(need decoder_start/eos/pad ids)."
            )

        model.config.decoder_start_token_id = decoder_start_id
        model.config.pad_token_id = pad_id
        model.config.eos_token_id = eos_id
        model.config.max_length = self.max_target_length
        model.config.early_stopping = True
        model.config.no_repeat_ngram_size = 0
        model.config.length_penalty = 1.0
        model.config.num_beams = 4

        def preprocess(batch):
            images = [Image.open(p).convert("RGB") for p in batch["image_path"]]
            pixel_values = processor(images=images, return_tensors="pt").pixel_values
            labels = processor.tokenizer(
                batch["text"],
                padding="max_length",
                max_length=self.max_target_length,
                truncation=True,
            ).input_ids
            labels = [
                [(token if token != processor.tokenizer.pad_token_id else -100) for token in label]
                for label in labels
            ]
            return {"pixel_values": pixel_values, "labels": labels}

        tokenized = dataset_dict.map(
            preprocess,
            batched=True,
            remove_columns=dataset_dict["train"].column_names,
            batch_size=self.batch_size,
        )

        cer_metric = evaluate.load("cer")
        wer_metric = evaluate.load("wer")

        def compute_metrics(eval_pred):
            pred_ids = eval_pred.predictions
            label_ids = eval_pred.label_ids
            label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

            pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
            label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
            pred_str = [normalize_text(x) for x in pred_str]
            label_str = [normalize_text(x) for x in label_str]

            cer = cer_metric.compute(predictions=pred_str, references=label_str)
            wer = wer_metric.compute(predictions=pred_str, references=label_str)
            return {"cer": cer, "wer": wer}

        args = Seq2SeqTrainingArguments(
            output_dir=str(run_dir / "checkpoints"),
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            predict_with_generate=True,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=20,
            save_total_limit=2,
            num_train_epochs=self.num_train_epochs,
            learning_rate=self.learning_rate,
            warmup_ratio=0.1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_cer",
            greater_is_better=False,
            fp16=self.fp16,
            report_to=[],
        )

        # Transformers 5.x: Trainer uses processing_class; older versions used tokenizer.
        trainer_kw = dict(
            model=model,
            args=args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["validation"],
            data_collator=default_data_collator,
            compute_metrics=compute_metrics,
        )
        _trainer_sig = inspect.signature(Seq2SeqTrainer.__init__)
        if "processing_class" in _trainer_sig.parameters:
            trainer_kw["processing_class"] = processor.tokenizer
        else:
            trainer_kw["tokenizer"] = processor.tokenizer
        trainer = Seq2SeqTrainer(**trainer_kw)

        trainer.train()
        test_metrics = trainer.evaluate(eval_dataset=tokenized["test"], metric_key_prefix="test")

        model_save_dir = run_dir / "best_model"
        trainer.save_model(str(model_save_dir))
        processor.save_pretrained(str(model_save_dir))

        out = {
            "model_name": self.model_name,
            "test_cer": float(test_metrics.get("test_cer", np.nan)),
            "test_wer": float(test_metrics.get("test_wer", np.nan)),
            "test_loss": float(test_metrics.get("test_loss", np.nan)),
            "num_train": len(dataset_dict["train"]),
            "num_val": len(dataset_dict["validation"]),
            "num_test": len(dataset_dict["test"]),
            "model_dir": str(model_save_dir),
        }
        (run_dir / "metrics.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return out


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate multiple HF OCR models from DOCX tables."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory with DOCX files (table format).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for extracted data, checkpoints, and metrics.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="HF model names to train/evaluate.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--val-size-within-train",
        type=float,
        default=0.2,
        help="Validation share inside train split.",
    )
    parser.add_argument("--min-chars", type=int, default=1)
    parser.add_argument("--max-target-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset_from_docx(
        input_dir=input_dir, work_dir=output_dir, min_chars=args.min_chars
    )
    dataset_dict = split_dataset(
        dataset=dataset,
        test_size=args.test_size,
        val_size_within_train=args.val_size_within_train,
        seed=args.seed,
    )

    all_results = []
    for model_name in args.models:
        runner = OCRRunner(
            model_name=model_name,
            output_root=output_dir / "runs",
            max_target_length=args.max_target_length,
            batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            fp16=args.fp16,
        )
        try:
            result = runner.train_and_eval(dataset_dict)
            all_results.append(result)
            print(f"[OK] {model_name}: CER={result['test_cer']:.4f}, WER={result['test_wer']:.4f}")
        except Exception as e:
            fail = {"model_name": model_name, "error": str(e)}
            all_results.append(fail)
            print(f"[FAIL] {model_name}: {e}")

    leaderboard = sorted(
        [r for r in all_results if "test_cer" in r],
        key=lambda x: x["test_cer"],
    )
    summary = {
        "dataset_size": len(dataset),
        "splits": {
            "train": len(dataset_dict["train"]),
            "validation": len(dataset_dict["validation"]),
            "test": len(dataset_dict["test"]),
        },
        "results": all_results,
        "leaderboard_by_cer": leaderboard,
    }
    (output_dir / "summary_results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
