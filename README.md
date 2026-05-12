# OCR_HTR

Проект обучения и прогонов OCR для документов: подготовка датасета и fine-tuning (Hugging Face).

## Структура

- `doc_ocr_train/` — скрипты (`prepare_doc_dataset.py`, `train_hf_docx_ocr.py`), датасет `dataset_docx_only/`, артефакты прогонов `hf_runs_multi/`.
- `тестовые данные/` — тестовые материалы.

## Требования

См. импорты в скриптах обучения (Python, зависимости для HF/transformers по необходимости).
