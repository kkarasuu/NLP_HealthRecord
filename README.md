# Medical Report Summarizer

This project extracts, classifies, and summarizes medical data from PDF-based Electronic Health Records (EHR) using NLP techniques and OCR.

## 📁 Project Structure

- `data/raw_pdfs/` — Original PDF documents (e.g. 400 patient histories)
- `data/processed_texts/` — Text extracted from PDFs via OCR
- `models/` — Pretrained and fine-tuned NLP models
- `notebooks/` — Jupyter notebooks for experimentation and training
- `src/ocr/` — Code for PDF → text conversion (e.g., using NVIDIA nv-ingest or PyMuPDF)
- `src/nlp/` — Named Entity Recognition and medical term extraction
- `src/classification/` — Specialty classification logic (e.g., ENT vs Cardio)
- `outputs/structured/` — Final structured summaries per specialty
- `web_app/` — Streamlit frontend for users to upload files and view summaries

## 🚀 Quick Start

1. Install dependencies (Tesseract, PyMuPDF, Transformers)
2. Place sample PDFs in `data/raw_pdfs/`
3. Run preprocessing pipeline in `src/ocr/`
4. Fine-tune and run NER + classification in `src/nlp/` and `src/classification/`
5. Launch Streamlit app in `web_app/`

## 🔍 Key Models

- Bio_ClinicalBERT for NER
- Logistic Regression / Transformers for text classification

## 🧪 Future Work

- Specialty-specific tuning
- Integration with hospital record systems
- Multilingual support
