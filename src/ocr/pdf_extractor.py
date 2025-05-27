import fitz  # PyMuPDF
import os
import argparse

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_all_pdfs(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(input_dir, filename)
            text = extract_text_from_pdf(pdf_path)

            txt_filename = filename.replace(".pdf", ".txt")
            with open(os.path.join(output_dir, txt_filename), "w", encoding="utf-8") as f:
                f.write(text)
            print(f"✓ Extracted: {filename}")

def extract_single_pdf(file_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    text = extract_text_from_pdf(file_path)
    filename = os.path.basename(file_path).replace(".pdf", ".txt")
    with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
        f.write(text)
    print(f"✓ Extracted: {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="Путь к конкретному PDF-файлу")
    parser.add_argument("--input_dir", type=str, default="data/raw_pdfs", help="Папка с PDF-файлами")
    parser.add_argument("--output_dir", type=str, default="data/processed_texts", help="Папка для сохранения TXT")

    args = parser.parse_args()

    if args.file:
        extract_single_pdf(args.file, args.output_dir)
    else:
        extract_all_pdfs(args.input_dir, args.output_dir)
