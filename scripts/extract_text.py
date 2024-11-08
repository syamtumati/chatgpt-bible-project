import os
import fitz  # PyMuPDF library

def extract_text_from_pdf(file_path):
    text = ""
    with fitz.open(file_path) as pdf:
        for page in pdf:
            text += page.get_text()
    return text

def process_all_pdfs(input_folder, output_file):
    combined_text = ""
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".pdf"):
                file_path = os.path.join(root, file)
                print(f"Processing {file_path}")
                combined_text += extract_text_from_pdf(file_path) + "\n\n"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(combined_text)
    print(f"Combined text saved to {output_file}")

# Define paths for Bible and sermon PDF files
bible_input_folder = "data/bible/"
output_bible_text = "data/sermons/bible_sermon_combined.txt"

# Process PDFs and save combined text files
process_all_pdfs(bible_input_folder, output_bible_text)
