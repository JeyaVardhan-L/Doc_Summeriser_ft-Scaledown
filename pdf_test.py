import fitz  # pymupdf

def extract_text_from_pdf(path):
    doc = fitz.open(path)
    full_text = ""

    for page in doc:
        full_text += page.get_text()

    return full_text

if __name__ == "__main__":
    text = extract_text_from_pdf("sample.pdf")
    print(text[:2000])  # print first 2000 chars
