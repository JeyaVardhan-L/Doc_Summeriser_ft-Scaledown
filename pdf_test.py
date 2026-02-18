import fitz  # this is part of pymupdf

def extract_text_from_pdf(path):
    doc = fitz.open(path) #Here it loads into memory and acts as collection of pages...a doc
    full_text = ""
    for page in doc:
        full_text += page.get_text() #this is the part where pdf converts to text
    return full_text

if __name__ == "__main__":
    text = extract_text_from_pdf("sample.pdf")
    print(text[:2000])  # print first 2000 chars...Cause o not flood the terminal
