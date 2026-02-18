import os
import fitz
from dotenv import load_dotenv
from scaledown.compressor.scaledown_compressor import ScaleDownCompressor

def extract_text_from_pdf(path):
    doc = fitz.open(path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

if __name__ == "__main__":

    load_dotenv() # load variables from .env file into environment
    api_key = os.getenv("SCALEDOWN_API_KEY")

    if not api_key:
        raise ValueError("API key not found in .env file")

    text = extract_text_from_pdf("sample.pdf")
    # here this create compressor object which talks to scaledown servers
    compressor = ScaleDownCompressor(
        target_model="gpt-4o",
        api_key=api_key
    )
    prompt = "Provide a concise technical summary of this document."
    # this part sends text + prompt to scaledown which then calls GPT internally..
    result = compressor.compress(
        context=text,
        prompt=prompt
    )

    print("\nCOMPRESSED OUTPUT:\n")
    print(result.content)          # shortened version of the document
    print("\nMETRICS:")
    print("Original Tokens:", result.tokens[0]) # before compression
    print("Compressed Tokens:", result.tokens[1]) # after compression
    print("Savings %:", result.savings_percent) # token reduction %
    print("Latency (ms):", result.latency) # API response time
