import os
import fitz  # pymupdf
from dotenv import load_dotenv
from scaledown.compressor.scaledown_compressor import ScaleDownCompressor


# ----------------------------
# 1️⃣ Extract Text from PDF
# ----------------------------
def extract_text_from_pdf(path):
    doc = fitz.open(path)
    full_text = ""

    for page in doc:
        full_text += page.get_text()

    return full_text


# ----------------------------
# 2️⃣ Main Execution
# ----------------------------
if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("SCALEDOWN_API_KEY")

    if not api_key:
        raise ValueError("API key not found in .env file")

    print("📄 Extracting PDF...")
    text = extract_text_from_pdf("sample.pdf")

    print("✅ Extraction complete")
    print("🧠 Initializing ScaleDown...")

    compressor = ScaleDownCompressor(
        target_model="gpt-4o",
        api_key=api_key
    )

    print("🚀 Compressing entire document...")

    prompt = "Provide a concise technical summary of this document."

    result = compressor.compress(
        context=text,
        prompt=prompt
    )

    print("\n===== COMPRESSED OUTPUT =====\n")
    print(result.content)

    print("\n===== METRICS =====")
    print("Original Tokens:", result.tokens[0])
    print("Compressed Tokens:", result.tokens[1])
    print("Savings %:", result.savings_percent)
    print("Latency (ms):", result.latency)
