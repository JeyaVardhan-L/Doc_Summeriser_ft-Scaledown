import os
import re
import fitz
from dotenv import load_dotenv
from scaledown.compressor.scaledown_compressor import ScaleDownCompressor


# ----------------------------
# Extract PDF Text
# ----------------------------
def extract_text_from_pdf(path):
    doc = fitz.open(path)
    text = ""

    for page in doc:
        text += page.get_text()

    return text


# ----------------------------
# Simple Section Splitter
# ----------------------------
def split_into_sections(text):
    # Split on common academic section markers
    pattern = r"(Introduction|Chapter\s+\d+|Conclusion|References)"
    
    parts = re.split(pattern, text)

    sections = {}
    current_title = "Start"

    for part in parts:
        if re.match(pattern, part):
            current_title = part.strip()
            sections[current_title] = ""
        else:
            sections[current_title] = sections.get(current_title, "") + part.strip()

    return sections


# ----------------------------
# Main Execution
# ----------------------------
if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("SCALEDOWN_API_KEY")

    if not api_key:
        raise ValueError("API key not found")

    print("📄 Extracting PDF...")
    full_text = extract_text_from_pdf("sample.pdf")

    print("🔎 Splitting into sections...")
    sections = split_into_sections(full_text)

    compressor = ScaleDownCompressor(
        target_model="gpt-4o",
        api_key=api_key
    )

    print("🚀 Compressing sections...\n")

    for title, content in sections.items():
        if len(content) < 500:  # skip tiny fragments
            continue

        print(f"\n===== SECTION: {title} =====")

        structured_prompt = """
You are summarizing an academic section.

Provide output in this exact format:

ELI5 Summary:
- Explain in very simple language.

Technical Summary:
- Provide a structured technical explanation.

Expert Summary:
- Deep, precise explanation preserving technical nuance.

Key Points:
- Bullet list of most important concepts.
"""

        result = compressor.compress(
            context=content,
            prompt=structured_prompt
        )

        print(result.content)
        print("\nTokens:", result.tokens)

