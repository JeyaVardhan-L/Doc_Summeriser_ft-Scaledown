import os
import re
import fitz
from dotenv import load_dotenv
from scaledown.compressor.scaledown_compressor import ScaleDownCompressor
def extract_text_from_pdf(path):
    doc = fitz.open(path)
    text = ""

    for page in doc:
        text += page.get_text()
    return text

def split_into_sections(text):
    pattern = r"(Introduction|Chapter\s+\d+|Conclusion|References)" # pattern to detect common section titles in academic documents
    parts = re.split(pattern, text) # this splits the text whenever it sees the above section names
    
    sections = {}
    current_title = "Start"

    for part in parts:
        # if the part matches section title pattern, treat it as new section header
        if re.match(pattern, part):
            current_title = part.strip()
            sections[current_title] = ""
        else:
            # otherwise append text under the current section title
            sections[current_title] = sections.get(current_title, "") + part.strip()
    return sections

if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("SCALEDOWN_API_KEY")

    if not api_key:
        raise ValueError("API key not found")

    full_text = extract_text_from_pdf("sample.pdf")
    sections = split_into_sections(full_text)  # split full document into logical sections before compression

    compressor = ScaleDownCompressor(
        target_model="gpt-4o",
        api_key=api_key
    )

    for title, content in sections.items():
        # skip very small sections to avoid unnecessary API calls
        if len(content) < 500:
            continue

        print("\nSECTION:", title)
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

        # compress only this section instead of entire document
        result = compressor.compress(
            context=content,
            prompt=structured_prompt
        )
        print(result.content)
        print("\nTokens:", result.tokens)
