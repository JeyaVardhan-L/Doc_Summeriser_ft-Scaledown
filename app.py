import os
import re
import fitz
import gradio as gr
from dotenv import load_dotenv
from scaledown.compressor.scaledown_compressor import ScaleDownCompressor


# ----------------------------
# Setup
# ----------------------------
load_dotenv()
api_key = os.getenv("SCALEDOWN_API_KEY")

compressor = ScaleDownCompressor(
    target_model="gpt-4o",
    api_key=api_key
)


# ----------------------------
# Extract PDF text
# ----------------------------
def extract_text_from_pdf(file):
    doc = fitz.open(file.name)
    text = ""

    for page in doc:
        text += page.get_text()

    return text


# ----------------------------
# Split into sections
# ----------------------------
def split_into_sections(text):
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
# Simple relevance filter
# ----------------------------
def filter_relevant_sections(sections, question):
    relevant = {}

    for title, content in sections.items():
        if any(word.lower() in content.lower() for word in question.split()):
            relevant[title] = content

    return relevant


# ----------------------------
# Main logic
# ----------------------------
def process_pdf(file, question):

    full_text = extract_text_from_pdf(file)
    sections = split_into_sections(full_text)

    relevant_sections = filter_relevant_sections(sections, question)

    if not relevant_sections:
        return "No relevant sections found."

    structured_prompt = f"""
You are answering a question based on an academic paper.

Question:
{question}

Provide output in this format:

Direct Answer:
- Clear, precise answer.

Technical Explanation:
- Explanation grounded strictly in the document context.

Key Evidence:
- Bullet points of supporting statements from the text.
"""

    final_output = ""

    for title, content in relevant_sections.items():

        if len(content) < 500:
            continue

        result = compressor.compress(
            context=content,
            prompt=structured_prompt
        )

        final_output += f"\n\n===== {title} =====\n\n"
        final_output += result.content
        final_output += "\n\n--------------------------------------\n"

    return final_output


# ----------------------------
# Gradio UI
# ----------------------------
interface = gr.Interface(
    fn=process_pdf,
    inputs=[
        gr.File(label="Upload Research Paper (PDF)"),
        gr.Textbox(label="Ask a Question About the Paper")
    ],
    outputs=gr.Textbox(label="Answer", lines=30),
    title="Academic Paper Q&A (ScaleDown RAG MVP)",
    description="Upload a PDF and ask questions about it."
)

if __name__ == "__main__":
    interface.launch()
