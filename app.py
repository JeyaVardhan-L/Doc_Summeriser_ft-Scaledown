import os
import re
import fitz
import gradio as gr
from dotenv import load_dotenv
from scaledown.compressor.scaledown_compressor import ScaleDownCompressor


load_dotenv()
api_key = os.getenv("SCALEDOWN_API_KEY")

if not api_key:
    raise ValueError("API key not found in .env")

compressor = ScaleDownCompressor(
    target_model="gpt-4o",
    api_key=api_key
)


def get_pdf_text(file):
    doc = fitz.open(file.name)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def split_sections(text):
    pattern = r"(Introduction|Chapter\s+\d+|Conclusion|References|ABSTRACT|Abstract)"
    parts = re.split(pattern, text)

    sections = {}
    current = "Start"

    for part in parts:
        if re.match(pattern, part):
            current = part.strip()
            sections[current] = ""
        else:
            sections[current] = sections.get(current, "") + part.strip()

    return sections


def full_summary(text):

    sections = split_sections(text)

    prompt = """
You are summarizing an academic research paper.

Strictly structure output as:

TITLE:
ABSTRACT SUMMARY:
PROBLEM STATEMENT:
PROPOSED SYSTEM:
MODELS USED:
MODEL PERFORMANCE:
KEY CONTRIBUTIONS:
LIMITATIONS:
CONCLUSION:

Use bullet points where appropriate.
Avoid long continuous paragraphs.
Keep sections clearly separated.
"""

    combined = ""

    for title, content in sections.items():

        if len(content) < 600:
            continue

        result = compressor.compress(
            context=content,
            prompt=prompt
        )

        formatted = result.content.replace("\n\n\n", "\n\n").strip()

        combined += "\n\n===== " + title + " =====\n\n"
        combined += formatted
        combined += "\n\n----------------------------------------\n"

    if combined.strip() == "":
        return "Document too small or sections not detected."

    return combined


def answer_question(text, question):

    sections = split_sections(text)

    structured_prompt = f"""
You are analyzing an academic research paper.

Question:
{question}

Strictly follow this output structure:

Direct Answer:
- Clear and precise answer

Technical Explanation:
- Grounded strictly in document context

Key Evidence:
- Bullet points extracted from the text

Avoid long continuous paragraphs.
"""

    final_output = ""

    for title, content in sections.items():

        if len(content) < 600:
            continue

        if any(word.lower() in content.lower() for word in question.split()):

            result = compressor.compress(
                context=content,
                prompt=structured_prompt
            )

            formatted = result.content.replace("\n\n\n", "\n\n").strip()

            final_output += "\n\n===== " + title + " =====\n\n"
            final_output += formatted
            final_output += "\n\n----------------------------------------\n"

    if final_output.strip() == "":
        return "No relevant sections found."

    return final_output


def run(file, question):

    if file is None:
        return "Upload a PDF file."

    text = get_pdf_text(file)

    if question.strip() == "":
        return full_summary(text)

    return answer_question(text, question)


css = """
body {background-color:#111;color:#e0e0e0;font-family:Century Gothic, sans-serif;}
.gradio-container {max-width:1100px;margin:auto;}
textarea {background:#1a1a1a !important;color:#e0e0e0 !important;}
button {background:#2c2c2c !important;color:#fff !important;border:1px solid #444;}
button:hover {background:#3a3a3a !important;}
"""


with gr.Blocks(css=css) as ui:

    gr.Markdown("## Academic Paper Summarizer")

    with gr.Row():
        file_input = gr.File(label="Upload PDF")
        question_input = gr.Textbox(
            label="Ask Question (optional)",
            placeholder="Leave empty to generate full structured summary"
        )

    output_box = gr.Textbox(
        label="Output",
        lines=30
    )

    submit = gr.Button("Run")

    submit.click(
        fn=run,
        inputs=[file_input, question_input],
        outputs=output_box
    )


if __name__ == "__main__":
    ui.launch()
