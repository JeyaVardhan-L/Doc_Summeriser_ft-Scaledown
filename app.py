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
    text=""
    for page in doc:
        text+=page.get_text()
    return text


def split_sections(text):
    pattern=r"(Introduction|Chapter\s+\d+|Conclusion|References)"
    parts=re.split(pattern,text)

    sections={}
    current="Start"

    for part in parts:
        if re.match(pattern,part):
            current=part.strip()
            sections[current]=""
        else:
            sections[current]=sections.get(current,"")+part.strip()

    return sections


def full_summary(text):

    prompt="""
You are summarizing a research paper.

Provide output in this format:

Document Overview:
- What is this paper about?

Key Contributions:
- What new idea or contribution is presented?

Methodology:
- How was it done?

Main Results:
- What are the key findings?

Limitations:
- Any limitations mentioned or implied.
"""

    result=compressor.compress(
        context=text,
        prompt=prompt
    )

    return result.content


def answer_question(sections,question):

    structured_prompt=f"""
You are answering a question using ONLY the provided academic content.

Question:
{question}

Provide output in this format:

Direct Answer:
- Clear answer

Technical Explanation:
- Grounded strictly in document context

Key Evidence:
- Bullet points from the paper
"""

    final_output=""

    for title,content in sections.items():

        if len(content)<600:
            continue

        if any(word.lower() in content.lower() for word in question.split()):

            result=compressor.compress(
                context=content,
                prompt=structured_prompt
            )

            final_output+=f"\n\n===== {title} =====\n\n"
            final_output+=result.content
            final_output+="\n\n------------------------------\n"

    if final_output.strip()=="":
        return "No relevant sections found."

    return final_output


def run(file,question):

    if file is None:
        return "Upload a PDF file."

    text=get_pdf_text(file)

    if question.strip()=="":
        return full_summary(text)

    sections=split_sections(text)

    return answer_question(sections,question)


css="""
body {background-color:#111;color:#e0e0e0;font-family:Century Gothic, sans-serif;}
.gradio-container {max-width:1100px;margin:auto;}
textarea {background:#1a1a1a !important;color:#e0e0e0 !important;}
button {background:#2c2c2c !important;color:#fff !important;border:1px solid #444;}
button:hover {background:#3a3a3a !important;}
"""


with gr.Blocks(css=css) as ui:

    gr.Markdown("## Academic Paper Summarizer")

    with gr.Row():
        file_input=gr.File(label="Upload PDF")
        question_input=gr.Textbox(
            label="Ask Question (optional)",
            placeholder="Leave empty to generate full structured summary"
        )

    output_box=gr.Textbox(
        label="Output",
        lines=30
    )

    submit=gr.Button("Run")

    submit.click(
        fn=run,
        inputs=[file_input,question_input],
        outputs=output_box
    )


if __name__=="__main__":
    ui.launch()
