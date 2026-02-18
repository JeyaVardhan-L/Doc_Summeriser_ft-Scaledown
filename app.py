import os
import re
import fitz
import gradio as gr
from dotenv import load_dotenv
from scaledown.compressor.scaledown_compressor import ScaleDownCompressor

load_dotenv()
api_key=os.getenv("SCALEDOWN_API_KEY")

if not api_key:
    raise ValueError("API key not found")

compressor = ScaleDownCompressor(
    target_model="gpt-4o",
    api_key=api_key
)


def read_pdf(file):
    doc = fitz.open(file.name)
    text=""

    for p in doc:
        text+=p.get_text()

    return text


def split_sec(text):
    pat=r"(Introduction|Chapter\s+\d+|Conclusion|References)"
    parts=re.split(pat , text)

    data={}
    cur="Start"

    for part in parts:
        if re.match(pat , part):
            cur=part.strip()
            data[cur]=""
        else:
            data[cur]=data.get(cur,"")+part.strip()

    return data


def filter_sec(data , q):
    out={}
    words=q.split()

    for k , v in data.items():
        if any(w.lower() in v.lower() for w in words):
            out[k]=v

    return out


def run(file , q):

    if file is None or q.strip()=="":
        return "Upload a PDF and enter a question."

    full=read_pdf(file)
    sections=split_sec(full)
    rel=filter_sec(sections , q)

    if not rel:
        return "No relevant sections found."

    prompt=f"""
You are answering a question based on an academic paper.

Question:
{q}

Provide output in this format:

Direct Answer:
- Clear answer.

Technical Explanation:
- Explanation grounded in document.

Key Evidence:
- Bullet points from text.
"""

    out=""

    for title , content in rel.items():

        if len(content)<500:
            continue

        res=compressor.compress(
            context=content,
            prompt=prompt
        )

        out+="\n\n"+title+"\n\n"
        out+=res.content
        out+="\n\n-------------------------------------\n"

    return out


custom_css="""
body {
    font-family: "Century Gothic", "Gothic A1", sans-serif;
    background-color: #0f1117;
    color: #e6e6e6;
}

.gradio-container {
    max-width: 1200px !important;
}

button {
    background-color: #1f2937 !important;
    color: #ffffff !important;
    border: 1px solid #374151 !important;
    transition: all 0.2s ease-in-out !important;
}

button:hover {
    background-color: #2563eb !important;
    border-color: #2563eb !important;
}

textarea {
    background-color: #111827 !important;
    color: #f3f4f6 !important;
    border: 1px solid #374151 !important;
    font-family: "Century Gothic", "Gothic A1", sans-serif;
    overflow-y: auto !important;
}

input[type="text"] {
    background-color: #111827 !important;
    color: #f3f4f6 !important;
    border: 1px solid #374151 !important;
}

h1, h2, h3 {
    font-weight: 500;
}

hr {
    border: 0;
    height: 1px;
    background: #2d3748;
}
"""


with gr.Blocks(css=custom_css) as ui:

    gr.Markdown("# Academic Paper Q&A")
    gr.Markdown("Upload a research paper and ask structured questions.")

    with gr.Row():

        with gr.Column(scale=1):
            file_input=gr.File(label="PDF File", file_types=[".pdf"])
            question_input=gr.Textbox(
                label="Question",
                placeholder="Enter question related to the document",
                lines=3
            )
            run_btn=gr.Button("Analyze")

        with gr.Column(scale=2):
            output_box=gr.Textbox(
                label="Answer",
                lines=28,
                
            )

    run_btn.click(
        fn=run,
        inputs=[file_input , question_input],
        outputs=output_box
    )


if __name__=="__main__":
    ui.launch()
