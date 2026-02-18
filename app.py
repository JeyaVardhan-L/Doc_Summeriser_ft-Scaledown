import os
import re
import fitz
import gradio as gr
from dotenv import load_dotenv
from scaledown.compressor.scaledown_compressor import ScaleDownCompressor


load_dotenv()

sd_key=os.getenv("SCALEDOWN_API_KEY")

if not sd_key:
    raise ValueError("Missing SCALEDOWN_API_KEY")


compressor=ScaleDownCompressor(
    target_model="gpt-4o",
    api_key=sd_key
)



def read_pdf(file):

    doc=fitz.open(file.name)
    text=""

    for p in doc:
        text+=p.get_text()

    return text



def chunk_text(text , size=2000 , overlap=300):

    chunks=[]
    start=0

    while start < len(text):
        end=start+size
        chunks.append(text[start:end])
        start=end-overlap

    return chunks



# ---------------- PHASE 1 ----------------
# Micro Summaries Per Chunk

def micro_summarize(chunk):

    prompt="""
Summarize this academic content concisely.
Focus only on technical content.
No storytelling.
Max 8 lines.
"""

    result=compressor.compress(
        context=chunk,
        prompt=prompt
    )

    return result.content.strip()



# ---------------- PHASE 2 ----------------
# Structured Extraction From Combined Micro Summary

def extract_structure(text):

    prompt="""
You are extracting structured academic information.

Return strictly in this format:

TITLE:
- inferred title

PROBLEM:
- core research problem

MODELS:
- list models used

PERFORMANCE:
- accuracy, precision, recall, F1 if available

CONTRIBUTIONS:
- key contributions

LIMITATIONS:
- stated or implied weaknesses
"""

    result=compressor.compress(
        context=text,
        prompt=prompt
    )

    return result.content.strip()



# ---------------- PHASE 3 ----------------
# Final Executive Brief Synthesis

def synthesize_final(structured_text):

    prompt="""
Generate a clean executive academic brief.

Rules:
- Use bullet formatting.
- No paragraph longer than 3 lines.
- Keep it concise.
- Professional tone.

Sections required:

TITLE
PROBLEM
PROPOSED APPROACH
MODELS USED
PERFORMANCE METRICS
KEY CONTRIBUTIONS
LIMITATIONS
CONCLUSION
"""

    result=compressor.compress(
        context=structured_text,
        prompt=prompt
    )

    return result.content.strip()



def executive_pipeline(text):

    chunks=chunk_text(text)

    micro_summaries=[]

    for ch in chunks:
        micro=micro_summarize(ch)
        micro_summaries.append(micro)

    combined="\n\n".join(micro_summaries)

    structured=extract_structure(combined)

    final_output=synthesize_final(structured)

    return final_output



# ---------------- QUESTION MODE ----------------

def answer_question(text , question):

    chunks=chunk_text(text)

    scored=[]

    for ch in chunks:
        score=sum(1 for w in question.lower().split() if w in ch.lower())
        scored.append((score , ch))

    scored.sort(reverse=True , key=lambda x:x[0])

    top=[c for _,c in scored[:4]]

    context="\n\n".join(top)

    prompt=f"""
Answer strictly from academic context.

Question:
{question}

Format:

DIRECT ANSWER:
- precise response

TECHNICAL DETAILS:
- explanation

EVIDENCE:
- supporting facts
"""

    result=compressor.compress(
        context=context,
        prompt=prompt
    )

    return result.content.strip()



def run(file , question):

    if file is None:
        return "Upload PDF file."

    text=read_pdf(file)

    if question.strip()=="":
        return executive_pipeline(text)

    return answer_question(text , question)



css="""
body {font-family:Century Gothic, sans-serif;background:#0e1117;color:#e6edf3;}
.gradio-container {max-width:1100px !important;margin:auto;}
textarea {background:#161b22 !important;color:#e6edf3 !important;border:1px solid #30363d !important;}
button {background:#2563eb !important;border:none;}
button:hover {background:#1d4ed8 !important;}
footer {display:none !important;}
"""


with gr.Blocks(css=css) as ui:

    gr.Markdown("## Academic Executive Brief Generator (Phase-Based Pipeline)")

    file_input=gr.File(label="Upload Research Paper (PDF)")

    question_input=gr.Textbox(
        label="Optional Question",
        placeholder="Leave empty for executive brief",
        lines=2
    )

    run_btn=gr.Button("Generate")

    output=gr.Textbox(
        label="Output",
        lines=32
    )

    run_btn.click(
        fn=run,
        inputs=[file_input , question_input],
        outputs=output
    )


if __name__=="__main__":
    ui.launch()
