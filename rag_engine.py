import os
import re
import fitz
import gradio as gr
from dotenv import load_dotenv
from scaledown.compressor.scaledown_compressor import ScaleDownCompressor


load_dotenv()

sd_key = os.getenv("SCALEDOWN_API_KEY")

if not sd_key:
    raise ValueError("Missing SCALEDOWN_API_KEY")


compressor = ScaleDownCompressor(
    target_model="gpt-4o",
    api_key=sd_key
)



def read_pdf(file):

    doc = fitz.open(file.name)
    text=""

    for p in doc:
        text+=p.get_text()

    return text



def chunk_text(text , size=1500 , overlap=300):

    chunks=[]
    start=0

    while start < len(text):
        end=start+size
        chunks.append(text[start:end])
        start=end-overlap

    return chunks



def score_chunk(chunk , question):

    q_words = question.lower().split()
    chunk_lower = chunk.lower()

    score=0

    for w in q_words:
        if w in chunk_lower:
            score+=1

    return score



def retrieve(question , chunks , k=4):

    scored=[]

    for ch in chunks:
        s=score_chunk(ch , question)
        scored.append((s , ch))

    scored.sort(reverse=True , key=lambda x:x[0])

    top_chunks=[c for _,c in scored[:k]]

    return top_chunks



def clean_output(text):

    text=text.strip()

    text=re.sub(r'\n{3,}', '\n\n', text)

    text=text.replace("•","-")

    return text



def executive_summary(text):

    prompt="""
You are generating an EXECUTIVE ACADEMIC BRIEF.

Strict Rules:
- Maximum 1 page equivalent.
- No storytelling tone.
- Bullet points where possible.
- No long continuous paragraphs.

Output format exactly as below:

TITLE:
- Paper title

PROBLEM:
- Core problem addressed

PROPOSED SYSTEM:
- Architecture summary in 4-6 bullets

MODELS USED:
- List models and purpose

PERFORMANCE METRICS:
- Extract accuracy, precision, recall, F1 if present
- Format clearly per model

KEY CONTRIBUTIONS:
- 4-6 bullets

LIMITATIONS:
- Short bullet list

CONCLUSION:
- 3-5 concise sentences
"""

    result=compressor.compress(
        context=text,
        prompt=prompt
    )

    return clean_output(result.content)



def answer_question(text , question):

    chunks=chunk_text(text)

    top_chunks=retrieve(question , chunks)

    context="\n\n".join(top_chunks)

    structured_prompt=f"""
You are answering strictly from academic context.

Rules:
- Direct and technical.
- No narrative tone.
- Bullet format preferred.

Question:
{question}

Output:

DIRECT ANSWER:
- Clear response

TECHNICAL DETAILS:
- Supporting explanation

EVIDENCE:
- Extracted supporting facts
"""

    result=compressor.compress(
        context=context,
        prompt=structured_prompt
    )

    return clean_output(result.content)



def run(file , question):

    if file is None:
        return "Upload a PDF file."

    text=read_pdf(file)

    if question.strip()=="":
        return executive_summary(text)

    return answer_question(text , question)



css="""
body {font-family:Century Gothic, sans-serif;background-color:#0e1117;color:#e6edf3;}
.gradio-container {max-width:1100px !important;margin:auto;}
textarea {background:#161b22 !important;color:#e6edf3 !important;border:1px solid #30363d !important;}
button {background:#2563eb !important;border:none;}
button:hover {background:#1d4ed8 !important;transition:0.2s;}
footer {display:none !important;}
"""


with gr.Blocks(css=css) as ui:

    gr.Markdown("## Academic Paper Executive Brief Generator")

    file_input=gr.File(label="Upload Research Paper (PDF)")

    question_input=gr.Textbox(
        label="Optional Question",
        placeholder="Leave empty to generate executive brief",
        lines=2
    )

    run_btn=gr.Button("Generate")

    output=gr.Textbox(
        label="Output",
        lines=30
    )

    run_btn.click(
        fn=run,
        inputs=[file_input , question_input],
        outputs=output
    )


if __name__=="__main__":
    ui.launch()
