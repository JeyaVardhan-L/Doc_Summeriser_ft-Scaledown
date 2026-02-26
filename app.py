import os
import re
import fitz
import gradio as gr
from dotenv import load_dotenv

from scaledown.compressor.scaledown_compressor import ScaleDownCompressor

load_dotenv()


sd_key = os.getenv("SCALEDOWN_API_KEY")

if not sd_key:
    raise ValueError("Missing SCALEDOWN_API_KEY in .env")

# estimating standard GPT-4o input pricing per 1k tokens for the dashboard
PRICE_PER_1K_TOKENS = 0.005  

compressor = ScaleDownCompressor(
    target_model="gpt-4o",
    api_key=sd_key
)


def read_pdf(file):
    doc = fitz.open(file.name)
    text = ""
    
    for p in doc:
        text += p.get_text()
        
    return text



def chunk_text(text, size=2000, overlap=300):
    chunks = []
    start = 0

    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap

    return chunks




# ---------------- MAIN PIPELINE ----------------
# doing this to pass the token counts back to the UI
def executive_pipeline(text):

    chunks = chunk_text(text)
    
    micro_summaries = []
    
    total_original_tokens = 0
    total_compressed_tokens = 0


    # Phase 1
    for ch in chunks:
        prompt = """
Summarize this academic content concisely.
Focus only on technical content.
No storytelling.
Max 8 lines.
"""
        res = compressor.compress(context=ch, prompt=prompt)
        micro_summaries.append(res.content.strip())
        
        # tallying tokens
        total_original_tokens += res.tokens[0]
        total_compressed_tokens += res.tokens[1]

    combined = "\n\n".join(micro_summaries)



    # Phase 2
    prompt_struct = """
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
    res_struct = compressor.compress(context=combined, prompt=prompt_struct)
    
    total_original_tokens += res_struct.tokens[0]
    total_compressed_tokens += res_struct.tokens[1]



    # Phase 3
    prompt_final = """
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
    res_final = compressor.compress(context=res_struct.content.strip(), prompt=prompt_final)
    
    total_original_tokens += res_final.tokens[0]
    total_compressed_tokens += res_final.tokens[1]


    return res_final.content.strip(), total_original_tokens, total_compressed_tokens




# ---------------- QUESTION MODE ----------------

def answer_question(text, question):
    
    if not text:
        return "You gotta upload and process a document first."

    chunks = chunk_text(text)
    
    scored = []

    for ch in chunks:
        score = sum(1 for w in question.lower().split() if w in ch.lower())
        scored.append((score, ch))

    scored.sort(reverse=True, key=lambda x: x[0])
    
    top = [c for _, c in scored[:4]]
    
    context = "\n\n".join(top)

    prompt = f"""
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

    res = compressor.compress(context=context, prompt=prompt)
    
    return res.content.strip()




# ---------------- GRADIO UI & WRAPPERS ----------------

def process_btn_click(file):
    
    if file is None:
        gr.Warning("Upload a PDF file.")
        return None, "Upload a PDF.", "0 tokens", "$0.00"
        
    gr.Info("Running phase-based extraction... this might take a sec depending on the PDF size.")
    
    raw_text = read_pdf(file)
    
    final_brief, orig_tok, comp_tok = executive_pipeline(raw_text)
    
    # Economics math
    tokens_saved = orig_tok - comp_tok
    money_saved = (tokens_saved / 1000) * PRICE_PER_1K_TOKENS
    
    
    # returning raw_text to the gr.State() so Q&A can use it
    return raw_text, final_brief, f"{tokens_saved:,} tokens", f"${money_saved:.4f}"



def ask_btn_click(question, doc_state):
    
    if not question.strip():
        return "Ask something!"
        
    return answer_question(doc_state, question)




# --- THE DASHBOARD ---

css = """
body {font-family:Century Gothic, sans-serif;background:#0e1117;color:#e6edf3;}
.gradio-container {max-width:1200px !important;margin:auto;}
textarea {background:#161b22 !important;color:#e6edf3 !important;border:1px solid #30363d !important;}
button.primary {background:#2563eb !important;border:none;}
button.primary:hover {background:#1d4ed8 !important;}
footer {display:none !important;}
"""

with gr.Blocks(css=css, theme=gr.themes.Base()) as ui:
    
    gr.Markdown("# 📄 Academic Executive Brief Generator (ScaleDown Powered)")
    gr.Markdown("Upload an academic paper. We'll run a multi-phase extraction pipeline and compress it to save token costs.")
    
    # holds the extracted text for the session
    doc_state = gr.State(None)
    
    with gr.Row():
        
        # Left Panel 
        with gr.Column(scale=1):
            
            file_input = gr.File(label="Upload Research Paper (PDF)")
            run_btn = gr.Button("Process & Summarize", variant="primary")
            
            gr.HTML("<hr>")
            gr.Markdown("### 💰 Token Economics Dashboard")
            
            with gr.Row():
                tokens_saved_box = gr.Textbox(label="Tokens Saved", interactive=False, value="0 tokens")
                money_saved_box = gr.Textbox(label="Est. Money Saved", interactive=False, value="$0.00")
                
                
        # Right Panel
        with gr.Column(scale=2):
            
            with gr.Tabs():
                
                with gr.TabItem("Executive Brief"):
                    summary_output = gr.Textbox(label="Structured Output", lines=22, interactive=False)
                    
                with gr.TabItem("Interactive Q&A"):
                    gr.Markdown("Ask questions against the loaded document context.")
                    
                    user_q = gr.Textbox(label="Question", placeholder="What models were compared in this paper?", lines=2)
                    ask_btn = gr.Button("Find Answer")
                    
                    answer_box = gr.Textbox(label="Extracted Answer", lines=12, interactive=False)


    # wires
    run_btn.click(
        fn=process_btn_click,
        inputs=[file_input],
        outputs=[doc_state, summary_output, tokens_saved_box, money_saved_box]
    )
    
    ask_btn.click(
        fn=ask_btn_click,
        inputs=[user_q, doc_state],
        outputs=[answer_box]
    )


if __name__ == "__main__":
    ui.launch()