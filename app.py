"""
Placement Records RAG Chatbot
Hugging Face Space — IBM Granite + FAISS + Sentence Transformers
Architecture: Excel → Pandas → Text Chunks → Embeddings → FAISS → Granite LLM
"""

import os
import numpy as np
import pandas as pd
import faiss
import torch
import gradio as gr
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
EXCEL_PATH   = "placements.xlsx"
EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
GRANITE_MODEL = "ibm-granite/granite-3.3-2b-instruct"
TOP_K        = 3
MAX_NEW_TOKENS = 256


# ─────────────────────────────────────────────
# 1. LOAD & PREPROCESS EXCEL
# ─────────────────────────────────────────────
def load_excel(path: str) -> tuple[pd.DataFrame, list[str]]:
    """Read Excel and convert each row to a plain-text document."""
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]

    documents = []
    for _, row in df.iterrows():
        doc = (
            f"Name: {row.get('Name', 'N/A')}\n"
            f"Branch: {row.get('Branch', 'N/A')}\n"
            f"Graduation Year: {row.get('Graduation_Year', 'N/A')}\n"
            f"CGPA: {row.get('CGPA', 'N/A')}\n"
            f"Skills: {row.get('Skills', 'N/A')}\n"
            f"Projects: {row.get('Projects', 'N/A')}\n"
            f"Placed At: {row.get('Placed_Company', 'N/A')}\n"
            f"Package (LPA): {row.get('Package_LPA', 'N/A')}"
        )
        documents.append(doc)

    print(f"[DATA] Loaded {len(df)} placement records.")
    return df, documents


# ─────────────────────────────────────────────
# 2. BUILD FAISS INDEX
# ─────────────────────────────────────────────
def build_faiss_index(documents: list[str], embed_model: SentenceTransformer):
    """Encode documents and build a flat L2 FAISS index."""
    print("[FAISS] Encoding documents...")
    embeddings = embed_model.encode(documents, convert_to_numpy=True, show_progress_bar=True)
    embeddings = embeddings.astype(np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    print(f"[FAISS] Index built — {index.ntotal} vectors, dim={dim}")
    return index, embeddings


# ─────────────────────────────────────────────
# 3. RETRIEVAL
# ─────────────────────────────────────────────
def retrieve(
    query: str,
    index: faiss.IndexFlatL2,
    documents: list[str],
    embed_model: SentenceTransformer,
    df: pd.DataFrame,
    branch_filter: str = "All",
    year_filter: str = "All",
    k: int = TOP_K,
) -> tuple[str, list[str]]:
    """
    Optional pre-filter by Branch / Graduation Year, then semantic retrieval.
    Returns (context_string, list_of_matched_docs).
    """
    # Pre-filter dataframe
    filtered_df = df.copy()
    if branch_filter != "All":
        filtered_df = filtered_df[filtered_df["Branch"] == branch_filter]
    if year_filter != "All":
        filtered_df = filtered_df[filtered_df["Graduation_Year"] == int(year_filter)]

    if filtered_df.empty:
        return "No records match the selected filters.", []

    # Rebuild a mini-index on filtered rows only
    filtered_indices = filtered_df.index.tolist()
    filtered_docs    = [documents[i] for i in filtered_indices]

    filtered_embeddings = embed_model.encode(filtered_docs, convert_to_numpy=True).astype(np.float32)
    mini_dim   = filtered_embeddings.shape[1]
    mini_index = faiss.IndexFlatL2(mini_dim)
    mini_index.add(filtered_embeddings)

    query_vec = embed_model.encode([query], convert_to_numpy=True).astype(np.float32)
    actual_k  = min(k, len(filtered_docs))
    _, local_indices = mini_index.search(query_vec, actual_k)

    matched = [filtered_docs[i] for i in local_indices[0] if i < len(filtered_docs)]
    context = "\n\n---\n\n".join(matched)
    return context, matched


# ─────────────────────────────────────────────
# 4. GRANITE GENERATION
# ─────────────────────────────────────────────
def generate_answer(
    question: str,
    context: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
) -> str:
    """Build RAG prompt and generate answer with IBM Granite."""
    prompt = (
        "You are a helpful placement assistant. "
        "Answer the question using ONLY the placement records provided below. "
        "If the answer is not in the context, say 'I don't have that information.'\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only newly generated tokens
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return answer


# ─────────────────────────────────────────────
# 5. INITIALIZE EVERYTHING (runs once at startup)
# ─────────────────────────────────────────────
print("[INIT] Loading embedding model...")
embed_model = SentenceTransformer(EMBED_MODEL)

print("[INIT] Loading Excel data...")
df, documents = load_excel(EXCEL_PATH)
faiss_index, _ = build_faiss_index(documents, embed_model)

print("[INIT] Loading IBM Granite model...")
tokenizer = AutoTokenizer.from_pretrained(GRANITE_MODEL)

# Use float16 on GPU, float32 on CPU
dtype  = torch.float16 if torch.cuda.is_available() else torch.float32
device = "cuda" if torch.cuda.is_available() else "cpu"
model  = AutoModelForCausalLM.from_pretrained(
    GRANITE_MODEL,
    torch_dtype=dtype,
    device_map="auto",
)
model.eval()
print(f"[INIT] Granite loaded on {device}. Ready!")

# Derived filter options
branch_options = ["All"] + sorted(df["Branch"].dropna().unique().tolist())
year_options   = ["All"] + sorted([str(y) for y in df["Graduation_Year"].dropna().unique()])


# ─────────────────────────────────────────────
# 6. GRADIO CHAT FUNCTION
# ─────────────────────────────────────────────
def chat(
    message: str,
    history: list,
    branch_filter: str,
    year_filter: str,
):
    if not message.strip():
        return history, history, ""

    context, matched_docs = retrieve(
        query=message,
        index=faiss_index,
        documents=documents,
        embed_model=embed_model,
        df=df,
        branch_filter=branch_filter,
        year_filter=year_filter,
    )

    if not matched_docs:
        answer = "No matching placement records found for your filters."
    else:
        answer = generate_answer(message, context, tokenizer, model)

    # Build retrieved context display
    context_display = "\n\n".join(
        [f"**Record {i+1}:**\n```\n{doc}\n```" for i, doc in enumerate(matched_docs)]
    ) if matched_docs else "No records retrieved."

    history = history or []
    history.append((message, answer))
    return history, history, context_display


def clear_chat():
    return [], [], ""


# ─────────────────────────────────────────────
# 7. GRADIO UI
# ─────────────────────────────────────────────
CSS = """
#chatbot { height: 420px; overflow-y: auto; }
#context-box { font-size: 0.82rem; }
.filter-row { background: #f0f4ff; border-radius: 8px; padding: 10px; }
"""

with gr.Blocks(css=CSS, title="Placement RAG Chatbot — IBM Granite") as demo:

    gr.Markdown(
        """
        # 🎓 Placement Records RAG Chatbot
        ### Powered by IBM Granite 3.3 · FAISS · Sentence Transformers
        Ask questions about student placement records. Optionally filter by Branch or Graduation Year first.
        """
    )

    with gr.Row(elem_classes="filter-row"):
        branch_dd = gr.Dropdown(
            choices=branch_options,
            value="All",
            label="🏫 Filter by Branch",
            scale=1,
        )
        year_dd = gr.Dropdown(
            choices=year_options,
            value="All",
            label="📅 Filter by Graduation Year",
            scale=1,
        )

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(elem_id="chatbot", label="Chat")
            msg_box = gr.Textbox(
                placeholder="e.g. Who has NLP skills? Which AIML student got the highest package?",
                label="Your Question",
                lines=2,
            )
            with gr.Row():
                send_btn  = gr.Button("Send 🚀", variant="primary")
                clear_btn = gr.Button("Clear 🗑️")

        with gr.Column(scale=2):
            gr.Markdown("### 📋 Retrieved Records")
            context_box = gr.Markdown(
                value="_Retrieved records will appear here..._",
                elem_id="context-box",
            )

    state = gr.State([])

    # Example questions
    gr.Examples(
        examples=[
            ["Which AIML student has NLP skills?"],
            ["Who got placed at IBM?"],
            ["List students with Python skills"],
            ["Who has the highest package?"],
            ["Which 2025 graduates know Deep Learning?"],
            ["Show me CSE students with React experience"],
        ],
        inputs=msg_box,
        label="💡 Try these questions",
    )

    # Wire up events
    send_btn.click(
        fn=chat,
        inputs=[msg_box, state, branch_dd, year_dd],
        outputs=[chatbot, state, context_box],
    ).then(lambda: "", outputs=msg_box)

    msg_box.submit(
        fn=chat,
        inputs=[msg_box, state, branch_dd, year_dd],
        outputs=[chatbot, state, context_box],
    ).then(lambda: "", outputs=msg_box)

    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot, state, context_box],
    )

    gr.Markdown(
        """
        ---
        **Architecture:** Excel → Pandas → Row-to-Text → `all-MiniLM-L6-v2` Embeddings → FAISS → IBM Granite 3.3-2B-Instruct  
        *LLM never performs arithmetic — all filtering is done via Pandas before retrieval.*
        """
    )

demo.launch(server_name="0.0.0.0", server_port=7860)