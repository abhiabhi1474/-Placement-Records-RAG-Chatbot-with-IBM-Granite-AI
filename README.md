---
title: Placement RAG Chatbot — IBM Granite
emoji: 🎓
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.42.0
app_file: app.py
pinned: false
license: apache-2.0
---

# 🎓 Placement Records RAG Chatbot

A minimal RAG (Retrieval-Augmented Generation) chatbot over student placement data, powered by **IBM Granite 3.3-2B-Instruct** and **FAISS**.

## Architecture

```
Excel File (placements.xlsx)
        ↓
  Pandas (row → text)
        ↓
  all-MiniLM-L6-v2 Embeddings
        ↓
  FAISS Index (flat L2)
        ↓
  Optional Pre-filter (Branch / Year)
        ↓
  Semantic Retrieval (Top-K)
        ↓
  IBM Granite 3.3-2B-Instruct
        ↓
  Answer
```

## Key Design Decisions

- **LLM never does arithmetic** — all filtering (branch, year, CGPA comparisons) is done by Pandas before passing to the model.
- **Pre-filter → then RAG** — branch/year dropdowns shrink the candidate pool first; semantic search runs on that filtered subset.
- **Lightweight embedding model** — `all-MiniLM-L6-v2` keeps inference fast even on CPU.
- **IBM Granite 3.3-2B-Instruct** — instruction-tuned, runs on GPU (float16) or CPU (float32) automatically.

## Excel Schema

| Column | Description |
|---|---|
| Name | Student name |
| Branch | AIML / CSE / ECE / MECH etc. |
| Graduation_Year | 2024 / 2025 etc. |
| CGPA | GPA out of 10 |
| Skills | Comma-separated skill list |
| Projects | Project title and description |
| Placed_Company | Company name |
| Package_LPA | CTC in LPA |

## Replacing with Your Own Data

Upload your `placements.xlsx` with the same column names. The app auto-reads it on startup.

## Running Locally

```bash
pip install -r requirements.txt
python app.py
```
