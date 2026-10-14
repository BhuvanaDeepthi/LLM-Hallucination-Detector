# LLM Hallucination Detector: Verifying AI-Generated Answers Using RAG and NLI Models

A complete, production-grade College Mini-Project designed to detect, classify, and audit hallucinations in Large Language Model (LLM) responses using **Retrieval-Augmented Generation (RAG)**, **Atomic Claim Decomposition**, and **Natural Language Inference (NLI)**.

---

## 📌 Project Overview

Large Language Models (LLMs) such as ChatGPT or Gemma frequently produce answers that sound fluent and authoritative, yet contain subtle factual falsehoods or completely fabricated details. In sensitive applications (medical, legal, technical), these hallucinations present significant operational risks.

The **LLM Hallucination Detector** solves this by:
1. Grounding AI responses in authoritative source documents (PDFs).
2. Decomposing generated answers into independent, atomic factual claims.
3. Cross-verifying each claim against retrieved document evidence using a specialized Natural Language Inference model (`roberta-large-mnli`).
4. Assigning calibrated, honest verdicts: **Supported**, **Contradicted**, or **Unsupported**.
5. Providing full citation trails (page numbers, similarity scores, premise quotes, and probability distributions).
6. Offering an optional second-stage **LLM-as-a-Judge** for ambiguous classifications.

---

## 🏗️ System Architecture & Workflow

```
                                  +-----------------------------+
                                  |   User Uploads PDF Doc      |
                                  +--------------+--------------+
                                                 |
                                                 | PyMuPDF (Page-level text extraction)
                                                 v
                                  +-----------------------------+
                                  |  Page-Aware Text Chunker    |
                                  +--------------+--------------+
                                                 |
                                                 | Sentence-Transformers (all-MiniLM-L6-v2)
                                                 v
+-----------------------------+   +-----------------------------+
|        User Question        |-->|     FAISS Vector Store      |
+-----------------------------+   +--------------+--------------+
                                                 |
                                                 | Top-K Cosine Retrieval
                                                 v
                                  +-----------------------------+
                                  | RAG Answer Generation       | (Local Ollama Gemma 3 / Gemini)
                                  +--------------+--------------+
                                                 |
                                                 | Structured Prompt Decomposer
                                                 v
                                  +-----------------------------+
                                  |   Atomic Claim Extraction   |
                                  +--------------+--------------+
                                                 |
                                                 | Premise (Evidence) vs Hypothesis (Claim)
                                                 v
                                  +-----------------------------+
                                  |  RoBERTa-large-MNLI Engine  |
                                  | [Entailment, Contradict, N] |
                                  +--------------+--------------+
                                                 |
                                                 | Calibrated Thresholds (0.50) + Optional LLM Judge
                                                 v
                                  +-----------------------------+
                                  | Results Dashboard & Audit   |
                                  | [Supported / Contra / Unsup]|
                                  +--------------+--------------+
                                                 |
                                                 | SQLite Storage + Report Generator
                                                 v
                                  +-----------------------------+
                                  | Export Audit Report (.md)   |
                                  +-----------------------------+
```

---

## 🚀 Key Features

1. **Modern AI SaaS Interface:** Built with Streamlit using custom dark-mode aesthetics, responsive metric cards, and badge indicators.
2. **Page-Preserving PDF Processing:** PyMuPDF extracts text while strictly tracking original page numbers for accurate evidence attribution.
3. **MiniLM & FAISS Vector Search:** High-performance dense retrieval with normalized embeddings and inner-product similarity search.
4. **Local Gemma 3 Integration:** Connects directly to local Ollama (`gemma3:latest`), ensuring complete data privacy and 100% offline functionality.
5. **Optional Google Gemini API Support:** Securely configures via `.env` (`GEMINI_API_KEY`) with graceful fallback to local Ollama if offline.
6. **Honest Verdict Classification:**
   - **Supported (Green):** Text provides direct, sufficient entailment ($\ge 0.50$).
   - **Contradicted (Red):** Evidence directly contradicts or refutes the claim ($\ge 0.50$).
   - **Unsupported (Amber):** Absence of evidence or neutral relationship. Does not treat lack of evidence as proof of falsehood.
7. **Optional LLM-as-a-Judge:** Employs Gemma as a second-stage adjudicator for borderline NLI distributions.
8. **Persistent SQLite Audit History:** Tracks every session, question, answer, claim, verdict, and citation.
9. **Exportable Audit Reports:** Generates clean, timestamped Markdown (`.md`) and plain text reports.
10. **HaluEval Benchmark Ready:** Contains a modular evaluation architecture (`modules/evaluation.py`) for benchmarking precision, recall, F1-score, and false-positive rates on reference datasets.

---

## 🛠️ Technology Stack

| Layer | Component / Library | Details |
| :--- | :--- | :--- |
| **User Interface** | Streamlit | Custom Dark Navy / Indigo UI with CSS design tokens |
| **PDF Extraction** | PyMuPDF (`pymupdf`) | Fast page-by-page text parsing with validation |
| **Embeddings** | `sentence-transformers` | `all-MiniLM-L6-v2` (384-dimensional dense vectors) |
| **Vector Index** | FAISS (`faiss-cpu`) | Inner-product similarity search |
| **Generative LLM** | Ollama (`gemma3:latest`) | Local answer generation & claim decomposition |
| **Optional LLM** | Google Gemini API (`google-genai`) | Optional cloud alternative via `.env` |
| **NLI Classifier** | Hugging Face Transformers | `roberta-large-mnli` (3-class sequence classification) |
| **Database** | SQLite 3 | Embedded database (`data/detector.db`) |
| **Testing** | Pytest | 14 automated unit and integration tests |
| **Evaluation** | Scikit-learn | Precision, Recall, F1, FPR, Confusion Matrix |

---

## 💻 Installation & Setup Guide

### 1. Prerequisites
- **Python 3.10+** (Tested on Python 3.12)
- **Ollama** installed with `gemma3:latest`:
  ```bash
  ollama list
  # Should show gemma3:latest
  ```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Configuration (Optional)
If you wish to use Google Gemini Cloud API as an alternative to local Ollama, create a `.env` file from the example:
```bash
copy .env.example .env
```
Open `.env` and insert your Gemini API Key:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```
*Note: If no API key is provided, the detector functions entirely offline using local Ollama `gemma3:latest`.*

---

## 🧪 Running Automated Tests

Run the full Pytest test suite:
```bash
pytest tests/ -v
```
All 14 tests covering PDF extraction, chunking, claim extraction, NLI verdict mapping, and SQLite storage will execute.

---

## 🚀 Launching the Application

Start the Streamlit application:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📖 Walkthrough: Step-by-Step Usage

1. **Upload & Verify Workspace (All-in-One):**
   - **Step 1 (Upload):** Upload any text-based PDF and click **"Process & Index Document"**.
   - **Step 2 (Ask & Verify):** Enter your question on the same page and click **"Generate Answer & Verify Claims"**.
   - **Step 3 (Results):** View the verification results dashboard with overall verdict banner, metric cards, individual claim cards (color-coded green/red/amber), page citations, evidence quotes, and downloadable audit reports.
2. **Home Dashboard:**
   - View live database statistics (documents processed, verification sessions, claims verified, claims flagged), workflow overview, and verdict criteria definitions.
3. **Audit History:**
   - Review past verification sessions stored in SQLite.
   - Inspect individual claims, view evidence passages, delete sessions, or export audit reports.

---

## 🎓 College Presentation & Viva Questions

### Q1: Why use an NLI model instead of asking an LLM "Is this true?"
> Generative LLMs exhibit severe self-confirmation bias and prompt-sensitivity. When asked to critique their own output, they frequently double-down on hallucinations. In contrast, `roberta-large-mnli` is an encoder model trained strictly on the Premise-Hypothesis entailment task, providing objective, deterministic probability distributions.

### Q2: What is the difference between "Contradicted" and "Unsupported"?
> - **Contradicted:** The source document directly conflicts with the claim (e.g., Document states 2020; Model asserts 1995).
> - **Unsupported:** The source document does not contain evidence about the claim (Neutral). In logic and law, absence of evidence is not proof of falsehood. Treating neutral statements as false is a common flaw in naive verification systems.

### Q3: Why is claim decomposition necessary?
> Multi-sentence answers often blend accurate information with one or two fabricated details. If an answer is checked as a single monolithic block, the entire response may be falsely rejected or accepted. Breaking it into atomic claims allows granular, claim-by-claim verification.

---

## 📂 Project Structure

```
llm-hallucination-detector/
├── app.py                     # Streamlit application entry point
├── styles.py                  # Custom CSS styling & themes
├── config.py                  # Global configurations, thresholds & paths
├── requirements.txt           # Project dependencies
├── .env.example               # Environment variable template
├── .gitignore                 # Git ignore rules
├── README.md                  # Comprehensive documentation
├── modules/
│   ├── __init__.py
│   ├── pdf_processor.py       # PyMuPDF page-aware extraction
│   ├── text_chunker.py        # Sliding-window passage chunker
│   ├── embeddings.py          # MiniLM Sentence-Transformers wrapper
│   ├── vector_store.py        # FAISS vector index & metadata store
│   ├── rag_pipeline.py        # Answer generation via Ollama / Gemini
│   ├── claim_extractor.py     # Atomic claim decomposition & parsing
│   ├── nli_verifier.py        # RoBERTa-large-MNLI verification engine
│   ├── llm_judge.py           # Optional second-stage LLM judge
│   ├── report_generator.py    # Markdown & plain-text report generation
│   ├── database.py            # SQLite session & claim persistence
│   └── evaluation.py          # Benchmark evaluation architecture (HaluEval)
├── pages/
│   ├── dashboard.py           # Home dashboard with live metrics
│   ├── upload_document.py     # PDF upload and indexing UI
│   ├── ask_question.py        # RAG query & verification results dashboard
│   ├── verification_history.py# Historical audit session review
│   └── about.py               # Technical architecture & viva guide
├── tests/
│   ├── __init__.py
│   ├── test_pdf_processor.py  # PDF validation & extraction tests
│   ├── test_text_chunker.py   # Chunking & overlap tests
│   ├── test_claim_extractor.py# JSON parsing & claim fallback tests
│   ├── test_nli_verifier.py   # NLI decision mapping tests
│   └── test_database.py       # SQLite persistence tests
└── data/
    ├── uploads/               # Uploaded PDFs
    ├── indexes/               # FAISS index storage
    └── detector.db            # SQLite database file
```

---

## 📄 License
Academic and Educational Use — College Mini Project 2026.
