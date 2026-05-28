# RAG Knowledge Assistant

A production-ready Retrieval-Augmented Generation (RAG) system with hybrid search capabilities, neural reranking, and RAGAS evaluation metrics.

## Features

- **Hybrid Retrieval**: Combines semantic search (embeddings) and keyword search (BM25) using Reciprocal Rank Fusion (RRF)
- **Neural Reranking**: Optional cross-encoder reranking for improved relevance
- **Multiple LLM Support**: Works with any Ollama model (Qwen, Gemma, Llama, Mistral)
- **RAGAS Evaluation**: Built-in faithfulness, answer relevancy, and context relevance metrics
- **Caching**: Automatic index caching with dirty-file detection for efficient re-indexing
- **Streamlit UI**: User-friendly interface for business users

## Architecture

```
src/
├── evaluation/     # RAGAS evaluation metrics
├── generation/     # Ollama LLM integration
├── ingestion/     # PDF loading and text chunking
├── retrieval/      # Hybrid search, embedding, reranking
├── storage/        # FAISS index and chunk management
└── system/        # RAG orchestrator and config
```

### Core Components

| Component         | Purpose                                            |
| ----------------- | -------------------------------------------------- |
| `DocumentLoader`  | Extracts text from PDFs with page/section tracking |
| `ChunkStore`      | Single source of truth for chunk text and metadata |
| `HybridRetriever` | Orchestrates semantic + keyword + neural retrieval |
| `OllamaGenerator` | Generates answers using retrieved context          |
| `RAGASEvaluator`  | Computes faithfulness, relevancy, context metrics  |

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Ollama

```bash
ollama serve
ollama pull gemma4:e4b  # Or any model from AVAILABLE_MODELS
```

### 3. Add Documents

Place PDF files in the `sources/` directory (auto-created on first run).

### 4. Run the Application

```bash
streamlit run streamlit_app.py
```

## Configuration

Edit `src/system/config.py` or use the sidebar in the web UI:

| Setting             | Default                | Description                             |
| ------------------- | ---------------------- | --------------------------------------- |
| `embedding_model`   | BAAI/bge-large-en-v1.5 | Sentence transformer model              |
| `semantic_top_k`    | 50                     | Initial semantic search candidates      |
| `keyword_top_k`     | 50                     | Initial BM25 search candidates          |
| `final_top_k`       | 3                      | Final results returned                  |
| `rrf_k`             | 60                     | RRF fusion parameter                    |
| `use_neural_rerank` | false                  | Enable cross-encoder reranking          |
| `temperature`       | 0.3                    | LLM response randomness                 |
| `rerank_mode`       | "hybrid"               | Retrieval mode: hybrid/semantic/keyword |

### Rerank Modes

The sidebar lets you switch between three retrieval modes:

| Mode       | Description                                      |
| ---------- | ------------------------------------------------ |
| `hybrid`   | Combines semantic + keyword search (recommended) |
| `semantic` | Pure embedding-based semantic similarity search  |
| `keyword`  | BM25 keyword matching for exact term matching    |

## Available Models

| Model        | Speed    | Use Case         |
| ------------ | -------- | ---------------- |
| Qwen 3.5 4B  | Fast     | Quick responses  |
| Qwen 3 8B    | Balanced | General purpose  |
| Gemma 3 4B   | Good     | Quality + speed  |
| Gemma 2 9B   | Quality  | Higher accuracy  |
| Gemma 4 4B   | Default  | Balanced default |
| Llama 3.2 3B | Fast     | Quick responses  |
| Mistral 7B   | Quality  | Complex queries  |

## Evaluation

The system computes RAGAS metrics automatically:

- **Faithfulness**: Whether the answer is grounded in the retrieved context
- **Answer Relevancy**: How well the answer addresses the question
- **Context Relevance**: Quality of retrieved context chunks

View scores alongside each answer in the Streamlit UI.

## API Usage

```python
from src.system.rag_system import RAGSystem
from src.system.config import RAGConfig

# Initialize
rag = RAGSystem(source_dir=Path("sources"))
rag.ingest_documents(Path("sources"))

# Query with evaluation
from src.evaluation.evaluator import RAGASEvaluator
from src.evaluation.test_case import EvalCase

evaluator = RAGASEvaluator(rag, rerank_mode="hybrid")
result = evaluator.run_case(EvalCase(question="What is X?"))

print(result.answer)
print(f"Faithfulness: {result.faithfulness}")
```

## Testing

```bash
pytest tests/
```
