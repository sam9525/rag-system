"""Streamlit web UI for RAG System.
Provides a simple interface for business users to query documents.
"""

import streamlit as st
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Default sources directory - relative to script location
DEFAULT_SOURCES_DIR = Path(__file__).parent / "sources"

from src.system.rag_system import RAGSystem
from src.system.config import RAGConfig, GenerationConfig

# Available models (name, size, speed)
# Format: (display_name, model_name)
AVAILABLE_MODELS = [
    ("Qwen 3.5 4B (Fast)", "qwen3.5:4b"),
    ("Qwen 3 8B (Balanced)", "qwen3:8b"),
    ("Gemma 3 4B (Good)", "gemma3:4b"),
    ("Gemma 2 9B (Quality)", "gemma2:9b"),
    ("Gemma 4 4B (Default)", "gemma4:e4b"),  # Default
    ("Llama 3.2 3B (Fast)", "llama3.2:3b"),
    ("Mistral 7B (Quality)", "mistral:7b"),
]

# Default model (Gemma 4 4B)
DEFAULT_MODEL_NAME = "gemma4:e4b"
# Derive index from model name for maintainability
DEFAULT_MODEL_INDEX = next(
    (i for i, m in enumerate(AVAILABLE_MODELS) if m[1] == DEFAULT_MODEL_NAME),
    4,  # Fallback to index 4 if not found
)

# Module-level config instance
_config = RAGConfig()

# Set default model to Gemma 4
_update = AVAILABLE_MODELS[DEFAULT_MODEL_INDEX]
_config.generation = GenerationConfig(
    base_url="http://localhost:11434",
    model=_update[1],
    temperature=0.3,
    max_tokens=512,
    stream=False,
)


def update_generation_config(temperature: float, model_index: int) -> None:
    """Update the generation config with sidebar selections."""
    model_key = AVAILABLE_MODELS[model_index]
    _config.generation = GenerationConfig(
        base_url="http://localhost:11434",
        model=model_key[1],
        temperature=temperature,
        max_tokens=512,
        stream=False,
    )


# Page configuration
st.set_page_config(page_title="RAG Knowledge Assistant", page_icon="📖", layout="wide")


def init_session_state():
    """Initialize Streamlit session state."""
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "model_index" not in st.session_state:
        st.session_state.model_index = DEFAULT_MODEL_INDEX
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.3


def sidebar_config():
    """Render sidebar configuration."""
    with st.sidebar:
        st.title("Configuration")

        st.subheader("Document Source")
        source_dir = st.text_input(
            "Source Directory",
            value=str(DEFAULT_SOURCES_DIR),
            help="Path to folder containing PDF documents",
        )

        st.subheader("Generation Settings")

        # Model selection
        model_names = [m[0] for m in AVAILABLE_MODELS]
        model_index = st.selectbox(
            "Model",
            options=range(len(AVAILABLE_MODELS)),
            index=int(st.session_state.model_index),
            format_func=lambda i: model_names[i],
            help="Smaller/faster models for quick responses, larger for quality",
            key="model_selector",
        )

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.temperature,
            step=0.1,
            help="Lower = more focused, Higher = more creative",
        )

# Update config only when settings change
        if (
            model_index != st.session_state.model_index
            or abs(temperature - st.session_state.temperature) > 0.05
        ):
            st.session_state.model_index = model_index
            st.session_state.temperature = temperature
            update_generation_config(temperature, model_index)

        st.subheader("Actions")
        reindex = st.button("Re-index Documents", type="primary")

        return source_dir, reindex


def render_chat_history():
    """Render chat history."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def render_sources(sources, show_citations=True):
    """Render source documents with numbered citations."""
    with st.expander("📄 View Source Chunks"):
        for i, source in enumerate(sources, 1):
            citation = f"[{i}]" if show_citations else f"Chunk {i}"
            st.markdown(f"**{citation}** (Score: {source['score']:.3f})")
            st.markdown(f"- **Source:** {source['source']}")
            st.markdown(f"- **Page:** {source['page']}")
            if source.get("section"):
                st.markdown(f"- **Section:** {source['section']}")
            st.markdown(f"```\n{source['text']}\n```")
            st.divider()


def main():
    """Main application."""
    init_session_state()

    # Sidebar
    source_dir, reindex = sidebar_config()

    # Main content
    st.title("RAG Knowledge Assistant")
    st.markdown(
        "Ask questions about your documents. The system will search and generate answers."
    )

    # Initialize RAG system
    if st.session_state.rag_system is None or reindex:
        with st.spinner("Initializing RAG system..."):
            try:
                st.session_state.rag_system = RAGSystem(
                    source_dir=Path(source_dir),
                    config=_config,
                )
                stats = st.session_state.rag_system.ingest_documents(Path(source_dir))
                st.success(
                    f"Indexed {stats['documents_loaded']} documents with {stats['chunks_created']} chunks"
                )
            except Exception as e:
                st.error(f"Error initializing RAG system: {e}")
                st.info(
                    "Check that Ollama is running and the selected model is available."
                )
                return

    # System stats
    stats = st.session_state.rag_system.get_stats()
    st.caption(
        f"Model: {stats['model']} | Embedding: {stats['embedding_model']} | Documents: {stats['document_count']}"
    )

    # Chat history
    render_chat_history()

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching and generating..."):
                try:
                    result = st.session_state.rag_system.query(prompt)

                    st.markdown(f"**Answer:**\n{result.answer}")

                    if result.sources:
                        st.markdown("**Sources:**")
                        for i, source in enumerate(result.sources, 1):
                            st.markdown(
                                f"- **[{i}]** {source['source']}, Page {source['page']}"
                            )

                        render_sources(result.sources)

                    st.session_state.messages.append(
                        {"role": "assistant", "content": result.answer}
                    )

                except Exception as e:
                    st.error(f"Generation failed: {e}")
                    st.info(
                        "Try selecting a faster model (Qwen or Llama 3.2) in the sidebar, "
                        "or check that Ollama is running."
                    )

                    # Show sources even when generation fails
                    try:
                        retrieved = st.session_state.rag_system.retrieve(
                            prompt, top_k=5
                        )
                        if retrieved:
                            st.markdown("**Retrieved Chunks (for debugging):**")
                            for i, chunk in enumerate(retrieved, 1):
                                st.markdown(
                                    f"- **[{i}]** {chunk['source']}, Page {chunk['page']}"
                                )
                            render_sources(retrieved, show_citations=True)
                    except Exception:
                        st.warning("Retrieval also failed.")


if __name__ == "__main__":
    main()
