"""Streamlit web UI for RAG System.
Provides a simple interface for business users to query documents.
"""

import streamlit as st
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.rag_system import RAGSystem
from src.config import RAGConfig

# Module-level config instance
_rag_config = RAGConfig()

# Page configuration
st.set_page_config(page_title="RAG Knowledge Assistant", page_icon="📚", layout="wide")


def init_session_state():
    """Initialize Streamlit session state."""
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None
    if "messages" not in st.session_state:
        st.session_state.messages = []


def sidebar_config():
    """Render sidebar configuration."""
    with st.sidebar:
        st.title("📚 Configuration")

        st.subheader("Document Source")
        source_dir = st.text_input(
            "Source Directory",
            value="sources",
            help="Path to folder containing PDF documents",
        )

        st.subheader("Generation Settings")
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=_rag_config.generation.temperature,
            step=0.1,
            help="Lower = more focused, Higher = more creative",
        )

        st.subheader("Actions")
        reindex = st.button("🔄 Re-index Documents", type="primary")

        return source_dir, temperature, reindex


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
    source_dir, temperature, reindex = sidebar_config()

    # Main content
    st.title("📚 RAG Knowledge Assistant")
    st.markdown(
        "Ask questions about your documents. The system will search and generate answers."
    )

    # Initialize RAG system
    if st.session_state.rag_system is None or reindex:
        with st.spinner("Initializing RAG system..."):
            try:
                st.session_state.rag_system = RAGSystem(
                    source_dir=Path(source_dir),
                    config=_rag_config,
                )
                stats = st.session_state.rag_system.ingest_documents(Path(source_dir))
                st.success(
                    f"Indexed {stats['documents_loaded']} documents with {stats['chunks_created']} chunks"
                )
            except Exception as e:
                st.error(f"Error: {e}")
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
                    st.error(f"Error: {e}")
                    st.info(
                        "Showing top retrieved chunks while we resolve the issue..."
                    )

                    # Show sources even when generation fails
                    try:
                        retrieved = st.session_state.rag_system.retrieve(
                            prompt, top_k=5
                        )
                        if retrieved:
                            st.markdown("**Retrieved Chunks:**")
                            for i, chunk in enumerate(retrieved, 1):
                                st.markdown(
                                    f"- **[{i}]** {chunk['source']}, Page {chunk['page']}"
                                )
                            render_sources(retrieved, show_citations=True)
                    except Exception:
                        pass  # Best effort - sources may not be available


if __name__ == "__main__":
    main()
