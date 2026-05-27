"""Tests for streamlit_app.py model selector configuration."""

import pytest
from pathlib import Path


class TestModelSelectorConfig:
    """Tests for model selector configuration in streamlit_app."""

    def test_available_models_has_gemma4(self, import_streamlit_app):
        """Test that AVAILABLE_MODELS contains Gemma 4."""
        gemma4_models = [
            m for m in import_streamlit_app.AVAILABLE_MODELS if "gemma4" in m[1].lower()
        ]
        assert (
            len(gemma4_models) > 0
        ), "AVAILABLE_MODELS should have at least one Gemma 4 model"

    def test_default_model_index_points_to_gemma4(self, import_streamlit_app):
        """Test that DEFAULT_MODEL_INDEX points to a Gemma 4 model."""
        default_model = import_streamlit_app.AVAILABLE_MODELS[
            import_streamlit_app.DEFAULT_MODEL_INDEX
        ]
        assert (
            "gemma4" in default_model[1].lower()
        ), f"DEFAULT_MODEL_INDEX should point to Gemma 4, got {default_model}"

    def test_default_model_name_follows_ollama_convention(self, import_streamlit_app):
        """Test that DEFAULT_MODEL_NAME follows Ollama naming convention."""
        model_name = import_streamlit_app.DEFAULT_MODEL_NAME
        # Ollama model names should have format: name:tag or name:version
        assert (
            ":" in model_name
        ), f"DEFAULT_MODEL_NAME should follow Ollama naming (name:tag), got {model_name}"
        assert (
            model_name
            == import_streamlit_app.AVAILABLE_MODELS[
                import_streamlit_app.DEFAULT_MODEL_INDEX
            ][1]
        ), "DEFAULT_MODEL_NAME should match the model at DEFAULT_MODEL_INDEX"

    def test_update_generation_config_changes_config(self, import_streamlit_app):
        """Test that update_generation_config() properly changes the config."""
        # Get initial config
        initial_model = import_streamlit_app._config.generation.model
        initial_temp = import_streamlit_app._config.generation.temperature

        # Update config with a different model (index 0 = first model)
        import_streamlit_app.update_generation_config(temperature=0.7, model_index=0)

        # Verify the config changed
        new_model = import_streamlit_app._config.generation.model
        new_temp = import_streamlit_app._config.generation.temperature

        assert (
            new_model != initial_model
        ), "update_generation_config should change the model"
        assert (
            new_model == import_streamlit_app.AVAILABLE_MODELS[0][1]
        ), f"Model should be {import_streamlit_app.AVAILABLE_MODELS[0][1]}, got {new_model}"
        assert new_temp == 0.7, f"Temperature should be 0.7, got {new_temp}"

    def test_model_list_has_descriptive_labels(self, import_streamlit_app):
        """Test that model display names have descriptive labels in parentheses."""
        for model in import_streamlit_app.AVAILABLE_MODELS:
            display_name = model[0]
            # Check for parentheses with descriptive text
            assert (
                "(" in display_name and ")" in display_name
            ), f"Display name should have descriptive label in parentheses: {display_name}"

    def test_model_count_is_reasonable(self, import_streamlit_app):
        """Test that model count is between 5 and 10."""
        model_count = len(import_streamlit_app.AVAILABLE_MODELS)
        assert 5 <= model_count <= 10, f"Model count should be 5-10, got {model_count}"

    def test_default_sources_dir_is_path(self, import_streamlit_app):
        """Test that DEFAULT_SOURCES_DIR is a Path object."""
        sources_dir = import_streamlit_app.DEFAULT_SOURCES_DIR
        assert isinstance(
            sources_dir, Path
        ), f"DEFAULT_SOURCES_DIR should be a Path object, got {type(sources_dir)}"

    def test_default_sources_dir_points_to_sources(self, import_streamlit_app):
        """Test that DEFAULT_SOURCES_DIR points to 'sources' directory."""
        sources_dir = import_streamlit_app.DEFAULT_SOURCES_DIR
        assert (
            sources_dir.name == "sources"
        ), f"DEFAULT_SOURCES_DIR should point to 'sources' directory, got {sources_dir}"

    def test_default_sources_dir_parent_exists(self, import_streamlit_app):
        """Test that DEFAULT_SOURCES_DIR parent directory exists."""
        sources_dir = import_streamlit_app.DEFAULT_SOURCES_DIR
        assert (
            sources_dir.parent.exists()
        ), f"DEFAULT_SOURCES_DIR parent should exist: {sources_dir.parent}"


@pytest.fixture
def import_streamlit_app():
    """Fixture to import streamlit_app module."""
    import sys
    from pathlib import Path

    # Add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    # Use importlib.util to load the module without running main
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "streamlit_app", project_root / "streamlit_app.py"
    )
    module = importlib.util.module_from_spec(spec)

    # Mock st module to avoid actually running Streamlit
    import types

    st_mock = types.ModuleType("streamlit")
    for attr in [
        "set_page_config",
        "title",
        "markdown",
        "sidebar",
        "subheader",
        "text_input",
        "selectbox",
        "slider",
        "button",
        "expander",
        "divider",
        "chat_message",
        "chat_input",
        "spinner",
        "success",
        "error",
        "info",
        "warning",
        "caption",
    ]:
        setattr(st_mock, attr, lambda *args, **kwargs: None)
    sys.modules["streamlit"] = st_mock

    # Mock RAGSystem
    mock_rag_system = types.ModuleType("src.system.rag_system")
    mock_rag_system.RAGSystem = type("RAGSystem", (), {})
    sys.modules["src.system.rag_system"] = mock_rag_system

    # Import actual config module
    sys.modules["src.system.config"] = __import__(
        "src.system.config", fromlist=["RAGConfig", "GenerationConfig"]
    )

    spec.loader.exec_module(module)
    return module
