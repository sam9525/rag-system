"""Tests for HeadingDetector module."""

import pytest
from src.heading_detector import HeadingDetector, RegexHeadingDetector


class TestHeadingDetector:
    """Test HeadingDetector interface."""

    def test_detector_interface_is_abstract(self):
        """Test HeadingDetector cannot be instantiated directly."""
        with pytest.raises(TypeError):
            HeadingDetector()

    def test_detector_has_abstract_is_heading_method(self):
        """Test HeadingDetector has abstract is_heading method."""
        # Verify the abstract method exists
        assert hasattr(HeadingDetector, 'is_heading')
        # Verify it's an abstract method
        import inspect
        assert getattr(HeadingDetector.is_heading, '__isabstractmethod__', False)


class TestRegexHeadingDetector:
    """Test RegexHeadingDetector implementation."""

    def test_detects_markdown_heading(self):
        """Test detection of markdown headings."""
        detector = RegexHeadingDetector()
        assert detector.is_heading("## Section Title")
        assert detector.is_heading("# Main Title")
        assert detector.is_heading("### Sub Section")

    def test_detects_all_caps_heading(self):
        """Test detection of ALL CAPS headings."""
        detector = RegexHeadingDetector()
        assert detector.is_heading("INTRODUCTION AND BACKGROUND")
        assert detector.is_heading("METHODOLOGY AND APPROACH")

    def test_rejects_plain_text(self):
        """Test rejection of non-heading text."""
        detector = RegexHeadingDetector()
        assert not detector.is_heading("This is regular text")
        assert not detector.is_heading("Some paragraph content here")