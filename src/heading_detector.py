"""Heading detection interface and implementations."""

import re
from abc import ABC, abstractmethod


class HeadingDetector(ABC):
    """Abstract interface for heading detection."""

    @abstractmethod
    def is_heading(self, line: str) -> bool:
        """Check if a line is a heading.

        Args:
            line: Text line to check.

        Returns:
            True if the line is a heading, False otherwise.
        """
        pass


class RegexHeadingDetector(HeadingDetector):
    """Heading detector using regex patterns."""

    DEFAULT_PATTERNS = [
        r'^#{1,3}\s+(.+)$',  # Markdown headings
        r'^([A-Z][A-Z\s]{5,})\s*$',  # ALL CAPS headings
        r'^(\d+\.\s+[A-Z].+)$',  # Numbered sections
    ]

    def __init__(self, patterns: list = None):
        """Initialize with custom patterns.

        Args:
            patterns: List of regex patterns. Uses defaults if None.
        """
        self.patterns = [re.compile(p) for p in (patterns or self.DEFAULT_PATTERNS)]

    def is_heading(self, line: str) -> bool:
        """Check if line matches any heading pattern."""
        line = line.strip()
        for pattern in self.patterns:
            if pattern.match(line):
                return True
        return False