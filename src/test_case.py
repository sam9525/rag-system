"""Test case dataclass and evaluation cases for RAG system."""

from dataclasses import dataclass
from typing import List


@dataclass
class EvalCase:
    """A test case for evaluating RAG system responses.

    Attributes:
        question: The question to ask the RAG system.
        ground_truth: Reference answer that represents ideal response.
        expected_topics: Key concepts that should appear in a good answer.
        source_hint: Which document/section this relates to (for debugging).
    """

    question: str
    expected_topics: List[str]
    source_hint: str = ""


# --- Core Security Test Cases ---
CASES = [
    # --- OWASP Top 10 for LLM ---
    EvalCase(
        question="How does prompt injection manipulate the original system prompts to change model behavior?",
        expected_topics=["prompt", "injection", "LLM", "vulnerability"],
        source_hint="OWASP Top 10 for LLM Applications",
    ),
    # --- Agentic Security ---
    EvalCase(
        question="What specific security risks are associated with an AI agent's persistent memory?",
        expected_topics=[
            "prompt injection",
            "privilege escalation",
            "data poisoning",
            "hallucinations",
            "persistent memory",
        ],
        source_hint="Securing Agentic Applications Guide",
    ),
    EvalCase(
        question="If an autonomous AI agent is compromised via prompt injection, what specific controls prevent it from escalating privileges?",
        expected_topics=["privilege", "escalation", "capabilities", "authorization"],
        source_hint="Agentic Security",
    ),
    # --- Red Teaming ---
    EvalCase(
        question="Why do organizations need AI red teaming if they already use standard vulnerability scanning?",
        expected_topics=[
            "red team",
            "adversarial",
            "testing",
            "vulnerabilities",
            "limitations",
        ],
        source_hint="AI Red Teaming Guide",
    ),
    # --- Model Misuse & Supply Chain ---
    EvalCase(
        question="What are two examples of how model misuse can bypass built-in ethical safeguards?",
        expected_topics=["misuse", "disinformation", "attacks", "safeguards"],
        source_hint="OWASP Top 10 for LLM",
    ),
    # --- AI in Offensive and Defensive Security ---
    EvalCase(
        question="What are the dual-use security concerns with AI systems?",
        expected_topics=[
            "dual-use",
            "offensive",
            "defensive",
            "automation",
            "AI-guided",
        ],
        source_hint="AI Security - Dual-Use",
    ),
    EvalCase(
        question="What are the key differences between offensive and defensive AI security approaches?",
        expected_topics=[
            "offensive",
            "defensive",
            "red team",
            "zero-trust",
            "threat detection",
        ],
        source_hint="AI Security - Offensive vs Defensive",
    ),
    EvalCase(
        question="What security controls are needed to defend against AI-guided attacks?",
        expected_topics=[
            "zero-trust",
            "least privilege",
            "input validation",
            "threat modeling",
            "resilience",
        ],
        source_hint="AI and Zero-Trust Roadmap",
    ),
    EvalCase(
        question="How do AI security frameworks address both offensive and defensive considerations?",
        expected_topics=[
            "NIST",
            "OWASP",
            "framework",
            "threat modeling",
            "supply chain",
        ],
        source_hint="AI Security Frameworks",
    ),
    EvalCase(
        question="What is model poisoning and how can it affect both offensive and defensive AI systems?",
        expected_topics=[
            "model poisoning",
            "supply chain",
            "training data",
            "model provenance",
        ],
        source_hint="AI Supply Chain Security",
    ),
]
