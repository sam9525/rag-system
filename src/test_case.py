"""Test case dataclass and evaluation cases for RAG system."""

from dataclasses import dataclass


@dataclass
class EvalCase:
    """A test case for evaluating RAG system responses.

    Attributes:
        question: The question to ask the RAG system.
    """

    question: str


# --- Core Security Test Cases ---
CASES = [
    # --- OWASP Top 10 for LLM ---
    EvalCase(
        question="How does prompt injection manipulate the original system prompts to change model behavior?",
    ),
    # --- Agentic Security ---
    EvalCase(
        question="What specific security risks are associated with an AI agent's persistent memory?",
    ),
    EvalCase(
        question="If an autonomous AI agent is compromised via prompt injection, what specific controls prevent it from escalating privileges?",
    ),
    # --- Red Teaming ---
    EvalCase(
        question="Why do organizations need AI red teaming if they already use standard vulnerability scanning?",
    ),
    # --- Model Misuse & Supply Chain ---
    EvalCase(
        question="What are two examples of how model misuse can bypass built-in ethical safeguards?",
    ),
    # --- AI in Offensive and Defensive Security ---
    EvalCase(
        question="What are the dual-use security concerns with AI systems?",
    ),
    EvalCase(
        question="What are the key differences between offensive and defensive AI security approaches?",
    ),
    EvalCase(
        question="What security controls are needed to defend against AI-guided attacks?",
    ),
    EvalCase(
        question="How do AI security frameworks address both offensive and defensive considerations?",
    ),
    EvalCase(
        question="What is model poisoning and how can it affect both offensive and defensive AI systems?",
    ),
]
