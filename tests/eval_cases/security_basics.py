"""Core security concepts test cases.

These test cases cover fundamental security concepts from the OWASP
documentation and security best practices.
"""

from src.test_case import EvalCase

CASES = [
    # --- OWASP Top 10 for LLM ---
    EvalCase(
        question="What is prompt injection in LLM applications?",
        ground_truth="Prompt injection is a vulnerability where attackers inject malicious content into LLM inputs to manipulate model behavior, causing it to follow attacker-provided instructions rather than the original system prompts.",
        expected_topics=["prompt", "injection", "LLM", "vulnerability"],
        source_hint="OWASP Top 10 for LLM Applications"
    ),
    EvalCase(
        question="What does the OWASP Top 10 for LLM say about data leakage?",
        ground_truth="Data leakage in LLM applications occurs when sensitive information from training data, prompts, or responses is exposed to unauthorized users, either through model outputs or inference artifacts.",
        expected_topics=["data", "leakage", "sensitive", "training"],
        source_hint="OWASP Top 10 for LLM"
    ),
    EvalCase(
        question="What is insecure output handling?",
        ground_truth="Insecure output handling occurs when LLM output is displayed to users without proper sanitization, potentially exposing the system to XSS, injection attacks, or leakage of sensitive data.",
        expected_topics=["output", "sanitization", "XSS", "injection"],
        source_hint="OWASP Top 10 for LLM"
    ),
    # --- Agentic Security ---
    EvalCase(
        question="What are the main security concerns for AI agents?",
        ground_truth="AI agents face security concerns including prompt injection, model misuse, agent privilege escalation, data poisoning, hallucinations, and emergent behaviors. Additional risks include context poisoning, data leakage, and unauthorized state retention from persistent memory.",
        expected_topics=["prompt injection", "privilege escalation", "data poisoning", "hallucinations"],
        source_hint="Securing Agentic Applications Guide"
    ),
    EvalCase(
        question="What is agent privilege escalation?",
        ground_truth="Agent privilege escalation occurs when an AI agent gains capabilities or access beyond what was intended by its design or authorization, potentially allowing it to perform actions it should not be permitted to do.",
        expected_topics=["privilege", "escalation", "capabilities", "authorization"],
        source_hint="Agentic Security"
    ),
    # --- Red Teaming ---
    EvalCase(
        question="What is red teaming in the context of AI security?",
        ground_truth="Red teaming in AI security is a structured adversarial testing approach where teams simulate attacks on AI systems to identify vulnerabilities, measure risks, and provide feedback for improving defenses.",
        expected_topics=["red team", "adversarial", "testing", "vulnerabilities"],
        source_hint="AI Red Teaming Guide"
    ),
]