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
    # --- Model Misuse & Supply Chain ---
    EvalCase(
        question="What are the risks of model misuse in AI applications?",
        ground_truth="Model misuse occurs when AI systems are exploited for unintended or harmful purposes, such as generating disinformation, automating attacks, bypassing security controls, or circumventing ethical safeguards designed into the model.",
        expected_topics=["misuse", "disinformation", "attacks", "safeguards"],
        source_hint="OWASP Top 10 for LLM"
    ),
    EvalCase(
        question="What is the AI model supply chain and its security implications?",
        ground_truth="The AI model supply chain includes model weights, training data, preprocessing pipelines, and third-party components. Security risks include poisoned training data, compromised model weights, unauthorized model modifications, and reliance on unverified third-party models.",
        expected_topics=["supply chain", "training data", "model weights", "third-party"],
        source_hint="AI Security"
    ),
    # --- Hallucination & Context Poisoning ---
    EvalCase(
        question="What is hallucination in LLM outputs and why is it a security concern?",
        ground_truth="Hallucination in LLMs refers to the model generating confident but incorrect, fabricated, or misleading information. In security contexts, this is dangerous because users may trust and act on false information, leading to security misconfigurations, incorrect vulnerability assessments, or compliance violations.",
        expected_topics=["hallucination", "confident", "incorrect", "misleading"],
        source_hint="OWASP Top 10 for LLM"
    ),
    EvalCase(
        question="What is context poisoning in agentic AI systems?",
        ground_truth="Context poisoning occurs when an attacker manipulates the input context of an AI agent to influence its behavior, memory, or decisions. This can happen through malicious inputs stored in memory, manipulated conversation history, or crafted prompts that gradually shift the agent's goals or bypass safety measures.",
        expected_topics=["context", "poisoning", "memory", "manipulation"],
        source_hint="Securing Agentic Applications Guide"
    ),
]