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
    ground_truth: str
    expected_topics: List[str]
    source_hint: str = ""


# --- Core Security Test Cases ---
CASES = [
    # --- OWASP Top 10 for LLM ---
    EvalCase(
        question="What is prompt injection in LLM applications?",
        ground_truth="Prompt injection is a vulnerability where attackers inject malicious content into LLM inputs to manipulate model behavior, causing it to follow attacker-provided instructions rather than the original system prompts.",
        expected_topics=["prompt", "injection", "LLM", "vulnerability"],
        source_hint="OWASP Top 10 for LLM Applications",
    ),
    EvalCase(
        question="What does the OWASP Top 10 for LLM say about data leakage?",
        ground_truth="Data leakage in LLM applications occurs when sensitive information from training data, prompts, or responses is exposed to unauthorized users, either through model outputs or inference artifacts.",
        expected_topics=["data", "leakage", "sensitive", "training"],
        source_hint="OWASP Top 10 for LLM",
    ),
    EvalCase(
        question="What is insecure output handling?",
        ground_truth="Insecure output handling occurs when LLM output is displayed to users without proper sanitization, potentially exposing the system to XSS, injection attacks, or leakage of sensitive data.",
        expected_topics=["output", "sanitization", "XSS", "injection"],
        source_hint="OWASP Top 10 for LLM",
    ),
    # --- Agentic Security ---
    EvalCase(
        question="What are the main security concerns for AI agents?",
        ground_truth="AI agents face security concerns including prompt injection, model misuse, agent privilege escalation, data poisoning, hallucinations, and emergent behaviors. Additional risks include context poisoning, data leakage, and unauthorized state retention from persistent memory.",
        expected_topics=[
            "prompt injection",
            "privilege escalation",
            "data poisoning",
            "hallucinations",
        ],
        source_hint="Securing Agentic Applications Guide",
    ),
    EvalCase(
        question="What is agent privilege escalation?",
        ground_truth="Agent privilege escalation occurs when an AI agent gains capabilities or access beyond what was intended by its design or authorization, potentially allowing it to perform actions it should not be permitted to do.",
        expected_topics=["privilege", "escalation", "capabilities", "authorization"],
        source_hint="Agentic Security",
    ),
    # --- Red Teaming ---
    EvalCase(
        question="What is red teaming in the context of AI security?",
        ground_truth="Red teaming in AI security is a structured adversarial testing approach where teams simulate attacks on AI systems to identify vulnerabilities, measure risks, and provide feedback for improving defenses.",
        expected_topics=["red team", "adversarial", "testing", "vulnerabilities"],
        source_hint="AI Red Teaming Guide",
    ),
    # --- Model Misuse & Supply Chain ---
    EvalCase(
        question="What are the risks of model misuse in AI applications?",
        ground_truth="Model misuse occurs when AI systems are exploited for unintended or harmful purposes, such as generating disinformation, automating attacks, bypassing security controls, or circumventing ethical safeguards designed into the model.",
        expected_topics=["misuse", "disinformation", "attacks", "safeguards"],
        source_hint="OWASP Top 10 for LLM",
    ),
    EvalCase(
        question="What is the AI model supply chain and its security implications?",
        ground_truth="The AI model supply chain includes model weights, training data, preprocessing pipelines, and third-party components. Security risks include poisoned training data, compromised model weights, unauthorized model modifications, and reliance on unverified third-party models.",
        expected_topics=[
            "supply chain",
            "training data",
            "model weights",
            "third-party",
        ],
        source_hint="AI Security",
    ),
    # --- Context Poisoning ---
    EvalCase(
        question="What is context poisoning in agentic AI systems?",
        ground_truth="Context poisoning occurs when an attacker manipulates the input context of an AI agent to influence its behavior, memory, or decisions. This can happen through malicious inputs stored in memory, manipulated conversation history, or crafted prompts that gradually shift the agent's goals or bypass safety measures.",
        expected_topics=["context", "poisoning", "memory", "manipulation"],
        source_hint="Securing Agentic Applications Guide",
    ),
]

# --- Dual-Use Dilemma: AI in Offensive vs. Defensive Security ---
CASES.extend(
    [
        EvalCase(
            question="What are the dual-use security concerns with AI systems?",
            ground_truth="AI systems present dual-use concerns because the same capabilities used for legitimate security tasks (threat detection, vulnerability scanning, automation) can be repurposed for offensive attacks (automated exploitation, AI-guided attacks). This includes LLM-guided tools like AutoAttacker that can automate reconnaissance, vulnerability analysis, and attack execution.",
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
            ground_truth="Offensive AI security focuses on identifying and exploiting vulnerabilities (automated attacks, red teaming, penetration testing), while defensive AI security focuses on protecting systems (threat detection, access control, incident response). Defensive approaches include zero-trust architecture, NIST CSF frameworks, and secure-by-design principles, while offensive approaches leverage AI for speed and scalability in attack execution.",
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
            ground_truth="Defenses against AI-guided attacks require: zero-trust architecture (never trust, always verify), continuous authentication, least-privilege access, AI-specific controls (input/output validation, model hardening), threat modeling for AI components, and framework-aligned practices like NIST CSF. Organizations should assume AI-powered attacks are inevitable and build resilient, layered defenses.",
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
            ground_truth="AI security frameworks like NIST CSF for AI and OWASP Top 10 for LLM/Agentic Applications provide defensive guidance (Govern, Identify, Protect, Detect, Respond, Recover) while acknowledging offensive realities. They address dual-use by recommending threat modeling, secure design principles, and supply chain security to mitigate risks from AI misuse and automated attack systems.",
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
            ground_truth="Model poisoning is a supply chain attack where malicious data or weights are introduced during training to compromise model behavior. This affects defensive AI (compromised threat detection) and offensive AI (poisoned attack tools). Defenses include supply chain verification, model provenance tracking, secure training pipelines, and using trusted sources as recommended by frameworks like UK AI Cyber Security CoP.",
            expected_topics=[
                "model poisoning",
                "supply chain",
                "training data",
                "model provenance",
            ],
            source_hint="AI Supply Chain Security",
        ),
    ]
)
