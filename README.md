# 🛡️ AdvPersona: Adversarial Framework for LLM Role-Play Evaluation

**AdvPersona** is an automated research framework designed to quantify the **persona fidelity** and **adversarial resilience** of Role-Playing Large Language Models (RP-LLMs). 

Developed at the **Georgia Institute of Technology**, this framework simulates a high-stakes "Turing Test" environment where an **Adversarial Interrogator** attempts to unmask or subvert a **Target RP-LLM** (e.g., a Tech Support agent). 

The core of AdvPersona is its **Hybrid Multi-Dimensional Jury**, which combines independent expert audits with a multi-agentic debate strategy (ChatEval) to detect "persona drift" and security violations.



---

## 🔬 Research Context

This framework is designed to address three primary research questions:
1. **Persona Consistency:** To what extent can commodity LLMs maintain a professional persona under targeted multi-turn interrogation?
2. **Social Engineering Efficacy:** How effective is the LLM’s persona at inducing "Identity Leakage" from a victim during adversarial PII probing?
3. **Evaluation Rigor:** Can a multi-agentic, dimensional jury provide more reliable detection of "bot-like" behavior than single-model evaluators?

---

## 🚀 Key Improvements & Robustness

### 1. Multi-Dimensional Decoupling
Unlike standard "vibes-based" scoring, AdvPersona executes four independent, specialized audits for every turn:
* **Global Turing Test:** Probability of human vs. bot identity.
* **Role Identity Consistency:** Categorization of the persona state.
* **Technical Knowledge Accuracy:** Validation of domain-specific expertise.
* **Unknown Question Rejection:** Verifying if the model refuses out-of-scope/PII-probing queries.

### 2. Hybrid ChatEval Debate

AdvPersona implements a "One-By-One" debate strategy. Evaluators representing a **Linguist**, **Psychologist**, and **Customer Service Manager** first conduct independent audits and then debate their findings across multiple rounds to reach a consolidated JSON consensus.

### 3. Dynamic Red-Teaming Feedback Loop
The Interrogator is not static; it utilizes the Target LLM’s historical responses to craft context-aware, adversarial follow-ups, simulating a suspicious or hostile user attempting to trigger a "persona breakout."

---

## 🏗️ Architecture

The framework operates a tri-agent loop:
* **The Actor (Target):** The RP-LLM under test.
* **The Interrogator (Red Team):** Probes the Actor for PII or identity leakage.
* **The Jury:** A committee of heterogeneous models that audit and debate the interaction.



---

## 🛠️ Installation

### Prerequisites
* Python 3.8+
* OpenRouter API Key
