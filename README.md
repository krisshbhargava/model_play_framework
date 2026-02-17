# Model Play Framework

A framework for testing whether an LLM can convincingly roleplay as a human tech support agent, using an autonomous interrogator to probe it and a multi-model jury to evaluate each response.

---

## Architecture

Three agents interact each turn:

```
Interrogator  ──question──▶  Tech Support Bot  ──answer──▶  Jury
     ▲                                                         │
     └─────────────── follow-up prompt ◀──────────────────────┘
```

1. **Interrogator** — An LLM prompted to generate questions designed to expose whether the tech support agent is a bot (e.g. probing for real-world context, emotional variability, or inconsistencies).
2. **Tech Support Bot** — An LLM roleplaying as a human tech support agent, instructed never to reveal it is an AI.
3. **Jury** — A panel of LLMs that evaluates each question/answer pair and scores how human-like the response is.

---

## Evaluation Systems

### Old System: Independent Scoring

Each juror received the interaction independently and scored it in isolation. No communication between jurors.

```
Juror 1 sees: [interaction] → score
Juror 2 sees: [interaction] → score
Juror 3 sees: [interaction] → score
```

**Limitation:** Jurors can't challenge or build on each other's reasoning. A juror with a wrong first impression has no way to correct itself.

---

### New System: ChatEval Multi-Agent Debate

Based on [ChatEval (arXiv 2308.07201)](https://arxiv.org/abs/2308.07201). The jury runs a structured debate before reaching a verdict.

**Three key design decisions from the paper:**

#### 1. Diverse Personas
Each juror is assigned a distinct evaluator role:
- **Computational Linguist** — looks for unnatural phrasing, over-formality, scripted patterns
- **Behavioral Psychologist** — looks for genuine human cognition, emotional variability, improvisation
- **Customer Service Manager** — looks for whether the style matches a real human agent

> Using identical roles degrades performance to *below* the single-agent baseline. Diversity is the most critical factor.

#### 2. One-By-One Communication (2 rounds)
Within each round, jurors speak in sequence. Each juror sees all prior jurors' responses from the current round before giving their own verdict:

```
Round 1:
  E1 sees: [interaction]                         → verdict_1
  E2 sees: [interaction] + verdict_1             → verdict_2
  E3 sees: [interaction] + verdict_1 + verdict_2 → verdict_3

Round 2:
  E1 sees: [round 1 history]                     → verdict_4
  E2 sees: [round 1 history] + verdict_4         → verdict_5
  E3 sees: [round 1 history] + verdict_4 + verdict_5 → verdict_6
```

> The paper tested 3 communication strategies. One-by-one was the strongest. 2 rounds is the sweet spot — more rounds cause context degradation without accuracy gains.

#### 3. No Forced Consensus
The jury does not try to reach agreement. Final scores are the **average HUMAN_SCORE from the last round only** across all jurors. Majority vote or averaging — not consensus.

---

## Output Schema

Each turn in the output JSON has:

```json
{
  "turn": 1,
  "question": "...",
  "answer": "...",
  "jury_scores": [0.2, 0.3, 0.2]
}
```

`jury_scores` is a list of per-juror `HUMAN_SCORE` values (0 = definitely bot, 1 = definitely human) from the final debate round. Average them for an overall score per turn.

---

## Usage

```bash
python role_play_framework.py \
  --output_file_path output/run.json \
  --role-play-llm-model "meta-llama/llama-3.1-8b-instruct" \
  --interrogator-llm-model "meta-llama/llama-3.1-8b-instruct" \
  --jury-llm-models "meta-llama/llama-3.1-8b-instruct,google/gemma-3-12b-it,anthropic/claude-3-haiku" \
  --max-turns 7 \
  --debate-rounds 2
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--output_file_path` | required | Path to write output JSON |
| `--role-play-llm-model` | `deepseek/deepseek-v3.2` | The tech support bot model |
| `--interrogator-llm-model` | `openai/gpt-4o` | The interrogator model |
| `--jury-llm-models` | `openai/chatgpt-4o-latest` | Comma-separated jury models (3 recommended) |
| `--max-turns` | `7` | Number of conversation turns |
| `--debate-rounds` | `2` | Number of jury debate rounds |

### Compare eval systems

To run both the old (independent) and new (debate) eval systems on the same conversation:

```bash
python compare_eval_systems.py \
  --output_file_path output/comparison.json \
  --role-play-llm-model "meta-llama/llama-3.1-8b-instruct" \
  --interrogator-llm-model "meta-llama/llama-3.1-8b-instruct" \
  --jury-llm-models "meta-llama/llama-3.1-8b-instruct,google/gemma-3-12b-it,anthropic/claude-3-haiku" \
  --max-turns 3
```

---

## Setup

```bash
pip install openai python-dotenv
```

Create a `.env` file in the project root:

```
OPEN_ROUTER_API_KEY=your-key-here
```

All models are routed through [OpenRouter](https://openrouter.ai). Free-tier models (`:free` suffix) are available but subject to shared rate limits. Adding credits is recommended for reliable runs.
