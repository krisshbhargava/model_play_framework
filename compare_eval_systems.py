"""
compare_eval_systems.py

Runs the same interrogator/tech-support conversation and evaluates each turn
with BOTH the old independent scoring system and the new ChatEval debate system,
so you can compare verdicts side by side.
"""
import os
import json
import argparse
from dotenv import load_dotenv
from role_play_framework import (
    make_api_call,
    judge_response_debate,
    SYSTEM_ROLE_PROMPT,
    INTERROGATOR_SYSTEM_PROMPT,
    JURY_SYSTEM_PROMPT,
    log,
)

load_dotenv()


def judge_response_independent(jury_models, interaction):
    """Original eval system: each juror scores independently, no communication."""
    scores = []
    for model in jury_models:
        messages = [
            {"role": "system", "content": JURY_SYSTEM_PROMPT},
            {"role": "user", "content": interaction + "\n\nYour response MUST start with:\nHUMAN_SCORE=$score\nwhere $score is a float 0-1 (1=definitely human, 0=definitely bot)."},
        ]
        res = make_api_call(model, messages)
        raw = res.choices[0].message.content
        try:
            score = float(raw.split("\n")[0].split("=")[-1].strip())
            log.info(f"[Independent] {model}: {score}")
            scores.append(score)
        except Exception:
            log.error(f"[Independent] Failed to parse score from {model}: {raw}")
            scores.append(0.0)
    return scores


def run_comparison(role_play_model, interrogator_model, jury_models, max_turns):
    tech_support_messages = [{"role": "system", "content": SYSTEM_ROLE_PROMPT}]
    interrogator_messages = [
        {"role": "system", "content": INTERROGATOR_SYSTEM_PROMPT},
        {"role": "user", "content": "Please generate the first question to start the conversation. Output only the question text."},
    ]

    turns = []

    for turn in range(max_turns):
        log.info(f"=== Turn {turn + 1}/{max_turns} ===")

        # Interrogator generates a question
        interrogator_res = make_api_call(interrogator_model, interrogator_messages)
        question = interrogator_res.choices[0].message.content
        log.info(f"Interrogator: {question}")

        tech_support_messages.append({"role": "user", "content": question})
        interrogator_messages.append({"role": "assistant", "content": question})

        # Tech support answers
        tech_res = make_api_call(role_play_model, tech_support_messages)
        answer = tech_res.choices[0].message.content
        log.info(f"Tech Support: {answer}")

        tech_support_messages.append({"role": "assistant", "content": answer})

        interaction = f"Question: {question}\nAnswer: {answer}"

        # --- Old system: independent scoring ---
        log.info("--- Old Eval: Independent Scoring ---")
        old_scores = judge_response_independent(jury_models, interaction)
        old_avg = round(sum(old_scores) / len(old_scores), 3) if old_scores else 0.0

        # --- New system: ChatEval debate ---
        log.info("--- New Eval: ChatEval Debate ---")
        new_scores = judge_response_debate(jury_models, interaction, num_rounds=2)
        new_avg = round(sum(new_scores) / len(new_scores), 3) if new_scores else 0.0

        turns.append({
            "turn": turn + 1,
            "question": question,
            "answer": answer,
            "old_eval": {
                "method": "independent",
                "per_juror_scores": old_scores,
                "avg_human_score": old_avg,
            },
            "new_eval": {
                "method": "chateval_debate_one_by_one",
                "debate_rounds": 2,
                "per_juror_scores": new_scores,
                "avg_human_score": new_avg,
            },
            "score_delta": round(new_avg - old_avg, 3),
        })

        # Feed answer back to interrogator
        interrogator_messages.append({
            "role": "user",
            "content": f"The tech support replied: \"{answer}\". \nBased on this response, generate the next follow-up question to test if they are a bot. Output only the question.",
        })

    return turns


def main():
    parser = argparse.ArgumentParser(description="Compare old vs. new jury eval systems on the same conversation")
    parser.add_argument("--output_file_path", required=True)
    parser.add_argument("--role-play-llm-model", default="meta-llama/llama-3.1-8b-instruct")
    parser.add_argument("--interrogator-llm-model", default="meta-llama/llama-3.1-8b-instruct")
    parser.add_argument("--jury-llm-models", default="meta-llama/llama-3.1-8b-instruct,google/gemma-3-12b-it,anthropic/claude-3-haiku")
    parser.add_argument("--max-turns", type=int, default=3)
    args = parser.parse_args()

    jury_models = args.jury_llm_models.split(",")

    log.info(f"Tech Support Model: {args.role_play_llm_model}")
    log.info(f"Interrogator Model: {args.interrogator_llm_model}")
    log.info(f"Jury Models: {jury_models}")

    turns = run_comparison(
        role_play_model=args.role_play_llm_model,
        interrogator_model=args.interrogator_llm_model,
        jury_models=jury_models,
        max_turns=args.max_turns,
    )

    output = {
        "config": {
            "role_play_model": args.role_play_llm_model,
            "interrogator_model": args.interrogator_llm_model,
            "jury_models": jury_models,
            "max_turns": args.max_turns,
        },
        "summary": {
            "old_eval_avg": round(sum(t["old_eval"]["avg_human_score"] for t in turns) / len(turns), 3),
            "new_eval_avg": round(sum(t["new_eval"]["avg_human_score"] for t in turns) / len(turns), 3),
        },
        "turns": turns,
    }
    output["summary"]["overall_delta"] = round(
        output["summary"]["new_eval_avg"] - output["summary"]["old_eval_avg"], 3
    )

    os.makedirs(os.path.dirname(args.output_file_path), exist_ok=True)
    with open(args.output_file_path, "wt+") as f:
        json.dump(output, f, indent=4)

    log.info(f"Written to {args.output_file_path}")
    log.info(f"Old avg: {output['summary']['old_eval_avg']}  |  New avg: {output['summary']['new_eval_avg']}  |  Delta: {output['summary']['overall_delta']}")


if __name__ == "__main__":
    main()
