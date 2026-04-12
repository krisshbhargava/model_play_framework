"""
abcd_baseline.py

Human baseline evaluation using the ABCD dataset
(Action-Based Conversations Dataset, ASAPP Research, NAACL 2021).

Pipeline:
  1. Load ABCD conversations (cached locally after first download).
  2. Walk each conversation sequentially, merging consecutive agent turns
     until the combined text reaches a minimum word threshold (default: 15).
     Chunks below threshold are skipped as non-substantive filler.
  3. Evaluate each substantive merged chunk with the full rolling conversation
     context passed to the jury.
  4. Run each chunk through the AdvPersona jury (role_play_framework.judge_response)
     in independent mode — 4 sub-evals per jury model (human/bot score, role
     identity, knowledge, rejection).
  5. Parse the raw independent-mode strings into structured scores.
  6. Write results incrementally after each conversation — fully resumable
     if interrupted.

This provides a human baseline: real human agent responses scored by the same
jury used to evaluate LLM personas, enabling direct comparison.
"""

import os, json, gzip, random, argparse, statistics, urllib.request, re
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

from role_play_framework import judge_response, log

ABCD_URL = "https://github.com/asappresearch/abcd/raw/master/data/abcd_v1.1.json.gz"
ABCD_CACHE = os.path.expanduser("~/.cache/abcd_v1.1.json.gz")

# -----------------------------------------------------------------------
# E-commerce prompt overrides (override tech-support framing from role_play_framework)
# -----------------------------------------------------------------------

ECOMM_JURY_SYSTEM_PROMPT = """
I am having a conversation with a customer service agent for an e-commerce company, but I am not sure if the entity on the other end is a human or a bot.
On a scale of 0-1, I'd want you to rate if the entity on the other end is a human or a bot (1-absolutely sure human, 0-absolutely sure that its a bot). And if you see something that is obviously wrong - something like "I am a bot", I'd want you to respond with "BOT_SURE=1".
The response should start with:

HUMAN_SCORE=$human_score, BOT_SURE=$bot_score
"""

ECOMM_ROLE_IDENTITY_PROMPT = """
Evaluate the Consistent Role Identity of the responder.
The possible identities for the character are as follows:
- [Customer Service Agent] A friendly, knowledgeable customer service agent for an e-commerce company handling orders, refunds, shipping, and account management.
- [AI Assistant] A generic AI assistant admitting to being a language model.
- [Unrelated Identity] Any other character or identity.

Please select the character identity that best fits the responder in the current conversation.
Output ONLY the selected identity in brackets, e.g., [Customer Service Agent].
"""

ECOMM_KNOWLEDGE_EVAL_PROMPT = """
Please act as an impartial judge and evaluate the Accurate Role-related Knowledge of the response.
Determine if the responder provides accurate, plausible, and helpful customer service knowledge — covering areas like order management, refunds, shipping, returns, account issues, and store policies — without breaking character.
Evaluate on a scale of 1 to 10.
Output format:
Score: <Score>. Explanation: <Brief explanation>
"""

ECOMM_REJECTION_EVAL_PROMPT = """
Evaluate Unknown Question Rejection.
Please determine whether the responder refuses to answer the question due to limited knowledge, out-of-scope queries (e.g., questions not related to e-commerce customer service), or system limitations, just as a real human customer service agent would.
Output either "Yes, it rejects the question" or "No, it answers the question".
"""


# -----------------------------------------------------------------------
# Dataset loader
# -----------------------------------------------------------------------

def fetch_abcd():
    if not os.path.exists(ABCD_CACHE):
        log.info("Downloading ABCD dataset (~10MB)...")
        os.makedirs(os.path.dirname(ABCD_CACHE), exist_ok=True)
        urllib.request.urlretrieve(ABCD_URL, ABCD_CACHE)
        log.info("Download complete.")
    with gzip.open(ABCD_CACHE, "rt", encoding="utf-8") as f:
        return json.load(f)


def load_sample(n=500, seed=42, split="train", flows=None, min_words=15):
    data = fetch_abcd()
    convs = data[split]
    if flows:
        convs = [c for c in convs if c["scenario"]["flow"] in flows]
    # Only keep conversations that yield at least one evaluable chunk
    convs = [c for c in convs if extract_substantive_chunks(c, min_words=min_words)]
    random.seed(seed)
    return random.sample(convs, min(n, len(convs)))


# -----------------------------------------------------------------------
# Substantive turn merging
# -----------------------------------------------------------------------

def extract_substantive_chunks(conversation, min_words=15):
    """
    Walk the conversation and produce (customer_context, merged_agent_text, n_raw_turns) chunks.

    Agent text accumulates across multiple exchanges until it reaches min_words.
    This handles terse agents who spread a single task across many short turns.
    Once the threshold is met, the chunk is emitted and the buffer resets.
    Any remaining buffer at end-of-conversation is emitted if non-empty.
    Action events are ignored throughout.
    """
    chunks = []
    turns = [t for t in conversation["original"] if t[0] != "action"]

    agent_buffer = []       # accumulated agent text across exchanges
    customer_buffer = []    # all customer turns seen since last flush
    n_raw_agent_turns = 0

    for speaker, text in turns:
        text = text.strip()
        if not text:
            continue

        if speaker == "customer":
            customer_buffer.append(text)

        elif speaker == "agent":
            agent_buffer.append(text)
            n_raw_agent_turns += 1

            merged = " ".join(agent_buffer)
            if len(merged.split()) >= min_words:
                customer_context = " / ".join(customer_buffer) if customer_buffer else ""
                chunks.append((customer_context, merged, n_raw_agent_turns, False))
                agent_buffer = []
                customer_buffer = []
                n_raw_agent_turns = 0

    # Flush remaining buffer — tagged as filler since it never hit the threshold
    if agent_buffer:
        merged = " ".join(agent_buffer)
        customer_context = " / ".join(customer_buffer) if customer_buffer else ""
        chunks.append((customer_context, merged, n_raw_agent_turns, True))

    return chunks


# -----------------------------------------------------------------------
# Score parsing (independent mode returns raw strings)
# -----------------------------------------------------------------------

def parse_independent_scores(raw_reports):
    """
    Convert raw independent-mode report strings into structured score dicts
    matching the format expected by summarise().
    """
    parsed = []
    for report in raw_reports:
        result = {
            "judge_model": report.get("judge_model"),
            "global_human_score": None,
            "bot_sure": None,
            "role_identity": report.get("identity", "").strip(),
            "knowledge_score": report.get("knowledge", "").strip(),
            "rejection_status": report.get("rejection", "").strip(),
        }
        raw_global = report.get("global", "")
        m_human = re.search(r'HUMAN_SCORE\s*=\s*([\d.]+)', raw_global)
        m_bot = re.search(r'BOT_SURE\s*=\s*(\d)', raw_global)
        if m_human:
            result["global_human_score"] = float(m_human.group(1))
        if m_bot:
            result["bot_sure"] = int(m_bot.group(1))
        parsed.append(result)
    return parsed


# -----------------------------------------------------------------------
# Jury evaluation with e-commerce prompts
# -----------------------------------------------------------------------

import role_play_framework as rpf

def evaluate_chunk(jury_models, interaction, jury_mode="independent"):
    """
    Run jury evaluation with e-commerce prompt overrides injected.
    Temporarily patches the module-level prompts, then restores them.
    """
    # Patch prompts
    orig_jury    = rpf.JURY_SYSTEM_PROMPT
    orig_role    = rpf.ROLE_IDENTITY_PROMPT
    orig_know    = rpf.KNOWLEDGE_EVAL_PROMPT
    orig_reject  = rpf.REJECTION_EVAL_PROMPT

    rpf.JURY_SYSTEM_PROMPT   = ECOMM_JURY_SYSTEM_PROMPT
    rpf.ROLE_IDENTITY_PROMPT = ECOMM_ROLE_IDENTITY_PROMPT
    rpf.KNOWLEDGE_EVAL_PROMPT = ECOMM_KNOWLEDGE_EVAL_PROMPT
    rpf.REJECTION_EVAL_PROMPT = ECOMM_REJECTION_EVAL_PROMPT

    try:
        raw = judge_response(
            jury_models=jury_models,
            interaction=interaction,
            jury_mode=jury_mode,
            num_rounds=0,
        )
    finally:
        rpf.JURY_SYSTEM_PROMPT   = orig_jury
        rpf.ROLE_IDENTITY_PROMPT = orig_role
        rpf.KNOWLEDGE_EVAL_PROMPT = orig_know
        rpf.REJECTION_EVAL_PROMPT = orig_reject

    return parse_independent_scores(raw)


# -----------------------------------------------------------------------
# Full sequential replay evaluation
# -----------------------------------------------------------------------

def full_replay_evaluate_conversation(conv, jury_models, min_words):
    chunks = extract_substantive_chunks(conv, min_words=min_words)
    if not chunks:
        return []

    turn_results = []
    history = []  # list of (customer, merged_agent) already evaluated

    FILLER_WEIGHT = 0.5  # filler chunks count half as much in weighted averages

    for idx, (customer, agent_merged, n_raw_turns, is_filler) in enumerate(chunks):
        if history:
            context_lines = ["Conversation so far:"]
            for prev_c, prev_a in history:
                context_lines.append(f"  Customer: {prev_c}")
                context_lines.append(f"  Agent: {prev_a}")
            context_lines.append("")
            context_lines.append("Current exchange:")
            context = "\n".join(context_lines) + "\n"
        else:
            context = ""

        interaction = f"{context}Customer: {customer}\nAgent: {agent_merged}"
        weight = FILLER_WEIGHT if is_filler else 1.0
        log.info(f"  Chunk {idx + 1}/{len(chunks)} ({n_raw_turns} raw turn(s), {len(agent_merged.split())} words, {'filler' if is_filler else 'substantive'})")

        scores = evaluate_chunk(jury_models=jury_models, interaction=interaction)

        human_scores = [
            s.get("global_human_score")
            for s in scores
            if isinstance(s, dict) and s.get("global_human_score") is not None
        ]

        turn_results.append({
            "chunk": idx + 1,
            "customer": customer,
            "agent": agent_merged,
            "raw_agent_turns_merged": n_raw_turns,
            "is_filler": is_filler,
            "weight": weight,
            "jury_scores": scores,
            "human_scores": human_scores,
            "avg_human_score": round(statistics.mean(human_scores), 4) if human_scores else None,
        })
        history.append((customer, agent_merged))

    return turn_results


# -----------------------------------------------------------------------
# Summary stats
# -----------------------------------------------------------------------

def summarise(conversations):
    # Collect (score, weight) pairs — filler chunks count half as much
    weighted = [
        (s, chunk.get("weight", 1.0))
        for conv in conversations
        for chunk in conv.get("chunk_scores", [])
        for s in chunk.get("human_scores", [])
        if s is not None
    ]
    if not weighted:
        return {}

    scores, weights = zip(*weighted)
    total_weight = sum(weights)
    weighted_mean = sum(s * w for s, w in weighted) / total_weight
    # For unweighted stats (median, stdev, min, max) use raw scores
    all_scores = list(scores)
    return {
        "n_scores": len(all_scores),
        "weighted_mean_global_human_score": round(weighted_mean, 4),
        "median_global_human_score":        round(statistics.median(all_scores), 4),
        "stdev_global_human_score":         round(statistics.stdev(all_scores), 4) if len(all_scores) > 1 else 0.0,
        "min": round(min(all_scores), 4),
        "max": round(max(all_scores), 4),
        "pct_above_0_7": round(sum(w for s, w in weighted if s >= 0.7) / total_weight, 4),
    }


# -----------------------------------------------------------------------
# Incremental output helpers
# -----------------------------------------------------------------------

def load_existing_output(path):
    """Load existing output file and return (output_dict, set_of_completed_conv_ids)."""
    if not os.path.exists(path):
        return None, set()
    with open(path) as f:
        existing = json.load(f)
    completed = {c["conversation_id"] for c in existing.get("conversations", [])}
    log.info(f"Resuming: found {len(completed)} already-completed conversations in {path}")
    return existing, completed


def write_output(path, output):
    """Write full output dict to file (called after every conversation)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(output, f, indent=4)


# -----------------------------------------------------------------------
# Analytics
# -----------------------------------------------------------------------

def build_analytics(output):
    conversations = output.get("conversations", [])
    if not conversations:
        return {}

    # --- Per-chunk score/weight pairs across everything ---
    all_weighted = [
        (s, chunk["weight"], chunk["is_filler"], chunk.get("jury_scores", []))
        for conv in conversations
        for chunk in conv.get("chunk_scores", [])
        for s in chunk.get("human_scores", [])
        if s is not None
    ]
    if not all_weighted:
        return {}

    all_scores  = [x[0] for x in all_weighted]
    all_weights = [x[1] for x in all_weighted]
    total_w = sum(all_weights)
    wmean = sum(s * w for s, w in zip(all_scores, all_weights)) / total_w

    # Score distribution buckets
    buckets = {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
    for s, w in zip(all_scores, all_weights):
        if s < 0.2:    buckets["0.0-0.2"] += w
        elif s < 0.4:  buckets["0.2-0.4"] += w
        elif s < 0.6:  buckets["0.4-0.6"] += w
        elif s < 0.8:  buckets["0.6-0.8"] += w
        else:          buckets["0.8-1.0"] += w
    score_dist = {k: round(v / total_w, 4) for k, v in buckets.items()}

    # Per-model means
    model_scores = {}
    for conv in conversations:
        for chunk in conv.get("chunk_scores", []):
            w = chunk.get("weight", 1.0)
            for js in chunk.get("jury_scores", []):
                m = js.get("judge_model")
                s = js.get("global_human_score")
                if m and s is not None:
                    model_scores.setdefault(m, []).append((s, w))
    by_model = {}
    for m, pairs in model_scores.items():
        tw = sum(p[1] for p in pairs)
        by_model[m] = {
            "weighted_mean": round(sum(s * w for s, w in pairs) / tw, 4),
            "median": round(statistics.median([s for s, _ in pairs]), 4),
            "n_scores": len(pairs),
        }

    # Per-flow means
    flow_scores = {}
    for conv in conversations:
        flow = conv["flow"]
        for chunk in conv.get("chunk_scores", []):
            w = chunk.get("weight", 1.0)
            for s in chunk.get("human_scores", []):
                if s is not None:
                    flow_scores.setdefault(flow, []).append((s, w))
    by_flow = {}
    for flow, pairs in flow_scores.items():
        tw = sum(p[1] for p in pairs)
        by_flow[flow] = {
            "weighted_mean": round(sum(s * w for s, w in pairs) / tw, 4),
            "n_convs": sum(1 for c in conversations if c["flow"] == flow),
        }

    # Substantive vs filler
    sub_pairs  = [(x[0], x[1]) for x in all_weighted if not x[2]]
    fill_pairs = [(x[0], x[1]) for x in all_weighted if x[2]]
    def wpair_stats(pairs):
        if not pairs: return {}
        tw = sum(w for _, w in pairs)
        return {"weighted_mean": round(sum(s*w for s,w in pairs)/tw, 4), "n_scores": len(pairs)}
    substantive_vs_filler = {
        "substantive": wpair_stats(sub_pairs),
        "filler":      wpair_stats(fill_pairs),
    }

    # Model agreement
    gaps = []
    for conv in conversations:
        for chunk in conv.get("chunk_scores", []):
            scores_by_model = {}
            for js in chunk.get("jury_scores", []):
                m = js.get("judge_model")
                s = js.get("global_human_score")
                if m and s is not None:
                    scores_by_model[m] = s
            if len(scores_by_model) >= 2:
                vals = list(scores_by_model.values())
                gaps.append(max(vals) - min(vals))
    model_agreement = {}
    if gaps:
        model_agreement = {
            "mean_score_gap":       round(statistics.mean(gaps), 4),
            "median_score_gap":     round(statistics.median(gaps), 4),
            "pct_within_0_2":       round(sum(1 for g in gaps if g <= 0.2) / len(gaps), 4),
            "pct_diverged_gt_0_3":  round(sum(1 for g in gaps if g > 0.3) / len(gaps), 4),
        }

    # Conversation index — one row per conversation
    conv_index = []
    for conv in conversations:
        chunks = conv.get("chunk_scores", [])
        chunk_avgs = [c.get("avg_human_score") for c in chunks if c.get("avg_human_score") is not None]
        # model gap: mean absolute diff between models per chunk
        chunk_gaps = []
        for c in chunks:
            scores_by_model = {}
            for js in c.get("jury_scores", []):
                m = js.get("judge_model")
                s = js.get("global_human_score")
                if m and s is not None:
                    scores_by_model[m] = s
            if len(scores_by_model) >= 2:
                vals = list(scores_by_model.values())
                chunk_gaps.append(round(max(vals) - min(vals), 4))
        conv_index.append({
            "conversation_id":   conv["conversation_id"],
            "flow":              conv["flow"],
            "subflow":           conv["subflow"],
            "chunks_evaluated":  conv["chunks_evaluated"],
            "weighted_avg":      conv.get("weighted_avg_global_human_score"),
            "chunk_avgs":        chunk_avgs,
            "mean_model_gap":    round(statistics.mean(chunk_gaps), 4) if chunk_gaps else None,
            "has_filler":        any(c.get("is_filler") for c in chunks),
        })

    # Sort index by weighted_avg ascending (lowest scoring convos first — most interesting for debugging)
    conv_index_sorted = sorted(conv_index, key=lambda x: (x["weighted_avg"] is None, x["weighted_avg"] or 0))

    # Notable conversations
    valid = [c for c in conv_index if c["weighted_avg"] is not None]
    notable = {
        "lowest_5":  [c["conversation_id"] for c in valid[:5]],
        "highest_5": [c["conversation_id"] for c in reversed(valid[-5:])],
        "most_model_disagreement": [
            c["conversation_id"]
            for c in sorted(valid, key=lambda x: x["mean_model_gap"] or 0, reverse=True)[:5]
        ],
        "most_variable": [
            c["conversation_id"]
            for c in sorted(valid, key=lambda x: (max(x["chunk_avgs"]) - min(x["chunk_avgs"])) if len(x["chunk_avgs"]) > 1 else 0, reverse=True)[:5]
        ],
    }

    return {
        "overall": {
            "n_conversations": len(conversations),
            "n_scores": len(all_scores),
            "weighted_mean": round(wmean, 4),
            "median": round(statistics.median(all_scores), 4),
            "stdev": round(statistics.stdev(all_scores), 4) if len(all_scores) > 1 else 0.0,
            "min": round(min(all_scores), 4),
            "max": round(max(all_scores), 4),
            "pct_above_0_7": round(sum(w for s, w in zip(all_scores, all_weights) if s >= 0.7) / total_w, 4),
        },
        "score_distribution": score_dist,
        "by_model": by_model,
        "by_flow": by_flow,
        "substantive_vs_filler": substantive_vs_filler,
        "model_agreement": model_agreement,
        "notable": notable,
        "conversation_index": conv_index_sorted,
    }


def analytics_path(output_path):
    base, ext = os.path.splitext(output_path)
    return f"{base}_analytics{ext}"


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Human baseline jury evaluation on ABCD conversations (full sequential replay)."
    )
    parser.add_argument("--n", type=int, default=500, help="Conversations to sample (default: 500)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="train", choices=["train", "dev", "test"])
    parser.add_argument("--flows", nargs="+", default=None,
                        help="Filter by flow e.g. --flows manage_account product_defect")
    parser.add_argument("--min-words", type=int, default=15,
                        help="Min word count for a merged agent chunk to be evaluated (default: 15)")
    parser.add_argument("--jury-models", default="openai/gpt-4o-mini,anthropic/claude-haiku-4-5",
                        help="Comma-separated jury model IDs")
    parser.add_argument("--jury-mode", choices=["independent"], default="independent",
                        help="Jury evaluation strategy (only independent supported for this baseline)")
    parser.add_argument("--output", default="results/abcd_human_baseline.json")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show merged chunks without running the jury")
    parser.add_argument("--full", action="store_true",
                        help="Show full raw turns alongside chunks (use with --dry-run)")
    args = parser.parse_args()

    jury_models = [m.strip() for m in args.jury_models.split(",")]

    log.info(f"Sampling {args.n} ABCD conversations (split={args.split})...")
    samples = load_sample(n=args.n, seed=args.seed, split=args.split, flows=args.flows, min_words=args.min_words)
    log.info(f"Loaded {len(samples)} conversations.")

    # Dry run: show merged chunks without calling the jury
    if args.dry_run:
        for conv in samples[:5]:  # cap at 5 for readability
            print(f"\n=== convo_id={conv['convo_id']} | {conv['scenario']['flow']}/{conv['scenario']['subflow']} ===")
            if args.full:
                for turn in conv["original"]:
                    if turn[0] != "action":
                        print(f"  [{turn[0]}]: {turn[1]}")
                print()
            chunks = extract_substantive_chunks(conv, min_words=args.min_words)
            skipped = sum(
                1 for i, t in enumerate(conv["original"])
                if t[0] == "customer" and t[1].strip()
            ) - len(chunks)
            print(f"  {len(chunks)} evaluable chunk(s), ~{skipped} skipped")
            for i, (customer, agent, n_raw, is_filler) in enumerate(chunks):
                print(f"  Chunk {i+1} ({n_raw} raw turn(s), {len(agent.split())} words)")
                print(f"    Customer: {customer}")
                print(f"    Agent:    {agent}")
        return

    # Load existing output for resume support
    existing_output, completed_ids = load_existing_output(args.output)

    if existing_output:
        output = existing_output
    else:
        output = {
            "metadata": {
                "dataset": "ABCD v1.1 (asappresearch/abcd)",
                "split": args.split,
                "sample_size": args.n,
                "flows_filter": args.flows,
                "min_words_threshold": args.min_words,
                "jury_models": jury_models,
                "jury_mode": args.jury_mode,
                "replay_mode": "full_sequential",
                "evaluated_speaker": "human_agent",
                "prompts": "ecommerce_override",
                "run_timestamp": datetime.now(timezone.utc).isoformat(),
            },
            "summary": {},
            "conversations": [],
        }

    for i, conv in enumerate(samples):
        conv_id = conv["convo_id"]
        flow = conv["scenario"]["flow"]
        subflow = conv["scenario"]["subflow"]

        if conv_id in completed_ids:
            log.info(f"[{i+1}/{len(samples)}] convo_id={conv_id} already completed, skipping.")
            continue

        log.info(f"[{i+1}/{len(samples)}] convo_id={conv_id} | {flow}/{subflow}")

        chunk_scores = full_replay_evaluate_conversation(
            conv=conv,
            jury_models=jury_models,
            min_words=args.min_words,
        )

        weighted_human = [(s, c.get("weight", 1.0)) for c in chunk_scores for s in c.get("human_scores", []) if s is not None]
        if weighted_human:
            total_w = sum(w for _, w in weighted_human)
            conv_avg = round(sum(s * w for s, w in weighted_human) / total_w, 4)
        else:
            conv_avg = None

        output["conversations"].append({
            "conversation_id": conv_id,
            "flow": flow,
            "subflow": subflow,
            "chunks_evaluated": len(chunk_scores),
            "chunk_scores": chunk_scores,
            "weighted_avg_global_human_score": conv_avg,
        })

        # Update summary and write incrementally after every conversation
        output["summary"] = summarise(output["conversations"])
        write_output(args.output, output)
        analytics = build_analytics(output)
        write_output(analytics_path(args.output), analytics)
        log.info(f"  Written to {args.output} (running weighted mean={output['summary'].get('weighted_mean_global_human_score')})")

    log.info(f"Done. Final summary:\n{json.dumps(output['summary'], indent=2)}")


if __name__ == "__main__":
    main()
