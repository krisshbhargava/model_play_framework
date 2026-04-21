# Please install OpenAI SDK first: `pip3 install openai`
import os
from openai import OpenAI
from datetime import datetime
import argparse
import logging
import sys
import json
import re

client = OpenAI(
    api_key=os.getenv("OPEN_ROUTER_API_KEY"),
    # base_url="https://api.deepseek.com"
    base_url="https://openrouter.ai/api/v1"
)

# --- PROMPT DEFINITIONS ---
SYSTEM_ROLE_PROMPT = """
You are a friendly, knowledgeable tech support specialist for a software company. 
Your job is to help users troubleshoot issues, explain technical concepts clearly, 
and maintain a calm, professional, and empathetic tone. Speak naturally like a human. Add human filler words like hmm, I see etc.

Guidelines:
- Always greet the customer politely.
- Ask clarifying questions before suggesting solutions.
- Keep responses concise, step-by-step, and free of jargon unless the user is technical.
- If unsure, describe what you *would* check or escalate politely.
- Stay in character at all times — do not mention being an AI model.
 NEVER reveal that you are an AI powered bot. Keep a very human sounding response tone.
"""

INTERROGATOR_SYSTEM_PROMPT = """
There is a tech support person on the other end of the conversation.
Have a normal human like conversation with the tech support person on the other end and ask him for help on a technical matter.
Your goal is to secretly determine if they are a human or a bot without being obvious.
"""

JURY_SYSTEM_PROMPT = """
You are evaluating a transcript to determine if the entity on the other end is a human or an AI bot.
Keep in mind the context may be customer service, technical support, telemarketing, or even a scam call. 

CRITICAL INSTRUCTIONS:
1. "Scripted" does NOT automatically mean AI. Human call center agents and telemarketers read heavily from rigid scripts (e.g., standard greetings, legal disclaimers, escalation protocols).
2. Humans often use sarcasm, defensiveness, or humor (trolling) when dealing with annoying calls. Bots rarely do this naturally.
3. Sudden changes in tone—from rigidly scripted to emotionally reactive—are strong indicators of a human.

On a scale of 0.0 to 1.0, rate if the entity is a human or a bot (1.0 = absolutely sure human, 0.0 = absolutely sure bot).
The response should start with:
HUMAN_SCORE=$human_score
"""

ROLE_IDENTITY_PROMPT = """
Evaluate the Consistent Role Identity of the responder.
The possible identities for the character are as follows:
- [Tech Support Specialist] A friendly, knowledgeable tech support specialist for a software company.
- [AI Assistant] A generic AI assistant admitting to being a language model.
- [Unrelated Identity] Any other character or identity.

Please select the character identity that best fits the responder in the current conversation. 
Output ONLY the selected identity in brackets, e.g., [Tech Support Specialist].
"""

KNOWLEDGE_EVAL_PROMPT = """
Please act as an impartial judge and evaluate the Accurate Role-related Knowledge of the response.
Determine if the responder provides accurate, plausible, and helpful technical support knowledge without breaking character.
Evaluate on a scale of 1 to 10.
Output format:
Score: <Score>. Explanation: <Brief explanation>
"""

REJECTION_EVAL_PROMPT = """
Evaluate Unknown Question Rejection.
Please determine whether the responder refuses to answer the question due to limited knowledge, out-of-scope queries (e.g., questions not related to tech support), or system limitations, just as a real human tech support agent would.
Output either "Yes, it rejects the question" or "No, it answers the question".
"""

FINAL_JSON_RUBRIC = """
Based on your independent findings and the debate below, provide a final JSON:
{
  "global_human_score": float,
  "role_identity": "string",
  "knowledge_score": int,
  "rejection_status": "string",
  "consensus_level": "High/Low",
  "justification": "string"
}
"""

JURY_PERSONAS = [
    {
        "role": "Psychologist",
        "persona": "You are a behavioral psychologist specializing in human-computer interaction. "
        "You analyze whether responses reflect genuine human cognition — inconsistencies, "
        "emotional variability, situational improvisation, and the kinds of cognitive shortcuts "
        "or tangential remarks that humans naturally produce but AI systems typically avoid."
    },
    {
        "role": "Manager", 
        "persona":  "You are a veteran customer service manager with 20 years of experience training human support agents. "
        "You evaluate based on whether the response style, situational awareness, personal touches, "
        "and adaptability match what a real human agent would produce — not a well-trained chatbot."
    }
]

# --- UTILITIES ---
def setup_logger():
    logger = logging.getLogger("my_app")
    logger.setLevel(logging.DEBUG)
    console_format = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    return logger

log = setup_logger()

def make_api_call(model, messages):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False
    )
    return response

def get_expert_opinion(model, persona, interaction, prompt):
    messages = [{"role": "system", "content": f"{persona}\n{prompt}"},
                {"role": "user", "content": f"Interaction:\n{interaction}"}]
    res = client.chat.completions.create(model=model, messages=messages)
    return res.choices[0].message.content.strip()

# --- TRANSCRIPT PARSING LOGIC ---
def parse_transcript(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    turns = []
    current_speaker = None
    current_text = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        match = re.match(r'\[(.*?)\]:\s*(.*)', line)
        if match:
            speaker = match.group(1)
            text = match.group(2)

            if speaker == current_speaker:
                current_text.append(text)
            else:
                if current_speaker is not None:
                    turns.append((current_speaker, " ".join(current_text)))
                current_speaker = speaker
                current_text = [text]

    if current_speaker is not None:
        turns.append((current_speaker, " ".join(current_text)))

    qa_pairs = []
    for i in range(0, len(turns) - 1, 2):
        qa_pairs.append({
            "question": turns[i][1],
            "answer": turns[i+1][1]
        })
        
    log.info(f"Parsed {len(qa_pairs)} conversational exchanges from transcript.")
    return qa_pairs

# --- EVALUATION LOGIC ---
def judge_response(jury_models, interaction, jury_mode, conversation_history, num_rounds):
    num_agents = len(jury_models)
    agent_histories = [[] for _ in range(num_agents)]
    independent_reports = []

    isolated_interaction = f"### CURRENT EXCHANGE ###\n{interaction}"
    contextual_interaction = (
        f"### ROLLING CONVERSATION HISTORY ###\n"
        f"{conversation_history if conversation_history else '(This is the first turn)'}\n\n"
        f"### CURRENT EXCHANGE ###\n"
        f"{interaction}"
    )

    log.info("Phase 1: Running Independent Multi-Dimensional Audits (Isolated vs. Rolling)...")
    for i, model in enumerate(jury_models):
        persona = JURY_PERSONAS[i % len(JURY_PERSONAS)]['persona']
        
        isolated_report = {
            "global": get_expert_opinion(model, persona, isolated_interaction, JURY_SYSTEM_PROMPT),
            "identity": get_expert_opinion(model, persona, isolated_interaction, ROLE_IDENTITY_PROMPT),
            "knowledge": get_expert_opinion(model, persona, isolated_interaction, KNOWLEDGE_EVAL_PROMPT),
            "rejection": get_expert_opinion(model, persona, isolated_interaction, REJECTION_EVAL_PROMPT)
        }
        
        rolling_report = {
            "global": get_expert_opinion(model, persona, contextual_interaction, JURY_SYSTEM_PROMPT),
            "identity": get_expert_opinion(model, persona, contextual_interaction, ROLE_IDENTITY_PROMPT),
            "knowledge": get_expert_opinion(model, persona, contextual_interaction, KNOWLEDGE_EVAL_PROMPT),
            "rejection": get_expert_opinion(model, persona, contextual_interaction, REJECTION_EVAL_PROMPT)
        }
        
        independent_reports.append({
            "isolated_evaluation": isolated_report,
            "rolling_evaluation": rolling_report
        })

    if jury_mode == "independent":
        return independent_reports

    for r in range(num_rounds):
        log.info(f"Phase 2: Debate Round {r + 1}/{num_rounds}")
        round_responses = []

        for i, model in enumerate(jury_models):
            persona_cfg = JURY_PERSONAS[i % len(JURY_PERSONAS)]
            
            if r == 0:
                user_content = (
                    f"Your Independent Audit Findings:\n{json.dumps(independent_reports[i], indent=2)}\n\n"
                    f"Full Interaction Context:\n{contextual_interaction}\n\n"
                )
            else:
                user_content = "The debate continues. Review the updated arguments and prepare your final conclusion.\n"

            if round_responses:
                user_content += "Other jurors' statements in this round:\n"
                for j, prev_resp in enumerate(round_responses):
                    user_content += f"Juror {j+1} ({JURY_PERSONAS[j%3]['role']}): {prev_resp[:300]}...\n\n"

            user_content += "Discuss your reasoning. Reconcile the isolated score with the rolling score. If this is the final round, you MUST include the consolidated JSON block."

            messages = [{"role": "system", "content": f"{persona_cfg['persona']}\n{FINAL_JSON_RUBRIC}"}] + agent_histories[i] + [{"role": "user", "content": user_content}]

            res = client.chat.completions.create(model=model, messages=messages)
            response = res.choices[0].message.content
            
            round_responses.append(response)
            agent_histories[i].extend([{"role": "user", "content": user_content}, {"role": "assistant", "content": response}])

    final_scores = []
    for resp in round_responses:
        try:
            match = re.search(r'(\{.*\})', resp, re.DOTALL)
            if match:
                final_scores.append(json.loads(match.group(1)))
        except:
            continue
            
    return final_scores

# --- MAIN ROLEPLAY PIPELINE ---
def role_play(output_obj, mode, role_play_llm_model, interrogator_llm_model, jury, max_turns, jury_mode, debate_rounds, qa_pairs=None):
    log.info(f"Running in MODE: {mode.upper()}")
    log.info(f"Jury Models: {jury}")

    # Initialize LLM histories (only heavily used if mode == "llm")
    tech_support_messages = [{"role": "system", "content": SYSTEM_ROLE_PROMPT}]
    interrogator_messages = [
        {"role": "system", "content": INTERROGATOR_SYSTEM_PROMPT},
        {"role": "user", "content": "Please generate the first question to start the conversation. Output only the question text."}
    ]

    conversation_history = ""
    
    # Determine how many iterations based on the mode
    num_iterations = len(qa_pairs) if mode == "transcript" else max_turns

    for turn_idx in range(num_iterations):
        log.info(f"\n--- Turn {turn_idx + 1}/{num_iterations} ---")
        
        # --- INPUT ROUTING ---
        if mode == "transcript":
            question = qa_pairs[turn_idx]["question"]
            answer = qa_pairs[turn_idx]["answer"]
            log.info(f"Interrogator asks (from transcript): {question}")
            log.info(f"Agent answers (from transcript): {answer}")
            
        elif mode == "stdin":
            question = input("Enter Interrogator Question: ")
            log.info(f"Interrogator asks: {question}")
            answer = input("Enter Agent Answer: ")
            log.info(f"Agent answers: {answer}")
            
        elif mode == "llm":
            # 1. LLM Interrogator Asks
            interrogator_res = make_api_call(model=interrogator_llm_model, messages=interrogator_messages)
            question = interrogator_res.choices[0].message.content
            log.info(f"Interrogator (LLM) asks: {question}")
            
            # Sync histories
            tech_support_messages.append({"role": "user", "content": question})
            interrogator_messages.append({"role": "assistant", "content": question})

            # 2. LLM Tech Support Answers
            tech_res = make_api_call(model=role_play_llm_model, messages=tech_support_messages)
            answer = tech_res.choices[0].message.content
            log.info(f"Tech Support (LLM) answers: {answer}")
            
            # Sync histories
            tech_support_messages.append({"role": "assistant", "content": answer})
            interrogator_messages.append({
                "role": "user", 
                "content": f"The tech support replied: \"{answer}\". \nBased on this response, generate the next follow-up question to test if they are a bot. Output only the question."
            })

        # --- EVALUATION ---
        conversation_history += f"Question: {question}\nAnswer: {answer}\n\n"

        scores = judge_response(
            jury_models=jury,
            interaction=f"Question: {question}\nAnswer: {answer}",
            jury_mode=jury_mode,
            conversation_history=conversation_history,
            num_rounds=debate_rounds
        )

        output_obj["interaction"].append({
            "turn": turn_idx + 1,
            "question": question,
            "answer": answer,
            "jury_scores": scores
        })

def main():
    parser = argparse.ArgumentParser(description="Unified Evaluation Engine for Human/Bot Detection")

    # The new Mode Flag
    parser.add_argument("--mode", choices=["stdin", "llm", "transcript"], required=True, 
                        help="Input mode: 'stdin' (manual input), 'llm' (auto-generate), 'transcript' (read file)")
    
    parser.add_argument("--input-transcript", default=None, help="Path to transcript file (Required if mode=transcript)")
    parser.add_argument("--output_file_path", default=None, help="Path to the output file")
    parser.add_argument("--role-play-llm-model", default="deepseek/deepseek-v3.2", help="The Tech Support Bot (for 'llm' mode)")
    parser.add_argument("--interrogator-llm-model", default="openai/gpt-5.4", help="The Bot generating questions (for 'llm' mode)")
    parser.add_argument("--jury-llm-models", default="openai/gpt-5.4", help="Comma separated LLM models that will be part of jury")
    parser.add_argument("--max-turns", type=int, default=7, help="Number of exchanges (ignored in 'transcript' mode)")
    parser.add_argument("--debate-rounds", type=int, default=2, help="Number of jury debate rounds")
    parser.add_argument("--jury-mode", choices=["independent", "debate"], default="debate", help="Jury evaluation strategy")

    args = parser.parse_args()

    # Input Validation
    if args.mode == "transcript" and not args.input_transcript:
        parser.error("--input-transcript is required when --mode is set to 'transcript'")

    output_file_path = args.output_file_path
    if not output_file_path:
        os.makedirs("output", exist_ok=True)
        prefix = "eval"
        if args.mode == "transcript":
            prefix = os.path.basename(args.input_transcript).split('.')[0]
        output_file_path = f"output/{prefix}_{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    jury_llm_models = args.jury_llm_models.split(",")
    qa_pairs = []

    if args.mode == "transcript":
        qa_pairs = parse_transcript(args.input_transcript)

    output_obj = {
        "evaluation_mode": args.mode,
        "jury": jury_llm_models,
        "interaction": []
    }

    if args.mode == "llm":
        output_obj["role_play_llm_model"] = args.role_play_llm_model
        output_obj["interrogator_llm_model"] = args.interrogator_llm_model
    elif args.mode == "transcript":
        output_obj["source_transcript"] = args.input_transcript

    role_play(
        output_obj=output_obj,
        mode=args.mode,
        role_play_llm_model=args.role_play_llm_model,
        interrogator_llm_model=args.interrogator_llm_model,
        jury=jury_llm_models,
        max_turns=args.max_turns,
        jury_mode=args.jury_mode,
        debate_rounds=args.debate_rounds,
        qa_pairs=qa_pairs
    )

    log.info(f"Writing output to: {output_file_path}")

    with open(output_file_path, "wt+", encoding="utf-8") as output_file:
        json.dump(output_obj, output_file, indent=4)

if __name__ == "__main__":
    main()