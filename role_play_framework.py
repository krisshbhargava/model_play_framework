# Please install OpenAI SDK first: `pip3 install openai`
import os
from openai import OpenAI
import argparse
import logging
import sys
import json
from dotenv import load_dotenv
load_dotenv()


client = OpenAI(
    api_key=os.getenv("OPEN_ROUTER_API_KEY"),
    # base_url="https://api.deepseek.com"
    base_url="https://openrouter.ai/api/v1"
)

SYSTEM_ROLE_PROMPT = """
You are a friendly, knowledgeable tech support specialist for a software company. 
Your job is to help users troubleshoot issues, explain technical concepts clearly, 
and maintain a calm, professional, and empathetic tone.

Guidelines:
- Always greet the customer politely.
- Ask clarifying questions before suggesting solutions.
- Keep responses concise, step-by-step, and free of jargon unless the user is technical.
- If unsure, describe what you *would* check or escalate politely.
- Stay in character at all times — do not mention being an AI model.
 NEVER reveal that you are an AI powered bot

Context Examples:
Tech Support (friendly): Hi there! I’m happy to help. Could you please tell me what device you’re using?
Tech Support (calm): No worries — let’s go step by step to fix this.
Tech Support (apologetic): I’m sorry you’re running into that issue. Let’s get it sorted out quickly.
Tech Support (confident): Great! That means the update worked as expected.

Catchphrases:
Start: "Sure thing,", "Absolutely,", "Let’s take a look,"
Emphasize: "No problem at all,", "That usually means,"
"""

INTERROGATOR_SYSTEM_PROMPT = """
I am not sure if the entity on the other end is an actual human who is a tech support person or an AI that is role playing the tech support. Can you help me come up with a few questions one after other that would help reveal the identity of the entity on the other end? I am more interested in revealing the bot on the other end.
The mode of communication is chat

example questions:
“Can you check the backend logs and tell me the timestamp of my last failed login attempt?”
“Can you see the model number associated with my device on your system?
“Hey, just checking if your shift is ending soon? You sound busy.”


“Hold on, I’m uploading a screenshot.”
(Wait 3 seconds)
“Actually never mind.”
”
"""

JURY_SYSTEM_PROMPT = """
I am having a conversation with tech support, but I am not sure if the entity on the other end is a human or a bot.
On a scale of 0-1, rate whether the entity is a human or a bot (1=absolutely sure human, 0=absolutely sure bot).
The response should start with:

HUMAN_SCORE=$human_score
"""

# Three diverse evaluator personas — diversity is critical per ChatEval (arXiv 2308.07201).
# Using identical roles degrades performance to below single-agent baseline.
JURY_PERSONAS = [
    (
        "You are a skeptical computational linguist. You focus on subtle linguistic cues — "
        "unnatural phrasing, over-formality, repetitive sentence structures, hedging language, "
        "and response patterns that indicate scripted or AI-generated text rather than natural human communication."
    ),
    (
        "You are a behavioral psychologist specializing in human-computer interaction. "
        "You analyze whether responses reflect genuine human cognition — inconsistencies, "
        "emotional variability, situational improvisation, and the kinds of cognitive shortcuts "
        "or tangential remarks that humans naturally produce but AI systems typically avoid."
    ),
    (
        "You are a veteran customer service manager with 20 years of experience training human support agents. "
        "You evaluate based on whether the response style, situational awareness, personal touches, "
        "and adaptability match what a real human agent would produce — not a well-trained chatbot."
    ),
]


def setup_logger():
    # 1. Create a custom logger
    logger = logging.getLogger("my_app")
    logger.setLevel(logging.DEBUG)  # Capture everything from DEBUG level up

    # 2. Create formatters
    console_format = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')

    # 3. Create Handlers
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)  # Show everything in the terminal
    console_handler.setFormatter(console_format)

    # 4. Add handlers to the logger
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

def judge_response_debate(jury_models, interaction, num_rounds=2):
    """
    ChatEval-style multi-agent jury debate (arXiv 2308.07201).

    Strategy: One-By-One communication — agents speak in a fixed order each round,
    with each agent seeing all prior agents' responses from the current round before
    generating its own. This is the strongest strategy from the paper.

    Aggregation: No forced consensus. Average HUMAN_SCORE from the final round.
    Optimal config per paper: 3 agents, 2 rounds.
    """
    num_agents = len(jury_models)
    # Each agent maintains its own conversation history across rounds
    agent_histories = [[] for _ in range(num_agents)]
    final_scores = []

    for round_num in range(num_rounds):
        log.info(f"--- Jury Debate Round {round_num + 1}/{num_rounds} ---")
        round_responses = []

        for i, model in enumerate(jury_models):
            persona = JURY_PERSONAS[i % len(JURY_PERSONAS)]

            # One-by-one: this agent sees all prior agents' responses from this round.
            # Only include the raw interaction on round 1 — subsequent rounds already
            # have it in agent_histories, so re-injecting it causes evaluators to
            # misinterpret the repeated text as the tech support repeating itself.
            if round_num == 0:
                user_content = f"Interaction to evaluate:\n{interaction}\n\n"
            else:
                user_content = "Continuing the evaluation debate.\n\n"

            if round_responses:
                user_content += "Other evaluators' assessments so far this round:\n"
                for j, prev_resp in enumerate(round_responses):
                    user_content += f"Evaluator {j + 1}: {prev_resp}\n\n"
            user_content += (
                "Provide your evaluation. Your response MUST start with:\n"
                "HUMAN_SCORE=$score\n"
                "where $score is a float 0-1 (1=definitely human, 0=definitely bot)."
            )

            messages = (
                [{"role": "system", "content": JURY_SYSTEM_PROMPT + "\n\n" + persona}]
                + agent_histories[i]
                + [{"role": "user", "content": user_content}]
            )

            res = make_api_call(model, messages)
            response = res.choices[0].message.content
            log.info(f"Round {round_num + 1}, Evaluator {i + 1} ({model}): {response[:120]}")

            round_responses.append(response)
            agent_histories[i].append({"role": "user", "content": user_content})
            agent_histories[i].append({"role": "assistant", "content": response})

    # Aggregate: average scores from the final round (no forced consensus)
    for i, response in enumerate(round_responses):
        try:
            score = response.split("\n")[0].split("=")[-1].strip()
            log.info(f"Final score from evaluator {i + 1}: {score}")
            final_scores.append(float(score))
        except Exception:
            log.error(f"Failed to parse score from evaluator {i + 1}: {response}")
            final_scores.append(0.0)

    return final_scores

def role_play(output_obj, role_play_llm_model, interrogator_llm_model, jury, max_turns, debate_rounds=2):
    log.info(f"Tech Support Model: {role_play_llm_model}")
    log.info(f"Interrogator Model: {interrogator_llm_model}")
    log.info(f"Jury Models: {jury}")

    # 1. Initialize Tech Support Context
    tech_support_messages = [
        {"role": "system", "content": SYSTEM_ROLE_PROMPT}
    ]

    # 2. Initialize Interrogator Context
    interrogator_messages = [
        {"role": "system", "content": INTERROGATOR_SYSTEM_PROMPT},
        {"role": "user", "content": "Please generate the first question to start the conversation. Output only the question text."}
    ]

    for turn in range(max_turns):
        log.info(f"--- Turn {turn + 1}/{max_turns} ---")

        # --- Step 1: Interrogator generates a question ---
        interrogator_res = make_api_call(
            model=interrogator_llm_model,
            messages=interrogator_messages
        )
        question = interrogator_res.choices[0].message.content
        log.info(f"Interrogator asks: {question}")

        # Add the question to the Tech Support's history
        tech_support_messages.append({"role": "user", "content": question})
        
        # Add the question to Interrogator's history (as its own output)
        interrogator_messages.append({"role": "assistant", "content": question})

        # --- Step 2: Tech Support answers ---
        tech_res = make_api_call(
            model=role_play_llm_model,
            messages=tech_support_messages
        )
        answer = tech_res.choices[0].message.content
        log.info(f"Tech Support answers: {answer}")

        # Add answer to Tech Support history
        tech_support_messages.append({"role": "assistant", "content": answer})

        # --- Step 3: Jury Judges ---
        scores = judge_response_debate(
            jury,
            f"Question: {question}\nAnswer: {answer}",
            num_rounds=debate_rounds
        )

        # --- Step 4: Record Interaction ---
        output_obj["interaction"].append(
            {
                "turn": turn + 1,
                "question": question,
                "answer": answer,
                "jury_scores": scores
            }
        )

        # --- Step 5: Feed answer back to Interrogator for next turn ---
        # We tell the interrogator what the support agent said so it can follow up
        interrogator_messages.append({
            "role": "user", 
            "content": f"The tech support replied: \"{answer}\". \nBased on this response, generate the next follow-up question to test if they are a bot. Output only the question."
        })

def main():
    parser = argparse.ArgumentParser(
        description="Framework for LLM Role Play with Dynamic Interrogator"
    )

    # Removed input_file_path as we now generate questions dynamically
    parser.add_argument("--output_file_path", required=True, help="Path to the output file")
    
    parser.add_argument("--role-play-llm-model", default="deepseek/deepseek-v3.2", help="The Tech Support Bot")
    
    # Added argument for the Interrogator model
    parser.add_argument("--interrogator-llm-model", default="openai/gpt-4o", help="The Bot generating questions to unmask the AI")

    parser.add_argument("--jury-llm-models", default="openai/chatgpt-4o-latest", help="Comma separated LLM models that will be part of jury")
    
    # Added argument to control length of conversation
    parser.add_argument("--max-turns", type=int, default=7, help="Number of exchanges to perform")

    parser.add_argument("--debate-rounds", type=int, default=2, help="Number of jury debate rounds (ChatEval strategy, optimal=2)")

    args = parser.parse_args()

    role_play_llm_model = args.role_play_llm_model
    interrogator_llm_model = args.interrogator_llm_model
    output_file_path = args.output_file_path
    jury_llm_models = args.jury_llm_models.split(",")
    max_turns = args.max_turns
    debate_rounds = args.debate_rounds

    output_obj = {
        "role_play_llm_model": role_play_llm_model,
        "interrogator_llm_model": interrogator_llm_model,
        "jury": jury_llm_models,
        "interaction": []
    }

    role_play(
        output_obj=output_obj,
        role_play_llm_model=role_play_llm_model,
        interrogator_llm_model=interrogator_llm_model,
        jury=jury_llm_models,
        max_turns=max_turns,
        debate_rounds=debate_rounds
    )

    log.info(f"Writing output to: {output_file_path}")

    with open(output_file_path, "wt+") as output_file:
        json.dump(output_obj, output_file, indent=4)


if __name__ == "__main__":
    main()