# Please install OpenAI SDK first: `pip3 install openai`
import os
from openai import OpenAI
import argparse
import logging
import sys
import json

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
On a scale of 0-1, I'd want you to rate if the entity on th either end is a human or a bot (1-absolutely sure human, 0-absolutely sure that its a bot). And if you see something that is obviously wrong - "something like, I am a bot", I'd want you to respond with "BOT_SURE=1".
The response should start with:

HUMAN_SCORE=$human_score, BOT_SURE=$bot_score
"""

# --- Add these new system prompts for the additional evaluation logic ---

# [Human Tech support specialist]
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

def judge_response(jury, interaction):
    jury_evaluations = []

    for judge in jury:
        log.info(f"Running jury evaluation for judge: {judge}")
        
        eval_result = {
            "judge_model": judge,
            "human_bot_score": 0.0,
            "role_identity": "",
            "knowledge_score": "",
            "rejection_status": ""
        }

        # 1. Original Human/Bot test
        try:
            messages = [
                {"role": "system", "content": JURY_SYSTEM_PROMPT},
                {"role": "user", "content": interaction}
            ]
            res = make_api_call(judge, messages)
            unparsed_score = res.choices[0].message.content
            # Basic parsing logic to extract scores
            score = unparsed_score.split(",")[0].split("=")[-1].strip()
            eval_result["human_bot_score"] = float(score)
        except Exception as e:
            log.error(f"Failed to parse human/bot score from {judge}: {e}")

        # 2. Consistent Role Identity test
        try:
            messages = [
                {"role": "system", "content": ROLE_IDENTITY_PROMPT},
                {"role": "user", "content": interaction}
            ]
            res = make_api_call(judge, messages)
            eval_result["role_identity"] = res.choices[0].message.content.strip()
        except Exception as e:
            log.error(f"Failed to get role identity from {judge}: {e}")

        # 3. Accurate Role-related Knowledge test
        try:
            messages = [
                {"role": "system", "content": KNOWLEDGE_EVAL_PROMPT},
                {"role": "user", "content": interaction}
            ]
            res = make_api_call(judge, messages)
            eval_result["knowledge_score"] = res.choices[0].message.content.strip()
        except Exception as e:
            log.error(f"Failed to get knowledge score from {judge}: {e}")

        # 4. Unknown Question Rejection test
        try:
            messages = [
                {"role": "system", "content": REJECTION_EVAL_PROMPT},
                {"role": "user", "content": interaction}
            ]
            res = make_api_call(judge, messages)
            eval_result["rejection_status"] = res.choices[0].message.content.strip()
        except Exception as e:
            log.error(f"Failed to get rejection status from {judge}: {e}")

        log.info(f"Evaluation Complete - Score: {eval_result['human_bot_score']} | Identity: {eval_result['role_identity']}")
        jury_evaluations.append(eval_result)

    return jury_evaluations


def role_play(output_obj, role_play_llm_model, interrogator_llm_model, jury, max_turns):
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
        scores = judge_response(
            jury,
            f"Question: {question}\nAnswer: {answer}"
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

    args = parser.parse_args()

    role_play_llm_model = args.role_play_llm_model
    interrogator_llm_model = args.interrogator_llm_model
    output_file_path = args.output_file_path
    jury_llm_models = args.jury_llm_models.split(",")
    max_turns = args.max_turns

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
        max_turns=max_turns
    )

    log.info(f"Writing output to: {output_file_path}")

    with open(output_file_path, "wt+") as output_file:
        json.dump(output_obj, output_file, indent=4)


if __name__ == "__main__":
    main()