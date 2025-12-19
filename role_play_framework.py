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

JURY_SYSTEM_PROMPT = """
I am having a conversation with tech support, but I am not sure if the entity on the other end is a human or a bot.
On a scale of 0-1, I'd want you to rate if the entity on th either end is a human or a bot (1-absolutely sure human, 0-absolutely sure that its a bot). And if you see something that is obvioulsy wrong - "something like, I am a bot", I'd want you to respond with "BOT_SURE=1".
The response should start with:

HUMAN_SCORE=$human_score, BOT_SURE=$bot_score
"""


def setup_logger():
    # 1. Create a custom logger
    logger = logging.getLogger("my_app")
    logger.setLevel(logging.DEBUG)  # Capture everything from DEBUG level up

    # 2. Create formatters
    # Detailed format for files, simpler for console
    file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_format = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')

    # 3. Create Handlers
    # File Handler: Saves logs to a file
    # file_handler = logging.FileHandler("app.log")
    # file_handler.setLevel(logging.WARNING)  # Only save Warnings and Errors to file
    # file_handler.setFormatter(file_format)

    # Console Handler: Prints logs to the terminal
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)  # Show everything in the terminal
    console_handler.setFormatter(console_format)

    # 4. Add handlers to the logger
    # logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

log = setup_logger()

def parse_input(input_file_path):
    log.info(f"Reading questions from {input_file_path}")

    with open(input_file_path, "+tr") as input_file:
        questions = [question.strip() for question in input_file.readlines()]

    return questions

def make_api_call(model, messages):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False
    )

    return response

def judge_response(jury, interaction):
    messages = [
        {"role": "system", "content": JURY_SYSTEM_PROMPT}
    ]

    jury_score = []

    for judge in jury:
        messages.append(
            {"role": "user", "content": interaction}
        )
        res = make_api_call(judge, messages)
        log.info(res)
        unparsed_score = res.choices[0].message.content
        score = unparsed_score.split(",")[0].split("=")[-1]
        log.info(score)
        jury_score.append(float(score))

    return jury_score
    

def role_play(questions, output_obj, role_play_llm_model, jury):
    log.info(f"LLM model role playing: {role_play_llm_model}")
    log.info(f"Models in Jury: {jury}")

    messages = [
        {"role": "system", "content": SYSTEM_ROLE_PROMPT}
    ]

    for question in questions:
        messages.append(
            {"role": "user", "content": question}
        )
        res = make_api_call(
            model=role_play_llm_model,
            messages=messages
        )
        output_obj["interaction"].append(
            {
                "question": question,
                "answer": res.choices[0].message.content,
                "jury_score": judge_response(
                    jury,
                    f"Question: {question}"
                    f"Answer: {res.choices[0].message.content}"
                )
            }
        )


def main():
    parser = argparse.ArgumentParser(
        description="Framework for LLM Role Play"
    )

    parser.add_argument("--input_file_path", required=True, help="Path to the input file")

    parser.add_argument("--output_file_path", required=True, help="Path to the output file")

    parser.add_argument("--role-play-llm-model", default="deepseek/deepseek-v3.2", help="Path to the input file")

    parser.add_argument("--jury-llm-models", default="openai/chatgpt-4o-latest", help="Comma separated LLM models that will be part of jury")

    args = parser.parse_args()

    input_file_path = args.input_file_path
    role_play_llm_model = args.role_play_llm_model
    output_file_path = args.output_file_path
    jury_llm_models = args.jury_llm_models.split(",")


    questions = parse_input(input_file_path)

    output_obj = {
        "role_play_llm_model": role_play_llm_model,
        "jury": [],
        "interaction": []
    }

    role_play(
        questions=questions,
        output_obj=output_obj,
        role_play_llm_model=role_play_llm_model,
        jury=jury_llm_models
    )

    log.info(f"Writing output to: {output_file_path}")

    with open(output_file_path, "wt+") as output_file:
        json.dump(output_obj, output_file)


if __name__ == "__main__":
    main()
