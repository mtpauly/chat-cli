# TODO: see https://github.com/marcolardera/chatgpt-cli/blob/main/chatgpt.py

import sys
import os
import openai
import tiktoken
from rich.markdown import Markdown
from rich.live import Live
from rich.console import Console
import time


def get_cost(in_tok, out_tok, model):
    in_tok_1k = in_tok / 1000
    out_tok_1k = out_tok / 1000
    if model == "gpt-3.5-turbo":
        return .0015 * in_tok_1k + .002 * out_tok_1k
    if model == "gpt-3.5-turbo-16k":
        return .003 * in_tok_1k + .004 * out_tok_1k
    if model == "gpt-4":
        return .03 * in_tok_1k + .06 * out_tok_1k
    if model == "gpt-4-32k":
        return .06 * in_tok_1k + .12 * out_tok_1k


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        # print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        # print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def get_metadata_str(tok, in_tok, out_tok, model, exec_time=None):
    cost = get_cost(in_tok, out_tok, model)
    if exec_time is not None:
        return f"(tok={tok}, in={in_tok}, out={out_tok}, cost={cost:.4f}, time={exec_time:.1f})"
    else:
        return f"(tok={tok}, in={in_tok}, out={out_tok}, cost={cost:.4f})"


ROLE_SYSTEM = "system"
ROLE_ASSISTANT = "assistant"
ROLE_USER = "user"

DEFAULT_SYSTEM_MESSAGE = """
        You are a helpful programming assistant. You write clean, easy to understand, and well documented code. Your responses should be succinct, prefering to let your code explain itself wherever possible.

        If the request by the user is unclear, or you need to make significant design decisions, ask follow up questions.

        Explain your thinking and plan out your decisions, still being succinct and ensuring you don't use extra or unnecessary words.
        """.strip().replace("    ", "")

MODELS = ["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-32k"]
TEMPERATURE = 0
STREAM = True

model = "gpt-3.5-turbo"
# model = "gpt-4"

if model not in MODELS:
    raise Exception(f"unknown model '{model}'")

openai.api_key = os.getenv("OPENAI_API_KEY")
enc = tiktoken.encoding_for_model(model)

messages = [{"role": ROLE_SYSTEM, "content": DEFAULT_SYSTEM_MESSAGE}]
total_in_tok = 0
total_out_tok = 0

console = Console()

console.print("------------ [0] SYSTEM ------------")
console.print(f"\n{DEFAULT_SYSTEM_MESSAGE}\n")

while True:
    try:
        console.print(f"------------ [{len(messages)}] USER ------------\n")
        user_input = input()
        if len(user_input) == 0:
            continue
    
        messages.append({"role": ROLE_USER, "content": user_input})
        in_tok = num_tokens_from_messages(messages)
        total_in_tok += in_tok

        console.print()
        console.print(get_metadata_str(in_tok, total_in_tok, total_out_tok, model))
        console.print()
        console.print(f"------------ [{len(messages)}] ASSISTANT ------------\n")

        response_start = time.time()
        if STREAM:
            response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=TEMPERATURE,
                    stream=True,
                    )

            response_str = ""
            with Live(Markdown(""), refresh_per_second=2, vertical_overflow="crop") as live:
                for chunk in response:
                    delta = chunk["choices"][0]["delta"]
                    if "content" in delta:
                        response_str += delta["content"]
                        live.update(Markdown(response_str))
        else:
            response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=TEMPERATURE,
                    )
            response_str = str(response["choices"][0]["message"]["content"])
            console.print(Markdown(response_str), end="")

        out_tok = len(enc.encode(response_str))
        total_out_tok += out_tok

        console.print()
        console.print(get_metadata_str(out_tok, total_in_tok, total_out_tok, model, time.time() - response_start))
        console.print()

        messages.append({
            "role": ROLE_ASSISTANT,
            "content": response_str,
        })

    except KeyboardInterrupt:
        console.print("\n\nUser exited")
        sys.exit()
