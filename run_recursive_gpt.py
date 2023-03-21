import os
import openai
import sys

# Load your API key from an environment variable or secret management service
openai.api_key = os.getenv("OPENAI_API_KEY")

if sys.stdin.isatty():
    print("You must pass a prompt string via STDIN.")
    exit(-1)

def recursively_prompt_llm(prompt, n=1):
    if prompt.startswith("You are a recursive function"):
        prompt = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0,
            max_tokens=2048,
        )["choices"][0]["text"].strip()
        print(f"response #{n}: {prompt}\n")
        recursively_prompt_llm(prompt, n + 1)

recursively_prompt_llm(sys.stdin.readline())
