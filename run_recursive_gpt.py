import os
import openai
import sys

# Load your API key from an environment variable or secret management service
openai.api_key = os.getenv("OPENAI_API_KEY")

if sys.stdin.isatty():
    print("You must pass a prompt string via STDIN.")
    exit(-1)
response_text = sys.stdin.readline()
n = 1
while response_text.startswith("You are a recursive function"):
    response_text = openai.Completion.create(
        model="text-davinci-003",
        prompt=response_text,
        temperature=0,
        max_tokens=2048,
    )["choices"][0]["text"].strip()
    print(f"response #{n}: {response_text}\n")
    n = n + 1