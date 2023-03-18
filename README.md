# Recursive LLM prompts

The idea is to implement recursion using English as the programming language and GPT as the runtime.

Basically we come up with a GPT prompt which causes the model to return another slightly GPT prompt. 

It’s kind of like traditional recursion in code, but instead of having a function that calls itself with a different set of arguments, there is a prompt that returns itself with specific parts updated to reflect the new arguments.

Here is an infinitely recursive fibonacci prompt:

    You are a recursive function. Instead of being written in a programming language, you are written in English. You have variables FIB_INDEX = 2, MINUS_TWO = 0, MINUS_ONE = 1, CURR_VALUE = 1. Output this paragraph but with updated variables to compute the next step of the Fibbonaci sequence.

To “run this program” we can paste it into OpenAI playground, and click run, and then take the result and run that, etc.

https://raw.githubusercontent.com/andyk/recursive_gpt/main/shell_demo_recursive_no_base_case_2.mp4

In theory, because this does not specify a base case, we could stay in this loop of copying and pasting and running these successive prompts forever, each prompt representing one number in the Fibonacci sequence.

In `run_recursive_gpt.py` we automate this recursion by writing a minimal loop in Python and using the OpenAI API to call the model with the prompt, check off the result satisfies the base case, and if not call the model with the result that was returned (which should be the updated prompt).

Essentially:
```
response_text = recursive prompt
while response_text.startswith("You are a recursive function"):
    response_text = openai.Completion.create(
        model="text-davinci-003",
        prompt=response_text,
        …)
```



## To run:

    pip install openai
    OPENAI_API_KEY=YOUR_KEY_HERE python run_recursive_gpt.py < prompt_fibonnaci_no_base_case.txt
    OPENAI_API_KEY=YOUR_KEY_HERE python run_recursive_gpt.py < prompt_fibonnaci_base_case.txt
    OPENAI_API_KEY=YOUR_KEY_HERE python run_recursive_gpt.py < prompt_counting.txt


To run without python (more manual process):

1. Open [OpenAI Playground](https://beta.openai.com/playground)
2. Paste the prompt into the model (text-davinci-003", temperature=0, max_tokens=2048)
3. Click Submit. The model output should be a new prompt.
4. Keep running each successive prompt till the base case is hit.
