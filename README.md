# Recursive LLM prompts

The idea here is to implement recursion using English as the programming language and GPT as the runtime.

Basically we come up with a GPT prompt which causes the model to return another slightly updated GPT prompt. More specifically, the prompts contain state and each recursively generated prompt updates that state to be closer to an end goal (i.e., a base case).

It’s kind of like traditional recursion in code, but instead of having a function that calls itself with a different set of arguments, there is a prompt that returns itself with specific parts updated to reflect the new arguments.

Here is an infinitely recursive fibonacci prompt:

> You are a recursive function. Instead of being written in a programming language, you are written in English.  You have variables FIB_INDEX = 2, MINUS_TWO = 0, MINUS_ONE = 1, CURR_VALUE = 1. Output this paragraph but with updated variables to compute the next step of the Fibbonaci sequence.

To “run this program” we can paste it into OpenAI playground, and click run, and then take the result and run that, etc.

https://user-images.githubusercontent.com/228998/226147800-fff1ba10-118c-47ae-9772-35be5b15e4c0.mp4


In theory, because this does not specify a base case, we could stay in this loop of copying and pasting and running these successive prompts forever, each prompt representing one number in the Fibonacci sequence.

In `run_recursive_gpt.py` we automate this recursion by writing a recursive function in Python that repeatedly calls the OpenAI API with the original and then subsequently generatated prompts until the result satisfies the base case. Here's the recursive Python function from [run_recursive_gpt.py](https://github.com/andyk/recursive_llm/blob/main/run_recursive_gpt.py):

```python
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
```

And here's what it looks like when you run it:

https://user-images.githubusercontent.com/228998/226147804-948151a5-f534-4e20-a957-a810c23516aa.mp4


## Big picture goal and related work

My bigger picture goal here is to explore of using prompts to generate new prompts, and more specifically the case where the prompts contain state and each recursively generated prompt updates that state to be closer to an end goal.

This is partly inspired by Patrick H. Winston's MIT OCW lecture on Cognitive Architectures, in particular [his summary](https://www.youtube.com/watch?v=PimSbFGrwXM&t=189s) of the historical system from CMU called General Problem Solver (GPS) in which they try to identify a goal and then have the AI evaluate the difference between the current state and the goal and try take steps to bridge the gap.

The ability for LLMs to break down problem into sub-steps (a la "Let's think step by step" \[3\]) reminded me of this part of Winston's lecture. And so I wanted to try making a prompt that (1) contains state and (2) can be used to generate another prompt which has updated state.

I also want to further explore how (and when) to best leverage what the LLM has memorized.

The way humans do math in our heads is an interesting analog: our brain (mind?) uses two types of rules that we have memorized:

1. algebraic rules for rewriting (part of) the math problem
2. atomic rules things like 2+2=4

So I'm wondering if we could write a "recursive" LLM prompt that achieves a similar thing.

This direction of reasoning is inpsired by another classic CMU AI research project on Cognitive Architectures, John R. Anderson's group explored how humans do math in their head as part of his ACT-R project \[4\].

The ACT-R group partnered up with cognitive scientists & neuroscientists and performed FMRIs on students while they were doing math problems. 

## Observations

I was a little suprised at how (and how frequently) the model generates incorrect results. E.g., with the Fibonacci sequence prompt, sometimes it skips a number entirely, sometimes it produces a number that is off-by-some but then gets the following number(s) correct. For example, at the very end of the screen capture video above (i.e., "response #16") it prints 2504 but the correct answer is 2584.

![wrong-answer-18th-fib-seq](https://user-images.githubusercontent.com/228998/226428779-845c299c-c158-4634-94d8-cc265aa86f19.png)

I wonder how much of this is because the model has memorized the Fibonacci sequence. It is possible to have it just return the sequence in a single call, but that isn't really the point here. Instead this is more an exploration of how to agent-ify the model in the spirit of \[1\]\[2\] via prompts that generate other prompts.

This reminds me a bit of how a CPU works, i.e., as a dumb loop that fetches and executes the next instruction, whatever it may be. Well in this case our "agent" is just a dumb python loop that fetches the next prompt (which is generated by the current prompt) whatever it may be... until it arrives at a prompt that doesn't lead to another prompt.

\[1\] A simple Python implementation of the ReAct pattern for LLMs. Simon Willison. https://til.simonwillison.net/llms/python-react-pattern

\[2\] ReAct: Synergizing Reasoning and Acting in Language Models. Shunyu Yao et al. https://react-lm.github.io/ 

\[3\] Large Language Models are Zero-Shot Reasoners - https://arxiv.org/abs/2205.11916 

\[4\] https://www.amazon.com/Soar-Cognitive-Architecture-MIT-Press/dp/0262538539

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
