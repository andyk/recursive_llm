# Recursive LLM prompts

The idea here is to implement recursion using English as the programming language and an LLM (e.g., GPT-3.5) as the runtime.

Basically we come up with an LLM prompt which causes the model to return another slightly updated GPT prompt. More specifically, the prompts contain state and each recursively generated prompt updates that state to be closer to an end goal (i.e., a base case).

It’s kind of like traditional recursion in code, but instead of having a function that calls itself with a different set of arguments, there is a prompt that returns itself with specific parts updated to reflect the new arguments.

Here is an infinitely recursive fibonacci prompt:

> You are a recursive function. Instead of being written in a programming language, you are written in English.  You have variables FIB_INDEX = 2, MINUS_TWO = 0, MINUS_ONE = 1, CURR_VALUE = 1. Output this paragraph but with updated variables to compute the next step of the Fibbonaci sequence.

To “run this program” we can paste it into OpenAI playground, and click run, and then take the result and run that, etc.

https://user-images.githubusercontent.com/228998/226147800-fff1ba10-118c-47ae-9772-35be5b15e4c0.mp4


In theory, because this does not specify a base case, we could stay in this loop of copying and pasting and running these successive prompts forever, each prompt representing one number in the Fibonacci sequence.

In `run_recursive_gpt.py` we automate this by writing a recursive Python function that repeatedly calls the OpenAI API with each successive prompt until the result satisfies the base case. Here's the recursive Python function from [run_recursive_gpt.py](https://github.com/andyk/recursive_llm/blob/main/run_recursive_gpt.py):

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

The bigger picture goal here is to explore of using prompts to generate new prompts, and more specifically the case where the prompts contain state and each recursively generated prompt updates that state to be closer to an end goal.

This is partly inspired by the historical AI system from CMU called [General Problem Solver (GPS)](https://en.wikipedia.org/wiki/General_Problem_Solver), which is built on an idea called [means-ends analysis](https://en.wikipedia.org/wiki/Means%E2%80%93ends_analysis). Here is how it works at a high level: the user specifies a goal, and then GPS evaluates the difference between current state of the world and the goal state and then tries take a step to reduce the gap, then repeat. For a high-level summary of GPS, see [Patrick H. Winston's description of it](https://www.youtube.com/watch?v=PimSbFGrwXM&t=189s) in his OCW lecture on Cognitive Architectures as part of his AI course at MIT. We also see very similar ideas in goal-conditioned RL \[[6](https://www.youtube.com/watch?v=tzieElmtAjs&list=PLkFD6_40KJIwhWJpGazJ9VSj9CFMkb79A&t=1170s)\].

The ability for LLMs to break down problem into sub-steps via [Chain of thought (CoT) prompting](https://en.wikipedia.org/wiki/Chain-of-thought_prompting) \[3\]\ reminded me of this part of Winston's lecture. And so I wanted to try making a prompt that (1) contains state and (2) can be used to generate another prompt which has updated state.

I also want to further explore how (and when) to best leverage what the LLM has memorized. The way humans do math in our heads is an interesting analog: our brain (mind?) uses two types of rules that we have memorized:

1. algebraic rules for rewriting (part of) the math problem
2. atomic rules things like 2+2=4

So I'm wondering if we could write a "recursive" LLM prompt that achieves a similar thing.

This direction of reasoning is inspired by another classic CMU AI research project called [ACT-R](https://en.wikipedia.org/wiki/ACT-R). This project, led by John R. Anderson, explored how humans do math in their head and tried to apply those lessons to their AI agent architecture \[4\].

The ACT-R group partnered up with cognitive scientists & neuroscientists and performed FMRIs on students while they were doing math problems. 


## Observations & discussion

I was a little suprised at how (and how frequently) the model generates incorrect results. E.g., with the Fibonacci sequence prompt, sometimes it skips a number entirely, sometimes it produces a number that is off-by-some but then gets the following number(s) correct. For example, at the very end of the screen capture video above (i.e., "response #16") it prints 2504 but the correct answer is 2584.

![wrong-answer-18th-fib-seq](https://user-images.githubusercontent.com/228998/226428779-845c299c-c158-4634-94d8-cc265aa86f19.png)

I wonder how much of this is because the model has memorized the Fibonacci sequence. It is possible to have it just return the sequence in a single call, but that isn't really the point here. Instead this is more an exploration of how to use these large language models as part of a more active agent system, in the spirit of \[1\]\[2\]. In this context our "agent" is just a dumb python tail recursion that uses the current prompt to generate the next prompt, etc. etc. until it arrives at a base case.


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

## References

\[1\] [A simple Python implementation of the ReAct pattern for LLMs. Simon Willison.](https://til.simonwillison.net/llms/python-react-pattern)

\[2\] [ReAct: Synergizing Reasoning and Acting in Language Models. Shunyu Yao et al.](https://react-lm.github.io/)

\[3\] [Large Language Models are Zero-Shot Reasoners. Takeshi Kojima et al.](https://arxiv.org/abs/2205.11916)

\[4\] [ACT-R project list of publications about Publications in Mathematical Problem Solving](http://act-r.psy.cmu.edu/category/problem-solving-and-decision-making/mathematical-problem-solving/)

\[5\] [Lecture by Sergey Levine as part of UC Berkeley CS285 - graduate level Deep Reinforcement Learning](https://www.youtube.com/watch?v=tzieElmtAjs&t=1170s)

## Acknowledgments

Thanks for [Andrew Krioukov](https://github.com/krioukov), [Nick Jalbert](https://github.com/nickjalbert/), [Beth Trushkowsky](https://www.cs.hmc.edu/~beth), and [Rob Carroll](https://www.linkedin.com/in/robert-carroll-97b71738/), and [the Hackernews community](https://news.ycombinator.com/item?id=35234276#35241825) for contributions and feedback! 
