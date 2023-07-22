from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage


TASK_PLANNER_SYSTEM_PROMPT = """Just do; no talk.

You are going to construct a plan to accomplish a task by breaking it down into subtasks. Each step MUST use a tool provided below. List out only the steps; all other information is irrelevant.

###
You have access to these tools:
name: Article
description: use this tool to fetch article data given an url
input: (url: str)

name: OpenAI
description: use this tool to access a powerful AI model that can perform natural language tasks exceptionally well given a piece of text and a prompt
input: (text: str, prompt: str)
"""

STEPS_PROMPT_TEMPLATE = "Steps for: '{task}'"


class TaskPlanner:
    """
    The TaskPlanner is responsible for breaking down a task into a series of subtask. It has access
    to `tools` which it can use.
    """

    def __init__(self, temperature=0.3):
        self.llm = ChatOpenAI(temperature=temperature)
        self.system = TASK_PLANNER_SYSTEM_PROMPT

    def plan(self, task: str) -> str:
        steps_prompt_template = PromptTemplate.from_template(STEPS_PROMPT_TEMPLATE)
        steps_prompt = steps_prompt_template.format(task=task)

        system_message = SystemMessage(content=self.system)
        steps_message = HumanMessage(content=steps_prompt)

        messages = [system_message, steps_message]
        response = self.llm(messages)
        steps = response.content

        return steps
