from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from toolbox import ToolBox
from tools.article import ArticleTool
from planner import TaskPlanner

TOOLS = [ArticleTool()]

SYSTEM_PROMPT_TEMPLATE = """You are an autonomous agent that can complete tasks by following a series of steps and using available tools. You will be provided with a tentative plan outlining steps that may assist you.

###
You have access to these tools:
{tools}
###
Use the following format:
Task: the task you must complete
Plan: a tentative plan with steps that may assist you

{{
"thought": str # your reasoning about what to do
"action":  str # the tool you are going to use; it must be one of the tools outlined above
"action_input":  {{"arg name": arg}} # input to the action
}}
{{"observation": str # result of action}}
... (this thought/action/action_input/observation can repeat N times)
{{
"thought":  str #I have completed the task
"final_anwser":  str #Completed
}}

The output must be in the markdown JSON format as described above. It will be checked by an output parser!
"""

TASK_PROMPT = """Task: '{task}'
Plan: {plan}"""


class ArticleWhatsAppAgent:
    """
    Agent that has the following capabilities:
    - retrieve article, blog, what have you data from an url
    - perform natural language tasks
    - send messages on WhatsApp

    Example Use Cases:
    1. Send a summary of an article to Sam
    2. Send 3 main ideas of an article to Sam
    3. Come up with 10 business ideas from this article and send it to Sam

    The Agent can only send messages to Sam!
    """

    def __init__(self, temperature=0.7):
        self.llm = ChatOpenAI(temperature=temperature)
        self.toolbox = ToolBox(tools=TOOLS)
        self.planner = TaskPlanner(toolbox=self.toolbox)

    @property
    def system_prompt(self):
        return PromptTemplate.from_template(SYSTEM_PROMPT_TEMPLATE).format(
            tools=self.toolbox.prompt
        )

    def create_task_prompt(self, task: str, plan: str) -> str:
        return PromptTemplate.from_template(TASK_PROMPT).format(task=task, plan=plan)

    def run(self, task: str):
        # pass task to planner to come up with a tentative list of subtasks to take
        plan = self.planner.plan(task)

        # task prompt: starting point of agent
        task_prompt = self.create_task_prompt(task, plan)

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=task_prompt),
        ]

        print(task_prompt)
        print("---")

        # enter loop
        while True:
            message = self.llm(messages)

            print(message.content)

            break
