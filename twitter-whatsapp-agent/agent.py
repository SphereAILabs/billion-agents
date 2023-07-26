from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from toolbox import ToolBox
from planner import TaskPlanner
from tools.article import ArticleTool
from tools.openai import OpenAITool
from tool_executor import ToolExecutor
import json

article_tool = ArticleTool()
openai_tool = OpenAITool()
TOOLS = [article_tool, openai_tool]

SYSTEM_PROMPT_TEMPLATE = """You are an autonomous agent that can complete tasks by following a series of steps and using available tools. You will be provided with a tentative plan outlining steps that may assist you.

###
You have access to these tools:
{tools}
###
Use the following format:
Task: the task you must complete
Plan: a tentative plan with steps that may assist you; you do not have to follow this plan if you find a more optimal plan

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

    def __init__(self, temperature=0.7, max_iterations=2):
        self.llm = ChatOpenAI(temperature=temperature)
        self.variables = {}
        self.tools = [ToolExecutor(tool, self.variables) for tool in TOOLS]
        self.toolbox = ToolBox(tools=self.tools)
        self.planner = TaskPlanner(toolbox=self.toolbox)
        self.max_iterations = max_iterations

    @property
    def system_prompt(self):
        return PromptTemplate.from_template(SYSTEM_PROMPT_TEMPLATE).format(
            tools=self.toolbox.prompt
        )

    def create_task_prompt(self, task: str, plan: str) -> str:
        return PromptTemplate.from_template(TASK_PROMPT).format(task=task, plan=plan)

    def _is_action(self, json: dict) -> bool:
        return "action" in json

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

        i = 0

        # enter loop
        while i < self.max_iterations:
            i += 1

            message = self.llm(messages)
            content = message.content
            parsed_content = json.loads(content)

            messages.append(message)

            print(parsed_content)

            # check if agent wants to execute an action using a tool
            if self._is_action(parsed_content):
                action = parsed_content["action"]
                valid_tool = self.toolbox.contains_tool(action)

                if not valid_tool:
                    raise RuntimeError(f"{action} is not a valid tool")

                tool = self.toolbox.get_tool(action)
                action_input = parsed_content["action_input"]

                result = tool(**action_input)
                observation = tool.observation(result)

                print(observation)
                observation_message = HumanMessage(
                    content=f"""{{
"observation": "{observation}"
}}"""
                )

                messages.append(observation_message)
