from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

SYSTEM_MESSAGE = """Objective: Answer questions I have about the weather

###
Actions:
Search[query]: search query on Google
ie. Search[what is the weather in NYC]

Answer[result]: answer the question
ie. Answer[the weather in NYC is 56F]

###
Use the following format:
Question: <the input question you must answer>
Thought: <reasoning on how you are going to accomplish the objective>
Action: <action you are going to take>
Observation: <the result of the action>
... (this Thought/Action/Observation can repeat N times)
Thought: <I now know the final answer>
Action: Answer[result]
"""


class WeatherAgent:
    def __init__(self, temperature=1.0, verbose=True):
        self.temperature = temperature
        self.verbose = verbose
        self.model = "gpt-3.5-turbo"
        self.stop_sequences = ["\nObservation:"]
        self.llm = ChatOpenAI(
            model=self.model, temperature=self.temperature, stop=self.stop_sequences
        )
        self.system_prompt = SystemMessage(content=SYSTEM_MESSAGE)

    def _create_question(self, question: str) -> HumanMessage:
        return HumanMessage(content=f"Question: {question}")

    def query(self, question: str) -> str:
        """
        Ask the agent a question about the weather
        """
        pass
