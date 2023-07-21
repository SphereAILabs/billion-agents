from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage,
    Document,
)
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings


SYSTEM_MESSAGE = """Objective: Answer questions I have about the weather

###
Actions:
- Search[query]: look up a query on Google
ie. Search[What is the weather in NYC]

###
Use the following format:
Question: <the input question you must answer>
Thought: <reasoning on how you are going to accomplish the objective>
Action: <action you are going to take>
Observation: <the result of the action>
... (this Thought/Action/Observation can repeat N times)
Thought: <I now know the final answer>
Final Answer: <the final answer to the original input question>
"""


class WeatherAgent:
    """
    An ReACT agent that answers questions about the weather by looking results up on Google Search. This is
    a "Hello World" example of an agent. Almost all queries should take 2 steps to complete.

    The flow goes as follows:
    1) Search google for answer
    2) Answer question

    The agent has no short term or long term memory. Every time you ask the agent a question, it does not
    have any memory of previous questions.

    When the agent uses Google Search, it only looks at the first 2 documents based on the heuristic
    that the top most result is probably the most relevant. Text from Google Search result is unstructured
    so weather results may or may not be accurate as it depends on how GPT decides to parse it.
    """

    def __init__(self, temperature=0.0, verbose=True):
        self.temperature = temperature
        self.verbose = verbose
        self.model = "gpt-3.5-turbo"
        self.stop = ["\nObservation:"]
        self.llm = ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
        )
        self.system_prompt = SystemMessage(content=SYSTEM_MESSAGE)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
        )
        self.embeddings = HuggingFaceEmbeddings()

    def _create_question(self, question: str) -> HumanMessage:
        return HumanMessage(content=f"Question: {question}")

    def _search(self, query: str) -> Document:
        # search google
        google_search_doc = self._google_search(query)

        # split the search doc
        docs = self.splitter.split_documents([google_search_doc])

        # get the top 2 results
        top_result = docs[0:2]

        content = ""
        for doc in top_result:
            content += doc.page_content

        doc = Document(page_content=content, metadata=top_result[0].metadata)

        return doc

    def _google_search(self, query: str) -> Document:
        uri = f"https://google.com/search?q={query}"
        loader = WebBaseLoader(uri)
        docs = loader.load()
        doc = docs[0]

        return doc

    def _is_final_answer(self, message: BaseMessage):
        content = message.content
        return "Final Answer: " in content

    def _get_action_and_input(self, message: BaseMessage) -> dict:
        content = message.content.strip()
        action_prefix = "Action: "
        action_index = content.index(action_prefix)

        unparsed_action = content[action_index + len(action_prefix) :]
        action, action_input = unparsed_action.split("[")
        action_input = action_input[:-1]

        return {"action": action, "action_input": action_input}

    def query(self, question: str) -> str:
        """
        Ask the agent a question about the weather.
        """
        question_message = self._create_question(question)
        messages = [self.system_prompt, question_message]

        print(question_message, messages)

        # agent runs loop until it gives a Final Answer
        while True:
            ai_message = self.llm(messages, stop=self.stop)

            return ai_message
            break
