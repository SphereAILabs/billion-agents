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
from langchain.vectorstores import Chroma
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

    The agent uses ChromaDB for storing embeddings of Google Search results and uses RAG (Retrieval Augmented Generation) to
    produce the final answer. After the agent answers the question, the vector database is disposed.
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
            chunk_size=500,
            chunk_overlap=20,
        )
        self.embeddings = HuggingFaceEmbeddings()

    def _create_question(self, question: str) -> HumanMessage:
        return HumanMessage(content=f"Question: {question}")

    def _google_search(self, query: str) -> Document:
        uri = f"https://google.com/search?q={query}"
        loader = WebBaseLoader(uri)
        docs = loader.load()
        doc = docs[0]

        return doc

    def _create_inmemory_db_from_doc(self, doc: Document) -> Chroma:
        # split the document in chunks using RecursiveCharacterTextSplitter
        docs = self.splitter.split_documents([doc])
        db = Chroma.from_documents(docs, embedding=self.embeddings)

        return db

    def _is_final_answer(self, message: BaseMessage):
        pass

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
