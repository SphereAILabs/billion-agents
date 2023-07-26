from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from .article import ArticleData
from .tool import Tool


class OpenAITool(Tool):
    name = "OpenAI"
    description = "use this tool to access a powerful AI model that can perform natural language tasks exceptionally well given article data and a prompt"
    input_schema = "(article: ArticleData, prompt: str)"
    output_schema = "str"

    def variable_params(self) -> list[str]:
        return ["article"]

    def __call__(self, article: ArticleData, prompt: str) -> str:
        llm = ChatOpenAI(temperature=1.0)

        print("doing shit to article")
        print(type(article))

        # for now no semantic search

        # basic: split article content into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=20,
            length_function=len,
            add_start_index=True,
        )

        contents = text_splitter.create_documents([article["content"]])

        # semantic search with Chromadb
        db = Chroma.from_documents(contents, embedding=HuggingFaceEmbeddings())

        # create qa chain (refactor this later)
        qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())

        # call chain
        result = qa_chain({"query": prompt})

        return result
