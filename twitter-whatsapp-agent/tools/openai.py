from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.schema import HumanMessage

from .article import ArticleData
from .tool import Tool


class OpenAITool(Tool):
    name = "OpenAI"
    description = "Use this tool to access a powerful AI model that can perform natural language tasks exceptionally well given article data and a prompt. The prompt should be formatted as a question."
    input_schema = "(article: ArticleData, prompt: str)"
    output_schema = "str"

    def variable_params(self) -> list[str]:
        return ["article"]

    def __call__(self, article: ArticleData, prompt: str) -> str:
        llm = ChatOpenAI(temperature=1.0)

        print("doing shit to article")

        # basic: split article content into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=0,
            length_function=len,
            add_start_index=True,
        )

        contents = text_splitter.create_documents([article["content"]])

        # mmr search with Chromadb
        db = Chroma.from_documents(contents, embedding=HuggingFaceEmbeddings())
        retriever = db.as_retriever(search_type="mmr")

        # combine docs
        docs = retriever.get_relevant_documents(prompt, n_results=5)
        print(len(docs))
        relevant_content = ""
        for doc in docs:
            relevant_content += doc.page_content

        # reconstruct article data with revelant content and pass into prompt
        article_data = {**article, "content": relevant_content}
        llm_prompt = f"""Context: You are an expert reader. You will be asked to read an article and follow an instruction.
Instruction: {prompt}
###
Article: {article_data}
"""
        result = llm([HumanMessage(content=llm_prompt)])

        return result.content
