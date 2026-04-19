import os
from dotenv import load_dotenv
import openai
import pinecone as pc_client

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

INDEX_NAME = "pinecone-practice-1"


def init_env():
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")


def create_clients():
    init_env()

    pc = pc_client.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    embedding = OpenAIEmbeddings()

    return pc, embedding


def create_index(pc, embedding):
    from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    loader = DirectoryLoader("./pdfs", glob="**/*.pdf",
                             loader_cls=PyMuPDFLoader)
    data = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=100)
    texts = splitter.split_documents(data)

    index = pc.create_index(
        INDEX_NAME, dimension=embedding.dimensions, metric="cosine")
    vectorstore = PineconeVectorStore(index, embedding)

    vectorstore.add_texts([t.page_content for t in texts])


def get_qa(pc, embedding):
    from langchain_openai import OpenAI
    from langchain_classic.chains import create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.prompts import ChatPromptTemplate

    existing_indexes = [i.name for i in pc.list_indexes()]

    if INDEX_NAME not in existing_indexes:
        create_index(pc, embedding)

    docsearch = PineconeVectorStore(pc.Index(INDEX_NAME), embedding)

    llm = OpenAI()
    retriever = docsearch.as_retriever(search_kwargs={"k": 5})

    system_prompt = (
        "Use the given context to answer the question. "
        "If you don't know the answer, say you don't know. "
        "Use three sentence maximum and keep the answer concise. "
        "Context: {context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    qa = create_retrieval_chain(retriever, question_answer_chain)

    return qa
