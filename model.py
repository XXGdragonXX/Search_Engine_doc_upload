import os
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQAWithSourcesChain

os.environ["HUGGINGFACEHUB_API_TOKEN"] ='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'


def create_llm(repo_id,retriever):
    llm = HuggingFaceHub(repo_id = repo_id, model_kwargs={"temperature":0.5, "max_length":64})
    chains = RetrievalQAWithSourcesChain.from_llm(
    llm = llm,
    retriever = retriever
    )
    return chains