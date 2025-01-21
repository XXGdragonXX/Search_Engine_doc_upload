import os
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQAWithSourcesChain

# Fetch the API token from the environment variables
# huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
huggingfacehub_api_token = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

# Ensure the token is set correctly before proceeding
if not huggingfacehub_api_token:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN is not set. Please set it in the environment.")

def create_llm(repo_id, retriever):
    llm = HuggingFaceHub(
        repo_id=repo_id,
        model_kwargs={"temperature": 0.5, "max_length": 64},
        huggingfacehub_api_token=huggingfacehub_api_token
    )
    chains = RetrievalQAWithSourcesChain.from_llm(
        llm=llm,
        retriever=retriever
    )
    return chains
