from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit as st
import pickle
import os
import nltk

nltk.download('averaged_perceptron_tagger')

import json
import os
from google.cloud import storage
from google.oauth2 import service_account

# Load the service account key from the environment variable
service_account_info = json.loads(os.environ['GCP_SERVICE_ACCOUNT'])
credentials = service_account.Credentials.from_service_account_info(service_account_info)

# Initialize the Google Cloud Storage client
client = storage.Client(credentials=credentials)

# Fetch the API token from the environment variables
huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Ensure the token is set correctly before proceeding
if not huggingfacehub_api_token:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN is not set. Please set it in the environment.")

def create_url_loader(urls):
    loader = UnstructuredURLLoader(
    urls=urls
    )
    
    url_loader = loader.load()
    print(type(url_loader))
    for doc in url_loader:
        print(f"Document metadata: {doc.metadata}")
    return url_loader


def split_text_into_chunks(data):
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        separators = ["\n", ".", "!", "?"]
    )

    chunks = text_splitter.split_documents(data)
    print(f"-------------{len(chunks)}------------")
    return chunks

def upload_pickle_to_gcs(local_file_path, bucket_name, blob_name):
    # Create a GCS client
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    # Upload the pickle file to GCS
    blob.upload_from_filename(local_file_path)
    print(f"Pickle file uploaded to {bucket_name}/{blob_name}")



def embedding_model(urls):
    url_loader = create_url_loader(urls)
    url_data = split_text_into_chunks(url_loader)
    embeddings = embeddings = HuggingFaceEmbeddings(
        model_name="bert-base-nli-mean-tokens"
    )
    vectors_index = FAISS.from_documents(url_data, embeddings)
    print(f"This is the vector index : {vectors_index}")
    # file_path = "/mnt/data/vectors_index.pkl"
    # if os.path.exists(file_path):
    #     # If file exists, remove it to regenerate
    #     os.remove(file_path)

    # # Save the vectors index as a pickle file
    temp_dir = './tmp'
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir,vectors_index.pkl)
    with open(file_path, "wb") as f:
        pickle.dump(vectors_index, f)
    upload_pickle_to_gcs(file_path, 'project_bucket_1998', 'project_bucket_1998/document_embedding')
    # st.session_state.vectors_index = vectors_index

    return f"Vectors Index created and saved to session state"

    # return f"Vectors Index created and saved as vectors_index.pkl"
    
    
    
def test():
    urls = ["https://en.wikipedia.org/wiki/Elon_Musk","https://en.wikipedia.org/wiki/SpaceX"]
    create_embedding_model(urls)
    
# test()
