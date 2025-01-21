from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import pickle
import os
import nltk

nltk.download('averaged_perceptron_tagger')

os.environ["HUGGINGFACEHUB_API_TOKEN"] ='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'


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



def create_embedding_model(urls):
    url_loader = create_url_loader(urls)
    url_data = split_text_into_chunks(url_loader)
    embeddings = embeddings = HuggingFaceEmbeddings(
        model_name="bert-base-nli-mean-tokens"
    )
    vectors_index = FAISS.from_documents(url_data, embeddings)
    if os.path.exists("/Users/nischaoj/Python_Work_DIR/Search_Engine_doc_upload/vectors_index.pkl"):
        os.remove("/Users/nischaoj/Python_Work_DIR/Search_Engine_doc_upload/vectors_index.pkl") 

    with open("/Users/nischaoj/Python_Work_DIR/Search_Engine_doc_upload/vectors_index.pkl", "wb") as f:
        pickle.dump(vectors_index, f)
    
        
    
    return f"Vectors Index created and saved as vectors_index.pkl"
    
    
    
def test():
    urls = ["https://en.wikipedia.org/wiki/Elon_Musk","https://en.wikipedia.org/wiki/SpaceX"]
    create_embedding_model(urls)
    
# test()