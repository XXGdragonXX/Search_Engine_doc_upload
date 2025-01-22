import streamlit as st
import os
from model import create_llm
import pickle
from create_embedding import embedding_model
import nltk

from google.cloud import storage
from google.oauth2 import service_account

# Load the service account key from the environment variable
service_account_info = json.loads(os.environ['GCP_SERVICE_ACCOUNT'])
credentials = service_account.Credentials.from_service_account_info(service_account_info)

# Initialize the Google Cloud Storage client
client = storage.Client(credentials=credentials)



nltk.download('punkt')

st.title("Document Analysis")

st.sidebar.title("Articles (Max 5 URLs)")


urls = []

for i in range(5):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)
    
    
process_url_clicked = st.sidebar.button("Process URLs")

empty_placeholder = st.empty()

if process_url_clicked:
        
    
    with st.spinner("Loading Data from URLs..."):
           
        message = embedding_model(urls)
        print(message) 
        
    st.success('Done!')
    
# vector_store = pickle.load(open("/mnt/data/vectors_index.pkl", "rb"))
try:
    vector_store = st.session_state.vectors_index
except Exception as e:
    print(e)
    

vector_store = vector_store.as_retriever()
chains = create_llm(
    "google/flan-t5-base",
    retriever = vector_store
)
user_input_text = st.text_input(
    "Write your question here...",
    placeholder="Type your question...",
    help="Please enter your question."
)
print("Your question:", user_input_text)

user_input_clicked = st.button("Enter")

if user_input_clicked:
    try:
        with st.spinner("Generating Answers..."):
            print("Please wait for the answer to appear below")
            response = chains({"question": user_input_text}, return_only_outputs=True)
            st.header("Answer")
            st.subheader(response["answer"])
        st.success('Done!')

    except Exception as e:
        print(e)
        st.error("Sorry, I couldn't find an answer to your question.")
        st.stop() 
        

    
    
    
