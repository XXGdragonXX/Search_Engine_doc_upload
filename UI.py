import streamlit as st
import os
from model import create_llm
import pickle
from create_embedding import embedding_model
import nltk



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
           
        embedding_model(urls)
        
    st.success('Done!')
    
            
# vector_store = pickle.load(open("/mnt/data/vectors_index.pkl", "rb"))
if "vectors_index" in st.session_state:
    vector_store = st.session_state.vectors_index
else:
    st.warning("Vectors index not available. Please generate it.")

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
        

    
    
    
