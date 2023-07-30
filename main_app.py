import dotenv
import os
import streamlit as st
import tempfile
from langchain.llms.huggingface_hub import HuggingFaceHub
from langchain.embeddings.huggingface import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA



dotenv.load_dotenv()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "xxxxxxxxxxxxxxxxxxxxxxxxxx"


st.header("Chat PDF")

file = st.file_uploader("Upload a PDF file", type=["pdf"])
if file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tf:
        tf.write(file.getbuffer())
        file_path = tf.name
    doc = PyPDFLoader(file_path).load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    docs = text_splitter.split_documents(doc)
    embeddings = HuggingFaceInstructEmbeddings(model_name="hku-nlp/instructor-base")
    knowledge_base = FAISS.from_documents(docs, embeddings)

    question = st.text_input("Ask your question here:")
    if question:
        llm = HuggingFaceHub(repo_id="OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5", model_kwargs={"temperature":0.5, "max_length":512})
        chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=knowledge_base.as_retriever())
        response = chain.run(question)
        
        st.success("Completed question.")
        st.write("Answer: ", response)