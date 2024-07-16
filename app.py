import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
groq_api_key=os.getenv("GROQ_API_KEY")
genai.configure(api_key=api_key)

file_path="Corpus.pdf"
loader=PyPDFLoader(file_path)

docs=loader.load()

llm = ChatGroq(model="llama3-8b-8192")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)

text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits=text_splitter.split_documents(docs)

texts = [docs.page_content for docs in splits]
vector_store = FAISS.from_texts(texts, embedding=embeddings)
retriever=vector_store.as_retriever()

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. Don't give, According to the provided context, this in the answer."
    "If the question is out of the context which provided then don't give any answer, give this statement only and nothing else"
    "For this information, contact us directly."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

st.title("BottleBuddy")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Ask a question:")

if user_input:
    st.session_state.chat_history.append({"role": "human", "content": user_input})
    combined_input = " ".join([message["content"] for message in st.session_state.chat_history if isinstance(message["content"], str)])
    results = rag_chain.invoke({"input": combined_input})
    st.session_state.chat_history.append({"role": "system", "content": results})
    st.write(results['answer'])
