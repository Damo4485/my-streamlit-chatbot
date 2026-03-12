import pdfplumber
import docx
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import os

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]  # Use secrets

st.header("Multi-Document Chatbot")

with st.sidebar:
    st.title("Your Documents")
    files = st.file_uploader(
        "Upload document files and start asking questions",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

if "messages" not in st.session_state:
    st.session_state["messages"] = []

def extract_text_from_file(file):
    if file.type == "application/pdf":
        with pdfplumber.open(file) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    elif file.type.startswith("text/"):
        return file.read().decode("utf-8")
    else:
        return ""

def display_chat_history():
    for msg in st.session_state["messages"]:
        if msg["role"] == "human":
            st.markdown(f"**You:** {msg['content']}")
        else:
            st.markdown(f"**Bot:** {msg['content']}")

# Combine text from all uploaded files
all_text = ""
if files:
    for file in files:
        file.seek(0)  # to reset file pointer in Streamlit
        all_text += extract_text_from_file(file) + "\n"

if all_text:
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(all_text)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY
    )
    vector_store = FAISS.from_texts(chunks, embeddings)

    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4}
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        max_tokens=1000,
        openai_api_key=OPENAI_API_KEY
    )

    with st.form("chat_input", clear_on_submit=True):
        display_chat_history()
        user_question = st.text_input("Type your question here", key="user_question")
        submitted = st.form_submit_button("Send")

    if submitted and user_question:
        st.session_state["messages"].append({"role": "human", "content": user_question})

        # Build prompt with history
        prompt_messages = [
            (
                "system",
                "You are a helpful assistant answering questions about the uploaded documents.\n\n"
                "Guidelines:\n"
                "1. Provide complete, well-explained answers using the context below.\n"
                "2. Include relevant details, numbers, and explanations to give a thorough response.\n"
                "3. If the context mentions related information, include it to give fuller picture.\n"
                "4. Only use information from the provided context - do not use outside knowledge.\n"
                "5. Summarize long information, ideally in bullets where needed\n"
                "6. If the information is not in the context, say so politely.\n\n"
                "Context:\n{context}"
            )
        ]
        for msg in st.session_state["messages"]:
            if msg["role"] == "human":
                prompt_messages.append(("human", msg["content"]))
            else:
                prompt_messages.append(("ai", msg["content"]))

        prompt = ChatPromptTemplate.from_messages(prompt_messages)

        chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
        )

        response = chain.invoke(user_question)
        st.session_state["messages"].append({"role": "ai", "content": response})
        st.rerun()
else:
    st.info("Please upload at least one document file (PDF, DOCX, or TXT) to start chatting.")
