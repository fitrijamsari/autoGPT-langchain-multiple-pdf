import os

import streamlit as st
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceInstructEmbeddings, OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS, Chroma
from PyPDF2 import PdfReader

from htmlTemplates import bot_template, css, user_template

# Access the variables using os.environ
openai_api_key = os.getenv("OPENAI_API_KEY")
huggingface_api_key = os.getenv("HUGGIGFACEHUB_API_TOKEN")


def get_pdf_text(pdf_docs):
    raw_text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            raw_text += page.extract_text()
        # raw_text += " ".join([page.extract_text() for page in pdf_reader.pages])
    return raw_text


def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks


def get_vector_store(text_chunks):
    # create embeddings using OpenAI. This is a paid service.
    embeddings = OpenAIEmbeddings()

    # If we want to use on local(free), we can try embedding using Instructor Embedding but it will be wayyyy slower since it will process using local compute
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")

    # create vector store in FAISS,
    # NOTE: FAISS will store the embeddning in our local machine, so after we close the project, it will disappear. We can user chromadb if we want to store them in cloud.
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    # vectorstore = Chroma.from_documents(text_chunks, embeddings)

    return vectorstore


def get_conversation_chain(vectorstore):
    # create converstaion memory chain using langchain ConversationBufferMemory
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({"question": user_question})
    # st.write(response)
    st.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(
                user_template.replace("{{MSG}}", message.content),
                unsafe_allow_html=True,
            )
        else:
            st.write(
                bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True
            )


def main():
    # Load variables from the .env file
    load_dotenv()

    st.set_page_config(
        page_title="Chat with Multiple PDFs",
        page_icon=":books:",
        # layout="wide",
    )
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with Multiple PDFs :books:")
    user_question = st.text_input(
        "Ask any questions about your documents.",
        placeholder="NOTE: You need to upload your documents first",
    )
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on Process", accept_multiple_files=True
        )
        if st.button("Process"):
            with st.spinner("Processing"):
                # get the pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                # st.write(text_chunks)

                # create vector store using openAI embedding
                vectorstore = get_vector_store(text_chunks)

                # create conversation chain.
                # IMPORTANT: To store the HISTORY of the conversation! In streamlit, when we clicks it will refresh and and rerun everything. So in streamlit we store the conversation in a session.
                st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == "__main__":
    main()
