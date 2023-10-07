'''
Main file for the Streamlit app.

Maverick, Ian, Jay, Richard
10.07.2023


TODO
- Use atlas instead (creative use of MongoDB Atlas)
- Information below
- 3 Columns with information
- Google cloud to generate PDF?
- Information to help the user
- Any cloud computing to help generate PDFs?
- Edit Theme for Morgan Colors
- Py thing to have langchain repeat until correct format

- Chat with our virtual assistant, upload a document for more context

https://www.youtube.com/watch?v=6fs80o7Xm4I&ab_channel=FaniloAndrianasolo

- Uploaded documents
How homeless do I look?
- Recommended Questions

'''


import os, tempfile
import streamlit as st
from langchain.prompts import *
from langchain import LLMChain
import configparser
import openai

from langchain.embeddings.openai import OpenAIEmbeddings

import streamlit as st
from langchain.llms.openai import OpenAI
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationChain

from dataclasses import dataclass
from typing import Literal

from langchain import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationSummaryMemory

from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential

import os

config = configparser.ConfigParser()
config.read("config.ini")

openai_api_key = st.secrets["OPENAI"]["OPENAI_API_KEY"]
azure_key = st.secrets["AZURE"]["AZURE_KEY"]
azure_endpoint = st.secrets["AZURE"]["AZURE_ENDPOINT"]

DOCUMENT_TYPES = [
    'Court Order',
    'Medical Record',
    'Medical Bill',
    'Correspondance',
    'Police Report',
    'Other'
]

def load_css():
    with open("static/styles.css", "r") as f:
        css = f"<style>{f.read()}</style>"
        st.markdown(css, unsafe_allow_html=True)


@dataclass
class Message:
    origin: Literal["human", "llm"]
    message: str

def initialize_session_state():
    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "conversation" not in st.session_state:
        llm = OpenAI(temperature=0, openai_api_key=openai_api_key, model_name="gpt-3.5-turbo")
        st.session_state.conversation = ConversationChain(llm=llm, memory=ConversationSummaryMemory(llm=llm))


def on_click_callback():
    human_prompt = st.session_state["human_prompt"]
    llm_response = st.session_state.conversation.run(human_prompt)

    st.session_state["history"].append(
        Message("human", human_prompt)
    )
    st.session_state["history"].append(
        Message("llm", llm_response)
    )


def main():
    initialize_session_state()
    load_css()
    
    # Begin the Streamlit App Here
    st.title('MorganAI')
    st.write('lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.')


    # Do the sidebar here
    st.sidebar.title('Enter Document URL')
    file = st.sidebar.file_uploader("Upload a file", type=["pdf", "png", "jpg", "jpeg"])
    if file:
        document_analysis_client = DocumentAnalysisClient(
            endpoint=azure_endpoint,
            credential=AzureKeyCredential(azure_key)
        )

        poller = document_analysis_client.begin_analyze_document(
            "prebuilt-receipt", file
        )

        result = poller.result()
        st.sidebar.write(result.content)
    else:
        st.warning('Please upload a file for further analysis')

              
    st.markdown('---')

    if file:
        st.success(f'Document type: {DOCUMENT_TYPES[int(get_type_of_document(result.content)) - 1]}')

    st.markdown('---')

    st.write('Hi, I am MorganAI. I am here to help you with your legal needs.')
    chat_palceholder = st.container()
    prompt_placeholder = st.form("Chat-form")

    with chat_palceholder:
        for chat in st.session_state.history:
            div = f"""
                <div class="chat-row {'' if chat.origin == 'llm' else 'row-reverse human-bubble'}">{chat.message}</div>
            """
            st.markdown(div, unsafe_allow_html=True)

    with prompt_placeholder:
        st.markdown("**Chat** - _Press Enter to submit_")
        cols = st.columns((6, 1))
        cols[0].text_input("Chat", value="Hello", key='human_prompt')
        cols[1].form_submit_button("Send", type="primary", on_click=on_click_callback)

def get_vectorstore(source_doc):
    # Save uploaded file temporarily to disk, load and split the file into pages, delete temp file
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(source_doc.read())

    loader = PyPDFLoader(tmp_file.name)
    pages = loader.load_and_split()
    os.remove(tmp_file.name)

    # Create embeddings for the pages and insert into Chroma database
    embeddings=OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectordb = Chroma.from_documents(pages, embeddings)

    return vectordb

def get_type_of_document(document_text: str) -> str:
    template = """You are an AI legal agent that is working at an injury law firm to classify certain documents. You are given a document and you need to classify it into one of the following categories:\n

    """ + "".join([f"{i+1}. {doc_type}\n" for i, doc_type in enumerate(DOCUMENT_TYPES)]) + """\nThe document you are given (by the user) has been put through an OCR system to convert it from an image to text. The OCR system is not perfect and there may be some errors in the text. Only use the information from the document, and limit your answer to only a single number (e.g. "1"..."7" without the quotes). The user will provide all information from the document between triple backticks in their prompt."""

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(template),
        HumanMessagePromptTemplate.from_template("Here is the documentation: ```{document_text}```"),
    ])

    chain = LLMChain(
        llm=OpenAI(temperature=0, openai_api_key=openai_api_key,
        model_name='gpt-3.5-turbo'),
        prompt=prompt
    )

    response = chain.run(document_text=document_text)
    return response



if __name__ == '__main__':
    main()