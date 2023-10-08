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

- Questions necessary for the ChatBot

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

import openai
import re
import json

from langchain.embeddings.openai import OpenAIEmbeddings

import streamlit as st
from langchain.llms.openai import OpenAI
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationSummaryMemory
from langchain.chains.conversation.memory import ConversationSummaryMemory
from langchain.chains import ConversationChain

from dataclasses import dataclass
from typing import Literal

from langchain import OpenAI
from langchain.callbacks import get_openai_callback

from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential

import os

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
    
    # Begin the Streamlit App Here
    st.title('LegalFlow')
    st.subheader('Your assistant for document analysis and legal advice.')
    st.write('LegalFlow was created by students participating in the UCF Knight Hacks 2023 Hackathon to help with document classification and extraction of information.')

    # Do the sidebar here
    st.sidebar.title('Upload a Legal Document')
    file = st.sidebar.file_uploader("Upload a file here", type=["pdf", "png", "jpg", "jpeg"])

    # Add information for the file
    if file:
        with st.spinner('Analyzing your document...'):
            document_analysis_client = DocumentAnalysisClient(
                endpoint=azure_endpoint,
                credential=AzureKeyCredential(azure_key)
            )
            poller = document_analysis_client.begin_analyze_document(
                "prebuilt-receipt", file
            )
        
        result = poller.result()
        st.sidebar.success('Document uploaded successfully')
        st.sidebar.write(result.content)
    else:
        st.warning('Please use the sidebar to upload a document for further environment, or chat with the LegalFlow assistant for general questions')

              
    st.markdown('---')

    if file:
        doc_type = get_type_of_document(result.content)

        st.subheader('Document Analysis')
        st.write(f'Number of pages: {len(result.pages)}')
        st.write(f'Document Type: {doc_type}')
        st.subheader('Additional Information')
        kv_pairs = get_important_information(result.content)
        for key, value in kv_pairs.items():
            st.write(f'{key}: {value}')

    st.markdown('---')

    st.write('Hello, I am LegalFlowChat and I am here to help you with any questions you may have about your legal documents. Please use the chatbox below to ask any questions you may have.')
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

    """ + "".join([f"{i+1}. {doc_type}\n" for i, doc_type in enumerate(DOCUMENT_TYPES)]) + """\nThe document you are given (by the user) has been put through an OCR system to convert it from an image to text. The OCR system is not perfect and there may be some errors in the text. The user will provide all information from the document between triple backticks in their prompt.
    
    As you walk through the document, take note of some of the information that you see. Make sure you identify the most important parts of the document. Look at important structures that you find and repeated words and phrases throughout the documents.

    Here are some observations that may help in the decision making process:
    - If a document contains "dear sir or madam" it is likely a correspondance
    - If a document has much information pertaining to physicians, Medicare, Medicaid, CT Scans, or the like, it is likely a medical record
    - If a document contains a reporting number, a date of incident, or a police officer's name, or a person record, it is likely a police report
    - If a document contains a judge's name, a case number, or a court name, it is likely a court order

    Make sure to think aloud as you make your solution. After you finish thinking aloud, put your final answer between <<< >>> as it is written above. Only use the options listed above. Here is an example of your process:
    
    
    User: ```<<<DOCUMENT INFORMATION>>>```
    AI Agent: In this document, I see information related to Federal Health 
Insurance Portability and Accountability Act (HIPAA) as well as the name of a healthcare provider. I do see the token 'correspondance' in this document, however, it is in the context of records that need to be send to the patient. I believe this document is a medical record. <<<Medical Record>>>
    
    Ensure you follow proper formatting as you use your best judgement to classify the document.
    """

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(template),
        HumanMessagePromptTemplate.from_template("```{document_text}```"),
    ])

    chain = LLMChain(
        llm=OpenAI(temperature=0, openai_api_key=openai_api_key,
        model_name='gpt-3.5-turbo'),
        prompt=prompt
    )

    response = chain.run(document_text=document_text)
    try:
        response = re.search(r'<<<(.+?)>>>', response).group(1) # Get the text between <<< >>>
        return response
    except:
        return response[-3:]



def get_important_information(document_text: str):
    template = """You are an AI legal agent that is working at an injury law firm to classify certain documents. You have been given a legal document and you need to extract important information from it. Keep in mind that the documents may be Medical Records, Medical Bills, Correspondance, Police Reports, or Court Orders. The document you are given (by the user) has been put through an OCR system to convert it from an image to text. The OCR system is not perfect and there may be some errors in the text. In addition, the text is a single string of tokens, however everything is relatively well-ordered you need to put the correct pieces of information together. The user will provide all information from the document between triple backticks in their prompt.

    Use this knowledge to get the pieces of information you believe are necessary. Don't forget to think aloud as you go through the document. Here is an example of your process:

    User: ```<<<DOCUMENT INFORMATION>>>```
    Agent: Since this document is a Medical Record, there may not be explicit key value pairs here. However, I will get any information that I can. The document does give some information about the attn (attorney) as well as the following healthcare provider. I will parse that information and create a json
    <<<{"attn": "Preston Blair", "healthcare_provider": {"name": "MD Now Medical Center - West Flagler", "Address": "20 N Orange Ave, Suite 1600, Orlando, FL 32801", "Phone": "813-223-5505"}}>>>

    USER: ```<<<DOCUMENT2 INFORMATION>>>```
    Agent: Since I know this is a police report, I will look for any possible key value pairs related to injury and severity. I want to gather not just basic data, but also data that may help the other agents estimate the value of the case and help deliver a possible solution to the client. In my json, I will include the following basic pieces of information that I see: the date, the billing date for this particular item, the name of the source provider, the name of the patient. More importantly, I see information about how her injury occurred, the type of injury, and the treatment that she received. I will include all of this information in my json as well since this is information that will help to steer their solution and support for our client in the right direction.
    
    <<<{"due date": "11/13/2021", "billing_date": "10/14/2021", "medical_record_provider": {"name": "C00053 MD Now Medical Centers Inc", "Address": "2007 PALM BEACH LAKES BLVD, WEST PALM BEACH, FL 33409"}, "patient": {"name": "Robert Moore", "dob": "02/18/1981"}, "injury": {"date": "04/22/2021", "type": "car accident", "description": "Struck by another car from driver side while in passenger seat."}}>>>


    Your jsons may be much longer than the examples provided. Remember, we are looking for information that it most important for the other agents to help create solutions for our clients. Please format your output as a json at the end of your thought between <<< >>> with proper key value pairs."""

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(template),
        HumanMessagePromptTemplate.from_template("```{document_text}```"),
    ])

    chain = LLMChain(
        llm=OpenAI(temperature=0, openai_api_key=openai_api_key,
        model_name='gpt-3.5-turbo'),
        prompt=prompt
    )

    response = chain.run(document_text=document_text)
    try:
        response = json.loads(re.search(r'<<<(.+?)>>>', response).group(1)) # Get the text between <<< >>>
    except:
        response = {"status": "error"}

    return response

if __name__ == '__main__':
    main()