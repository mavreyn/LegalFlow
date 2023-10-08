'''
LegalFlow
October 08, 2023
- Maverick Reynolds
- Ian Ordonez
- James Trent
- Richard Brito 

LegalFlow is designed to connect clients with the associates of the legal firm by helping to inform clients of the services they provides as well as the next steps of the legal processes clients are involved with. LegalFlow also helps expedite administrative tasks of agents at Morgan & Morgan so that more time can be given to the associated to connect with their client.

Currenty, LegalFlow features a ChatBot designed to engage with clients who have questions regarding legal inquiries and processes for their case or general questions. LegalFlow also features a document analysis system that can classify documents into major categories based on their content. The document analysis system is designed to help Morgan & Morgan agents organize information as they continue to work with their clients.

With more time, LegalFlow will be able to provide clients with a more seamless and personalized experience with the firm's resources and associates. LegalFlow's Chatbot is available to connect with clients 24/7 and will be able to prepare preliminary documentation for clients to sign and send to the firm. We believe that incorporation of recent advancements in NLP, AI, and CV will help LegalFlow become a powerful tool for Morgan & Morgan to help clients connect with their associates and for associates to best serve the individuals and families in need.

---

App.py is the main file for the LegalFlow application. It uses dependencies from Langchain, Azure, and Streamlit to create a deployment for the Morgan & Morgan challenge at the 2023 Knight Hacks Hackathon.

Dependencies and technologies
- Langchain
- Azure AI Document Intelligence
- OpenAI GPT-3.5
- Streamlit

Github:
https://github.com/mavreyn/LegalFlow

Deployment:
http://legal-flow.streamlit.app

'''


import os, tempfile
import streamlit as st
from langchain.prompts import *
from langchain import LLMChain
from gtts import gTTS
from io import BytesIO

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
    if "accessibility_audio" not in st.session_state:
        st.session_state["accessibility_audio"] = False
    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "conversation" not in st.session_state:
        llm = OpenAI(temperature=0, openai_api_key=openai_api_key, model_name="gpt-3.5-turbo")
        st.session_state.conversation = ConversationChain(
            llm=llm,
            memory=ConversationSummaryMemory(llm=llm),
            prompt=PromptTemplate(input_variables=['history', 'input'], template="""You are 
LegalFlowAI, an AI agent at an injury law firm that is designed to help clients with any questions they may have regarding legal processes. You play a very important role in helping confort and inform our clients on questions they may have about documents, processes, and the like. 

It is important that you respond in a formal, polite, and comforting manner. Recognize that these individuals need all the resources they can get and you are here to help them. You are not a lawyer, but you are a legal assistant that can help them with any questions they may have. If they have questions that do not pertain to legality and legal process, please politefully decline to answer the question and redirect the conversation to legal topics. Please make sure to continue to interact with the client and keep asking questions to gain information; end every response with another question to the user. You can ask questions about the client's situation, the documents they have, and the like. Please make sure to keep the conversation going and make sure to ask questions that will help you understand the client's situation.
                                                          
Consider the following queries and responses between the triple backticks and how the conversation continues after each response:

```
CONVERSATION 1:

Individual: I was wondering if you knew the process for filing for a divorce?
LegalFlowAI: Of course, I would be happy to help you with that. The process for divorce depends on the state you are in. What state are you in?
Individual: I am in California.
LegalFlowAI: Great, I can help you with that. In California, you need to file a petition for divorce. You can find the form here: https://www.courts.ca.gov/documents/fl100.pdf. You will need to fill out the form and file it with the court. You will also need to serve the form to your spouse. You can find more information here: https://www.courts.ca.gov/1032.html. Are you experiencing servere domestic violence?
Individual: ...

CONVERSATION 2:
                                            
Individual: I am ready to proceed with a legal claim, What's the next step?
LegalFlowAI: I would be happy to help you with that. What kind of legal claim are you looking to proceed with?
Individual: ...
```
                                            
In addition to the above, here are some questions that clients have asked agents int he past. Be prepared to give thorough, genuine, and thoughtful conversation based on the following client texts between triple backticks:

```
- I was recently in a car accident and need an attorney
- I received my MRI results back and they were positive. What are the next steps?
- I'm experiencing pain and discomfort after the accident. What should I do?
- I have evidence of the other driver's negligence
- I have questions about the legal process for personal injury cases
- I need to understand the legal timeline for my case
- I'm experiencing emotional distress after the accident
```
                                  
Here is your conversation, continue it with the client:
```
{history}
                                  
{input}
```""")
            )


def on_click_callback():
    human_prompt = st.session_state["human_prompt"]
    llm_response = st.session_state.conversation.run(human_prompt)

    st.session_state["history"].append(
        Message("human", human_prompt)
    )
    st.session_state["history"].append(
        Message("llm", llm_response)
    )

    if st.session_state["accessibility_audio"]:
        st.sidebar.subheader('Audio Transcription')
        sound_file = BytesIO()
        tts = gTTS(llm_response, lang='en')
        tts.write_to_fp(sound_file)
        st.sidebar.audio(sound_file, format='audio/mp3')


def main():
    initialize_session_state()
    load_css()
    
    
    # Begin the Streamlit App Here
    st.markdown("<h1 style='text-align: center; color: #ffc107;'>LegalFlow</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Your assistant for document analysis & legal guidance.</h3>", unsafe_allow_html=True)

    # Do the sidebar here
    st.sidebar.title('Accessibility')
    use_accessibility = st.sidebar.checkbox('Enable Text to Speech Responses')
    if use_accessibility:
        st.session_state["accessibility_audio"] = True
    else:
        st.session_state["accessibility_audio"] = False

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
    else:
        st.warning('Please use the sidebar to upload a document for classification, or continue to interact with LegalFlowAI below')

    if file:
        st.markdown('---')
        doc_type = get_type_of_document(result.content)
        st.markdown(f"<h3 style='text-align: center;'>Document Type: {doc_type}</h3>", unsafe_allow_html=True)

    st.markdown('---')

    st.write('Hello, I am LegalFlowAI and I am here to help you with any questions you may have about your legal documents or legal procedures. Please use the chatbox below to ask any questions you may have.')
    st.write('')
    chat_palceholder = st.container()
    prompt_placeholder = st.form("Chat-form")

    with chat_palceholder:
        for chat in st.session_state.history:
            div = f"""
                <div class="chat-row {'llm' if chat.origin == 'llm' else 'human-bubble'}">{chat.message}</div>
            """
            st.markdown(div, unsafe_allow_html=True)

    with prompt_placeholder:
        st.text_input("LegalFlowAI  -  Begin a chat", value="", key='human_prompt')
        c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
        with c1:
            pass
        with c2:
            pass
        with c3:
            pass
        with c4:
            pass
        with c5:
            pass
        with c6:
            pass
        with c7:
            st.form_submit_button("Send", type="primary", on_click=on_click_callback)


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
    except:
        response = "".join([f'{x} ' for x in response.split()[-3:]])
    
    if response[1] == ' ':
        response = response[2:]
    
    return response


if __name__ == '__main__':
    main()
