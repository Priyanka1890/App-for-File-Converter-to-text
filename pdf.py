
import os
import faiss
import openai
import requests
import langchain
import numpy as np
import streamlit as st
import speech_recognition as sr


from urllib.parse import uses_query
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain_core.messages import AIMessage, HumanMessage
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from io import BytesIO
from gtts import gTTS


api_key = "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"
openai.api_key = api_key

langchain.debug=True
langchain.verbose=True




def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def text_to_audio(chat_history, lang='en'):
    sound_file = BytesIO()
    tts = gTTS(chat_history, lang='en')
    tts.write_to_fp(sound_file)
    st.audio(sound_file)


def identify_and_translate(text, source_lang, target_lang):
    #identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
    #detected_lang, _ = identifier.classify(text)
    #if detected_lang != source_lang:
    # Perform translation only if detected language is not the same as the source language
    translation_url = "https://translate.googleapis.com/translate_a/single?client=gtx&sl=" + source_lang + "&tl=" + target_lang + "&dt=t&q=" + text
    translation_response = requests.get(translation_url)
    translation_result = translation_response.text
    indexx = translation_result.index('","')
    translated_text = translation_result[4:int(indexx)]
    return translated_text
    #else:
        #return text
    
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def get_input_text(user_question, widget_num):
    if user_question.startswith("http"):
        # If the user question is a URL, fetch the text from the URL
        text = get_pdf_text(user_question)
    else:
        # If the user question is not a URL, use it as the text
        text = user_question

    return text

def handle_userinput(user_question, widget_num, source_lang, target_lang):
    if st.session_state.conversation is None:
        # If conversation is not initialized, create a new one
        vectorstore = get_vectorstore([])
        st.session_state.conversation = get_conversation_chain(vectorstore)
    
    text = get_input_text(user_question, widget_num)
    translated_text = identify_and_translate(text, source_lang, target_lang)
    
    response = st.session_state.conversation({'question': translated_text})
    st.session_state.chat_history = response['chat_history']

    # Display the response for the current question
    if st.session_state.chat_history:
        last_message = st.session_state.chat_history[-1]
        st.write(bot_template.replace("{{MSG}}", last_message.content), unsafe_allow_html=True)
        text_to_audio(last_message.content, lang=target_lang)

    # Create a new text input widget for the next question
    next_question = st.text_input(f"Ask a question about your documents:", key=f"question_{widget_num}")
    if next_question:
        handle_userinput(next_question, widget_num + 1, source_lang, target_lang)

def main():
    load_dotenv()
    
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    
    
    
    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:", key="question_0")
    
    source_lang = st.selectbox('Select source language', ('de', 'en', 'ru', 'es', 'hr_hr', 'fr'))
    target_lang = st.selectbox('Select target language', ('de', 'en', 'ru', 'es', 'hr_hr', 'fr'))
    
    if user_question:
        handle_userinput(user_question, 1, source_lang, target_lang)

    with st.sidebar:
        st.subheader("Upload Anything")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here", accept_multiple_files=True)
    

        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)

        




if __name__ == '__main__':
    main()

