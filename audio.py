

	
from urllib.parse import uses_query
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import langchain
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

import os
import openai
import faiss
import numpy as np
from gtts import gTTS
from io import BytesIO
import streamlit as st
import requests
from translate import Translator
from pydub import AudioSegment
from pydub.playback import play

api_key = "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"
openai.api_key = api_key

langchain.debug=True
langchain.verbose=True

urls = [
    ]


def fetch_text_from_url(urls):
    loaders = UnstructuredURLLoader(urls=urls)
    data = loaders.load()

    return data


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
    
    '''
    translation_url = "https://translate.googleapis.com/translate_a/single?client=gtx&sl=" + source_lang + "&tl=" + target_lang + "&dt=t&q=" + text
    translation_response = requests.get(translation_url)
    translation_result = translation_response.text
    indexx = translation_result.index('","')
    translated_text = translation_result[4:int(indexx)]
    return translated_text
    '''

    # Perform translation only if detected language is not the same as the source language
    if source_lang.lower() != target_lang.lower():
        translator = Translator(from_lang=source_lang, to_lang=target_lang)
        translated_text = translator.translate(text)
        return translated_text
    else:
        return text
    
def save_uploaded_file(uploaded_file):
    # Create a temporary directory to save the uploaded file
    temp_dir = "temp_audio_files"
    os.makedirs(temp_dir, exist_ok=True)

    # Save the uploaded file to the temporary directory
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path

def transcribe_audio(input_audio):
    # Load the audio file
    
    audio_file_path = input_audio
    audio = AudioSegment.from_file(audio_file_path)

    # Play the audio
    play(audio)

    audio_file = open(audio_file_path, "rb")

    transcription = openai.audio.transcriptions.create(
    model="whisper-1", 
    file=audio_file, 
    response_format="text"
    )
    return transcription

def display_audio_transcript(transcript):
    st.subheader("Audio Transcript:")
    st.write(transcript)


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
        text = fetch_text_from_url(user_question)
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
    st.set_page_config(page_title="Chat with multiple inputs",
                       page_icon="🤖")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:", key="question_0")
    
    source_lang = st.selectbox('Select source language', ('en','de', 'ru', 'es','fr'))
    target_lang = st.selectbox('Select target language', ('en','de', 'ru', 'es', 'fr'))
    
    if user_question:
        handle_userinput(user_question, 1, source_lang, target_lang)

    with st.sidebar:
        st.subheader("Upload Anything")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here", accept_multiple_files=True)
        
        if st.button("Process 1"):
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
                
        st.subheader("Upload Website Link here")
        website_url = st.text_input("Website URL")

        # Audio Processing
        st.subheader("Audio Processing")
        audio_file = st.file_uploader("Upload your audio file here", type=["mp3", "wav", "ogg"])
        print(audio_file)
    
        
        if st.button("Process 2"):
            with st.spinner("Processing"):
                if audio_file:
                    st.audio(audio_file)
                    # Save the uploaded file and get the file path
                    file_path = save_uploaded_file(audio_file)
                    audio_text = transcribe_audio(file_path)
                    display_audio_transcript(audio_text)
        




if __name__ == '__main__':
    main()
