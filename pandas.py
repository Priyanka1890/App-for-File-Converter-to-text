from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
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
from langid.langid import LanguageIdentifier, model
import os, time
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

import pandas as pd
from langchain_openai import OpenAI

import os

# Set the OpenAI API key as an environment variable
os.environ["OPENAI_API_KEY"] = "your api key"

langchain.debug=True
langchain.verbose=True

def csv_to_pdf(csv_files, save_directory):
    
    # Initialize an empty list to hold individual DataFrames
    dataframes = []

    for i, csv_file in enumerate(csv_files):
        df = pd.read_csv(csv_file)
        
        # Append the DataFrame to the list
        dataframes.append(df)
        df.to_csv(os.path.join(save_directory, f'csv_{i}.csv'), index=False)
    
    # Concatenate the list of DataFrames into a single DataFrame
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Define a list of agent configurations
    # questions can be made on your choice and csv file
    agent_configs = [
        {"model": "gpt-3.5-turbo-0613", "question": "Make 10 questions from the CSV file"},
        {"model": "gpt-3.5-turbo-0613", "question": "How many males and females are there in the dataset?"},
        {"model": "gpt-3.5-turbo-0613", "question": "What is the maximum number of steps recorded by the Applewatch?"},
        {"model": "gpt-3.5-turbo-0613", "question": "Is there a correlation between Applewatch steps and distance covered?"}
        # Add more configurations as needed
    ]

    for i, agent_config in enumerate(agent_configs):
        # Create the agent
        agent = create_pandas_dataframe_agent(
            ChatOpenAI(temperature=0, model=agent_config["model"]),
            combined_df,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
        )
        time.sleep(5)

        # Get the start time
        start_time = time.time()

        # Run the agent
        output = agent.run(agent_config["question"])

        # Calculate the time taken by the agent
        time_taken = time.time() - start_time

        print(f"Agent {i + 1} took {time_taken} seconds to process.")

def main():
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    st.header("Ask your CSV 📈")
    st.text_input("Ask a question about your documents:", key="question_0")

    with st.sidebar:
        st.subheader("Upload Anything")
        csv_files = st.file_uploader("Upload a CSV file", type="csv", accept_multiple_files=True)
        if csv_files:
            save_directory = 'csv_files'
            os.makedirs(save_directory, exist_ok=True)
            if st.button("Process 1"):
                with st.spinner("Processing"):
                    # get pdf text
                    csv_to_pdf(csv_files, save_directory)
    
if __name__ == '__main__':
    main()

