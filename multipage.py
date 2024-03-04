import streamlit as st



st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# App for File Converter to text ! 👋")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
    "Welcome to our Streamlit app, where you can easily convert your PDF, audio, and CSV files into text! Our app makes it simple to extract valuable information from your files, whether you need to analyze data, read documents, or transcribe audio recordings. Just upload your files and let our app do the rest. Try it out now and see how easy it is to transform your files into text!"
    ### Want to know more?
    - Go to sidebar
    - Select one service you want to convert 
    - Upload file from your system
    - Ask a question from your file
    - Get answers in different language
    - Chat with your uploaded file
    - Enjoy with answers
   
"""
)



