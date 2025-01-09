

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain 
from langchain_community.chat_models import ChatOpenAI

#OPENAI_API_KEY="replace with open ai api key here"


#pdf upload section
st.header("My First Personal Chat bot")
with st.sidebar:
    st.title("Your Documents")
    file=st.file_uploader("Please Upload your PDF File and start asking questions",type="pdf")

#text Extraction
if file is not None:
    pdf_reader=PdfReader(file)
    text=""
    for page in pdf_reader.pages:
        text=text+page.extract_text()
        #st.write(text)
#split into chunks
    text_splitter=RecursiveCharacterTextSplitter(
    separators="\n",
    chunk_size=1000,
    chunk_overlap=150,
    length_function=len
    )

    chunks= text_splitter.split_text(text)
    #st.write(chunks)

#generating embeddings
    embeddings=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    #creating vector store - FAISS
    
    vector_store=FAISS.from_texts(chunks,embeddings)

    #user Question 
    user_question=st.text_input("Type your Questions here")

    #similarity search
    if user_question:
        match=vector_store.similarity_search(user_question)
        #st.write(match)

        llm=ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            temperature=0,
            max_tokens=1000,
            model_name="gpt-3.5-turbo"
        )
        #output results
        chain=load_qa_chain(llm,chain_type="stuff")
        res=chain.run(input_documents=match,question=user_question)
        st.write(res)

