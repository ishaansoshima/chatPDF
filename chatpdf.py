import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract text from uploaded PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create and save FAISS vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    
    # Ensure the directory exists
    if not os.path.exists("faiss_index"):
        os.makedirs("faiss_index")
    
    # Save the FAISS index
    vector_store.save_local("faiss_index")

# Function to create the conversational chain for question answering
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, just say, "The answer is not available in the context." Do not provide a wrong answer.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    
    # Initialize the Google Generative AI model
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    # Set up the QA chain with the prompt
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

# Function to handle user input and run the QA process
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Check if the FAISS index exists before attempting to load it
    if not os.path.exists("faiss_index/index.faiss"):
        st.error("FAISS index not found. Please upload and process PDF files first.")
        return

    # Load the FAISS index and allow dangerous deserialization
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    # Perform similarity search
    docs = new_db.similarity_search(user_question)

    # Get the QA chain
    chain = get_conversational_chain()

    # Get response from the chain
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    # Display the response
    st.write("Reply: ", response["output_text"])

# Main function to define the Streamlit app interface
def main():
    st.set_page_config("Chat with PDF using RAG")
    st.header("Chat with PDF using RAG")

    # Get user question input
    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    # Sidebar for PDF file upload
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)

        # Process PDFs and create FAISS index when button is clicked
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Processing complete. You can now ask questions.")
            else:
                st.error("Please upload PDF files before processing.")

if __name__ == "__main__":
    main()
