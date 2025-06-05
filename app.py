import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get and validate API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found. Please create a .env file with your key.")
    st.stop()

def main():
    st.title("📄 Document Q&A with RAG")
    
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        loader = PyPDFLoader(tmp_file_path)
        pages = loader.load_and_split()
        os.unlink(tmp_file_path)  # Delete temp file
        
        if "prev_file" not in st.session_state or st.session_state.prev_file != uploaded_file.name:
            with st.spinner("Processing document..."):
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                )
                chunks = text_splitter.split_documents(pages)
                
                # Use the validated API key
                embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY,model="text-embedding-3-small")
                st.session_state.vector_store = FAISS.from_documents(chunks, embeddings)
                st.session_state.prev_file = uploaded_file.name
            
            st.success("Document processed! Ask questions below.")
    
    question = st.text_input("Ask a question about the document:")
    
    if question and st.session_state.vector_store:
        # Use the validated API key
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0,
            openai_api_key=OPENAI_API_KEY
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=st.session_state.vector_store.as_retriever(),
            return_source_documents=True
        )
        
        with st.spinner("Thinking..."):
            result = qa_chain.invoke({"query": question})
            st.subheader("Answer:")
            st.write(result["result"])
            
            with st.expander("See relevant sections"):
                for i, doc in enumerate(result["source_documents"]):
                    st.caption(f"Source {i+1}:")
                    st.text(doc.page_content[:500] + "...")
                    st.divider()

if __name__ == "__main__":
    main()