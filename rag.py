import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma 
from dotenv.ipython import load_dotenv
from langchain_groq import ChatGroq

load_dotenv(override=True)
from langchain_text_splitters import(
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

prompt_template = """
Réponds à la question suivante en utilisant seulement le contexte fourni:
<context>
    {context}
</context>
<question>
    {input}
</question>
"""

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

def main():
    st.set_page_config(page_title="RAG Demo", layout="wide")
    st.subheader("Retrieval Augmented Generation (RAG) Demo", divider="blue")

    with st.sidebar:
        st.sidebar.title("Data loader")
        pdf_docs = st.file_uploader(label="Load your pdfs", accept_multiple_files=True)
        if st.button("Submit"):
            with st.spinner("Loading"): #Affiche un spinner pendant le chargement
                content =""
                for pdf in pdf_docs:
                    reader = PdfReader(pdf)
                    for page in reader.pages:
                        content += page.extract_text()
                
                splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=512, chunk_overlap=50)

                chunks = splitter.split_text(content)
                st.write(chunks)

                embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2" )
                vector_store = Chroma.from_texts(chunks, embedding_model, collection_name="data_collection")
                retriever = vector_store.as_retriever(kwargs={"k": 5}) #Récupère les 5 documents les plus pertinents

                st.session_state.retriever = retriever #Stocke le retriever dans la session pour l'utiliser dans le chatbot
    st.subheader("Chatbot")
    user_question = st.text_input("Posez votre question ici")
    if user_question:
        context_docs = st.session_state.retriever.invoke(user_question)
        context_list = [doc.page_content for doc in context_docs]
        context_text = ". ".join(context_list)
        
        prompt = prompt_template.format(context=context_text, input=user_question)

        resp = llm.invoke(prompt)

        st.write(resp.content)

if __name__ == "__main__":
    main()


