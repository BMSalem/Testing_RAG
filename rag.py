import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma 
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Chargement des variables d'environnement (.env)
load_dotenv()

# Configuration du modèle LLM (Modèle issu de ta capture d'écran)
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

def main():
    st.set_page_config(page_title="Assistant RAG Pro", layout="wide")
    st.header("📚 Chatbot PDF Intelligent")

    # --- INITIALISATION DU SESSION STATE ---
    # Pour que les données survivent aux interactions (mais pas au F5)
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # --- BARRE LATÉRALE : CHARGEMENT ET INDEXATION ---
    with st.sidebar:
        st.title("📂 Gestion des Documents")
        pdf_docs = st.file_uploader("Chargez vos PDFs", accept_multiple_files=True, type="pdf")
        
        if st.button("Indexer les documents"):
            if pdf_docs:
                with st.spinner("Analyse et vectorisation..."):
                    # 1. Nettoyage de l'ancien retriever pour éviter les conflits
                    st.session_state.retriever = None
                    
                    all_docs = []
                    for pdf in pdf_docs:
                        try:
                            reader = PdfReader(pdf)
                            for i, page in enumerate(reader.pages):
                                text = page.extract_text()
                                if text and text.strip():
                                    metadata = {"source": pdf.name, "page": i + 1}
                                    all_docs.append(Document(page_content=text, metadata=metadata))
                        except Exception as e:
                            st.error(f"Erreur sur {pdf.name}: {e}")

                    if all_docs:
                        # 2. Découpage du texte en segments (Chunks)
                        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                        chunks = splitter.split_documents(all_docs)

                        # 3. Création des Embeddings et du Vector Store (en RAM)
                        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                        
                        # On utilise une collection nommée pour garantir la fraîcheur des données
                        vector_store = Chroma.from_documents(
                            documents=chunks, 
                            embedding=embedding_model,
                            collection_name="current_session_docs"
                        )
                        
                        # 4. Sauvegarde du retriever dans le state
                        st.session_state.retriever = vector_store.as_retriever(search_kwargs={"k": 3})
                        st.success(f"Indexation réussie ! {len(all_docs)} pages prêtes.")
                    else:
                        st.error("Aucun texte n'a pu être extrait des fichiers fournis.")
            else:
                st.warning("Veuillez d'abord sélectionner des fichiers.")

        if st.button("Effacer la discussion"):
            st.session_state.messages = []
            st.rerun()

    # --- INTERFACE DE DISCUSSION ---
    
    # Affichage de l'historique
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("🔍 Sources consultées"):
                    st.info(message["sources"])

    # Zone de saisie utilisateur
    if prompt := st.chat_input("Posez votre question ici..."):
        # Ajout du message utilisateur
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Génération de la réponse
        if st.session_state.retriever:
            with st.chat_message("assistant"):
                with st.spinner("Recherche dans les documents..."):
                    # 1. Récupération du contexte
                    docs = st.session_state.retriever.invoke(prompt)
                    context_text = "\n\n".join([d.page_content for d in docs])
                    
                    # 2. Construction du prompt strict
                    system_prompt = f"""Réponds à la question en utilisant uniquement le contexte fourni ci-dessous.
                    Si la réponse n'est pas dans le contexte, réponds exactement : "Désolé, je ne trouve pas cette information dans les documents fournis."
                    
                    CONTEXTE:
                    {context_text}
                    
                    QUESTION:
                    {prompt}
                    """
                    
                    # 3. Appel au modèle Groq
                    response = llm.invoke(system_prompt)
                    answer = response.content
                    st.markdown(answer)

                    # 4. Gestion intelligente des sources
                    sources_to_save = None
                    # On affiche les sources seulement si le bot a trouvé une réponse
                    if "Désolé" not in answer:
                        source_list = []
                        for d in docs:
                            source_list.append(f"📄 {d.metadata['source']} (Page {d.metadata['page']})")
                        
                        sources_formatted = "\n".join(set(source_list))
                        sources_to_save = sources_formatted
                        with st.expander("🔍 Sources consultées"):
                            st.info(sources_formatted)

                    # 5. Sauvegarde dans l'historique
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": sources_to_save
                    })
        else:
            with st.chat_message("assistant"):
                st.error("Le moteur de recherche est vide. Merci de charger un PDF et de cliquer sur 'Indexer'.")

if __name__ == "__main__":
    main()
