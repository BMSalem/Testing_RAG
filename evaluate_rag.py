import os
from dotenv import load_dotenv
from datasets import Dataset
from PyPDF2 import PdfReader

# Imports LangChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq

# Imports RAGas
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerSimilarity, ContextPrecision, ContextRecall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# 1. Chargement de l'environnement
load_dotenv()

# =====================================================================
# CONFIGURATION DES DEUX LLM
# =====================================================================
# Le LLM de VOTRE RAG (Le modèle léger et rapide à tester)
llm_rag = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# Le LLM JUGE pour RAGas (Le modèle puissant indispensable pour noter)
llm_juge = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# On associe explicitement le LLM juge puissant aux métriques RAGas
metrics = [
    Faithfulness(),
    AnswerSimilarity(),
    ContextPrecision(),
    ContextRecall()
]

def load_local_pdfs(pdf_paths):
    """Lit les PDF locaux et extrait le texte comme dans votre app Streamlit"""
    all_docs = []
    for path in pdf_paths:
        try:
            reader = PdfReader(path)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    metadata = {"source": os.path.basename(path), "page": i + 1}
                    all_docs.append(Document(page_content=text, metadata=metadata))
        except Exception as e:
            print(f"Erreur lors de la lecture de {path}: {e}")
    return all_docs

def main():
    # --- CONFIGURATION DU TEST SET ---
    # ⚠️ METTEZ ICI LES VRAIS CHEMINS DE VOS FICHIERS PDF POUR LE TEST
    eval_pdfs = ["D:/Lab_DL/rag/pdfs/emsi.pdf", "D:/Lab_DL/rag/pdfs/LLMASJUDGE.pdf"]
    
    if not os.path.exists(eval_pdfs[0]):
        print(f"⚠️ Fichier introuvable : '{eval_pdfs[0]}'. Veuillez créer un dossier ou modifier les chemins vers de vrais PDF.")
        return

    print("📚 Extraction et indexation des documents de test (Votre RAG)...")
    all_docs = load_local_pdfs(eval_pdfs)
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(all_docs)
    
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(
        documents=chunks, 
        embedding=embedding_model,
        collection_name="ragas_eval_collection"
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # --- QUESTIONS DE RÉFÉRENCE (Vérité terrain) ---
    # ⚠️ ADAPTEZ CES QUESTIONS/RÉPONSES EN FONCTION DE VOS PDF
    test_questions = [
        {
            "question": "Qu'est-ce que le paradigme « LLM-as-a-Judge » ?",
            "ground_truth": "C'est l'utilisation de grands modèles de langage (LLM) comme évaluateurs automatiques pour juger des tâches complexes."
        },
        {
            "question": "Quel type de biais affecte souvent un LLM-juge lorsqu'il doit choisir entre deux réponses ?",
            "ground_truth": "Le biais de position (tendance à préférer la première réponse présentée) et le biais de verbosité (tendance à préférer les réponses plus longues)."
        },
        {
            "question": "Qu'est-ce que l'EMSI ?",
            "ground_truth": "C'est un réseau d'écoles d'ingénieurs privées au Maroc, reconnu par l'État."
        },
        {
            "question": "Dans quelles villes du Maroc trouve-t-Cons des campus de l'EMSI ?",
            "ground_truth": "À Casablanca, Rabat, Marrakech, Tanger et Fès."
        }
    ]

    # --- PIPELINE DE COLLECTE DES DONNÉES ---
    print("🤖 Exécution du RAG sur les questions de test...")
    
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for item in test_questions:
        query = item["question"]
        
        # 1. Recherche (Retriever de votre RAG)
        docs = retriever.invoke(query)
        context_text_list = [d.page_content for d in docs]
        context_text_unified = "\n\n".join(context_text_list) # <-- Corrigé ici
        
        # 2. Génération (Prompt + LLM de votre RAG)
        system_prompt = f"""Réponds à la question en utilisant uniquement le contexte fourni ci-dessous.
        Si la réponse n'est pas dans le contexte, réponds exactement : "Désolé, je ne trouve pas cette information dans les documents fournis."
        
        CONTEXTE:
        {context_text_unified}
        
        QUESTION:
        {query}"""
        
        response = llm_rag.invoke(system_prompt)
        
        # 3. Stockage pour RAGas
        questions.append(query)
        answers.append(response.content)
        contexts.append(context_text_list) 
        ground_truths.append(item["ground_truth"])

    # Préparation du Dataset au format attendu par RAGas
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(data)

    # --- ÉVALUATION RAGAS ---
    print("📊 Lancement de l'évaluation RAGas (LLM-as-a-Judge avec Llama-3.1-70b)...")

    ragas_llm_juge = LangchainLLMWrapper(llm_juge)
    ragas_embeddings = LangchainEmbeddingsWrapper(embedding_model)
    
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=ragas_llm_juge,      # Le juge pour analyser les textes (Llama 70B)
        embeddings=ragas_embeddings
    )
    
    print("\n=================================")
    print("🏆 RÉSULTATS DE VOTRE RAG (BASELINE)")
    print("=================================")
    print(result)

if __name__ == "__main__":
    main()