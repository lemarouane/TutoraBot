import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import DocArrayInMemorySearch

# Load environment variables
load_dotenv()

# Get API key from environment variable
openai_key = os.getenv('OPENAI_API_KEY')

# Define the directory for pre-existing documents
PRE_EXISTING_DOCS_DIR = 'pre_existing_docs'

# Ensure the directory exists
os.makedirs(PRE_EXISTING_DOCS_DIR, exist_ok=True)

# Initialize Streamlit
st.set_page_config(
    page_title="Outil QA Éducatif | TutoBot",
    page_icon="📚"
)

st.header('📚 TutoBot : Outil QA Éducatif')

st.subheader('Améliorez Votre Expérience d’Apprentissage')

st.success("Bienvenue sur TutoBot, un outil conçu pour aider les étudiants avec leurs essais, leurs travaux de recherche et leurs documents éducatifs.")

st.write('''
Avec TutoBot, vous pouvez interagir avec votre contenu personnalisé pour obtenir des informations et du soutien pour vos besoins académiques.
Imaginez avoir un assistant virtuel qui vous aide à :

- Résumer des documents et des travaux de recherche complexes.
- Extraire les points clés de textes longs.
- Identifier et clarifier toute section confuse ou contradictoire.
- Obtenir des réponses à des questions spécifiques liées à votre contenu.
''')

pdf_file = st.file_uploader("Téléchargez un document PDF", type=["pdf"])

if pdf_file is not None:
    # Handle the uploaded document
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(pdf_file.read())
        temp_file.seek(0)
        loader = PyPDFLoader(temp_file.name)
        documents = loader.load()
else:
    # Load documents from the pre-existing directory
    documents = []
    for filename in os.listdir(PRE_EXISTING_DOCS_DIR):
        file_path = os.path.join(PRE_EXISTING_DOCS_DIR, filename)
        if os.path.isfile(file_path) and file_path.lower().endswith('.pdf'):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())

if documents:
    # Initialize LangChain components
    embeddings = OpenAIEmbeddings(api_key=openai_key)
    vector_store = DocArrayInMemorySearch.from_documents(documents, embeddings)
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(api_key=openai_key),
        chain_type="map_reduce",
        retriever=vector_store.as_retriever()
    )

    # Input box for user queries
    query = st.text_input("Posez votre question :")

    if query:
        response = qa_chain.run(query)
        st.write("### Réponse du Chatbot")
        st.write(response)
else:
    st.write("Aucun document à traiter. Veuillez télécharger un document ou assurez-vous qu'il y a des documents dans le répertoire pré-existant.")
