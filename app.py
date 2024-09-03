import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from fpdf import FPDF
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.schema import HumanMessage
import requests
from bs4 import BeautifulSoup

# Load environment variables
load_dotenv()

# Get API key from environment variable
openai_key = os.getenv('OPENAI_API_KEY')

# Define the directory for pre-existing documents
PRE_EXISTING_DOCS_DIR = 'pre_existing_docs'
os.makedirs(PRE_EXISTING_DOCS_DIR, exist_ok=True)

# Simplified system prompt to focus purely on content reformulation
SYSTEM_PROMPT = """
You are TutoBot, an assistant that reformulates content while maintaining the original structure and information volume.
Please reformulate the following text, ensuring it remains detailed and accurate, without summarization or added content.
"""

# Initialize Streamlit
st.set_page_config(page_title="Outil de Regénération de Contenu | TutoBot", page_icon="📚")
st.header('📚 TutoBot : Outil de Regénération de Contenu')
st.subheader('Améliorez Votre Expérience d’Apprentissage')
st.success("Bienvenue sur TutoBot, un outil conçu pour regénérer le contenu des documents et des pages web sans modification du volume d'informations.")

# Tabs for different functionalities
tab_doc_generation, = st.tabs(["📝 Génération de Document"])

# Global state for document handling
document_state = {"documents": []}

def handle_document_upload():
    pdf_file = st.file_uploader("Téléchargez un document PDF", type=["pdf"])
    if pdf_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(pdf_file.read())
            temp_file.seek(0)
            loader = PyPDFLoader(temp_file.name)
            document_state["documents"] = loader.load()

    if not document_state["documents"]:
        for filename in os.listdir(PRE_EXISTING_DOCS_DIR):
            file_path = os.path.join(PRE_EXISTING_DOCS_DIR, filename)
            if os.path.isfile(file_path) and file_path.lower().endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                document_state["documents"].extend(loader.load())

def extract_text_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text(separator=' ', strip=True)
        return text
    except Exception as e:
        st.error(f"Erreur lors de la récupération du contenu à partir de l'URL: {e}")
        return None

def generate_content(user_prompt):
    system_message = HumanMessage(content=SYSTEM_PROMPT)
    user_message = HumanMessage(content=user_prompt)
    chat_llm = ChatOpenAI(api_key=openai_key)
    messages = [system_message, user_message]
    response = chat_llm(messages)
    return response.content

def create_pdf(content, output_path):
    pdf = FPDF()
    pdf.add_page()
    font_path_regular = os.path.join(os.path.dirname(__file__), 'DejaVuSans.ttf')
    font_path_bold = os.path.join(os.path.dirname(__file__), 'DejaVuSans-Bold.ttf')
    pdf.add_font("DejaVu", '', font_path_regular, uni=True)
    pdf.add_font("DejaVu", 'B', font_path_bold, uni=True)
    pdf.set_font("DejaVu", 'B', 16)
    pdf.ln(30)
    pdf.cell(0, 10, 'Génération de Document', ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("DejaVu", '', 12)
    pdf.multi_cell(0, 10, content)
    pdf.output(output_path)

with tab_doc_generation:
    st.write("Générez un document en français basé sur un document PDF existant, un lien URL, ou téléchargez le vôtre, puis téléchargez-le sous forme de PDF.")
    st.markdown("### Source du contenu")
    
    pdf_files = [f for f in os.listdir(PRE_EXISTING_DOCS_DIR) if f.endswith('.pdf')]
    uploaded_file = st.file_uploader("Téléchargez votre propre document PDF", type=["pdf"])

    if uploaded_file:
        uploaded_pdf_path = os.path.join(PRE_EXISTING_DOCS_DIR, uploaded_file.name)
        with open(uploaded_pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Le fichier '{uploaded_file.name}' a été téléchargé avec succès.")
        pdf_files.append(uploaded_file.name)

    url_input = st.text_input("Ou entrez un lien URL pour générer un contenu similaire:")

    if pdf_files or url_input:
        st.markdown("### Choisissez la source du contenu")
        content_source = st.selectbox("Sélectionnez la source:", ["PDF", "URL"])
        
        selected_pdf = None
        url_content = None
        
        if content_source == "PDF" and pdf_files:
            selected_pdf = st.selectbox("Choisissez un document PDF pour générer un contenu similaire:", pdf_files)
        
        if content_source == "URL" and url_input:
            url_content = extract_text_from_url(url_input)

        if st.button("Générer"):
            if content_source == "PDF" and selected_pdf:
                pdf_path = os.path.join(PRE_EXISTING_DOCS_DIR, selected_pdf)
                loader = PyPDFLoader(pdf_path)
                pdf_documents = loader.load()
                pdf_content = " ".join([doc.page_content for doc in pdf_documents])
                user_prompt = pdf_content
            elif content_source == "URL" and url_content:
                user_prompt = url_content

            if user_prompt:
                with st.spinner("Génération du contenu..."):
                    generated_content = generate_content(user_prompt)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                    create_pdf(generated_content, temp_pdf.name)
                    pdf_file_path = temp_pdf.name
                st.success("Document PDF généré avec succès.")
                st.download_button(
                    label="Télécharger le PDF",
                    data=open(pdf_file_path, "rb"),
                    file_name="document_generé.pdf",
                    mime="application/pdf"
                )
            else:
                st.error("Veuillez sélectionner un document PDF ou fournir un lien URL valide.")
    else:
        st.warning("Aucun document PDF disponible dans le répertoire pré-existant.")
