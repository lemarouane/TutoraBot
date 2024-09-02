import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from fpdf import FPDF  # Import FPDF for PDF generation
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.schema import HumanMessage  # Import the correct schema

# Load environment variables
load_dotenv()

# Get API key from environment variable
openai_key = os.getenv('OPENAI_API_KEY')

# Define the directory for pre-existing documents
PRE_EXISTING_DOCS_DIR = 'pre_existing_docs'

# Ensure the directory exists
os.makedirs(PRE_EXISTING_DOCS_DIR, exist_ok=True)

# Define the system prompt
SYSTEM_PROMPT = """
You are TutoBot, an educational assistant specialized in generating academic materials such as tests, quizzes, and study guides in French. When a user requests to generate a document, follow these guidelines:

1. **Test Generation**:
   - Include multiple-choice questions, true/false statements, and short answer questions.
   - Cover a range of difficulty levels.
   - Provide an answer key at the end of the document.

2. **Quiz Creation**:
   - Focus on specific topics as requested by the user.
   - Include a mix of question types.
   - Ensure clarity and conciseness in questions.
   - Focus on giving a text story and based on this text give questions.
   - Include a mix of question types.
   - Do not give any answers .. let the users do the tests by themselves.   

3. **Study Guides**:
   - Summarize key concepts and topics.
   - Include bullet points for easy readability.
   - Highlight important dates, formulas, or definitions as necessary.

Always ensure the content is well-structured, grammatically correct, and suitable for educational purposes.
"""

# Initialize Streamlit
st.set_page_config(
    page_title="Outil QA Éducatif | TutoBot",
    page_icon="📚"
)

st.header('📚 TutoBot : Outil QA Éducatif')
st.subheader('Améliorez Votre Expérience d’Apprentissage')

st.success("Bienvenue sur TutoBot, un outil conçu pour offrir un soutien académique général et une analyse de documents.")

# Tabs for different functionalities
tab_general_chat, tab_document_analysis, tab_doc_generation = st.tabs(["💬 Chat Général", "📄 Analyse de Document", "📝 Génération de Document"])

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
        # Load pre-existing documents
        for filename in os.listdir(PRE_EXISTING_DOCS_DIR):
            file_path = os.path.join(PRE_EXISTING_DOCS_DIR, filename)
            if os.path.isfile(file_path) and file_path.lower().endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                document_state["documents"].extend(loader.load())

# Define a function to generate the content using OpenAI's API with system prompt
def generate_content(user_prompt):
    # Define the system message
    system_message = HumanMessage(content=SYSTEM_PROMPT)
    
    # Define the user message
    user_message = HumanMessage(content=user_prompt)
    
    # Initialize the ChatOpenAI model with the system and user messages
    chat_llm = ChatOpenAI(api_key=openai_key)
    
    # Combine system and user messages
    messages = [system_message, user_message]
    
    # Generate the response
    response = chat_llm(messages)
    
    return response.content

def create_pdf(content, output_path):
    pdf = FPDF()
    pdf.add_page()

    # Path to your font files
    font_path_regular = os.path.join(os.path.dirname(__file__), 'DejaVuSans.ttf')
    font_path_bold = os.path.join(os.path.dirname(__file__), 'DejaVuSans-Bold.ttf')

    # Add regular and bold fonts
    pdf.add_font("DejaVu", '', font_path_regular, uni=True)
    pdf.add_font("DejaVu", 'B', font_path_bold, uni=True)

    # Add the header image
    
    # Add a custom title under the header image
    pdf.set_font("DejaVu", 'B', 16)  # Bold font, size 16
    pdf.ln(30)  # Add some vertical space after the image
    pdf.cell(0, 10, 'Génération de Document', ln=True, align='C')  # Centered title
    
    # Add some space before the content
    pdf.ln(10)
    
    # Set the main content font
    pdf.set_font("DejaVu", '', 12)  # Regular font, size 12
    
    # Add the content
    pdf.multi_cell(0, 10, content)
    
    # Output the PDF to the specified path
    pdf.output(output_path)

# Define a custom CSS style and load FontAwesome
st.markdown("""
<style>
.chat-box {
    max-height: 400px;
    overflow-y: auto;
    padding: 10px;
    border: 1px solid #ccc;
    border-radius: 10px;
    background-color: #f9f9f9;
}

.user-message, .bot-message {
    display: flex;
    align-items: center;
    margin-bottom: 10px;
}

.user-message {
    justify-content: flex-end;
}

.bot-message {
    justify-content: flex-start;
}

.user-message .message-content, .bot-message .message-content {
    max-width: 60%;
    padding: 10px;
    border-radius: 10px;
}

.user-message .message-content {
    background-color: #e0f7fa;
    text-align: right;
}

.bot-message .message-content {
    background-color: #e8eaf6;
    text-align: left;
}

.user-icon, .bot-icon {
    width: 30px;
    height: 30px;
    margin: 0 10px;
    font-size: 30px;
}

.user-icon {
    order: 2;
}

.message-container {
    display: flex;
    flex-direction: column;
}

.fa-robot {
    color: #6c757d;
}
</style>

<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
""", unsafe_allow_html=True)

# General Chat Tab
with tab_general_chat:
    st.write("Dans cet onglet, vous pouvez discuter librement avec TutoBot pour obtenir du soutien académique ou des conseils.")
    
    if 'general_conversation' not in st.session_state:
        st.session_state['general_conversation'] = []

    general_query = st.text_input("Entrez votre question ou message ici :")
    
    if general_query:
        chat_llm = ChatOpenAI(api_key=openai_key)
        general_response = chat_llm([HumanMessage(content=general_query)])
        
        st.session_state['general_conversation'].append(('user', general_query))
        st.session_state['general_conversation'].append(('bot', general_response.content))
    
    # Display the conversation in a box with a scrollbar
    with st.container():
        st.markdown("<div class='chat-box message-container'>", unsafe_allow_html=True)
        
        for message_type, message_content in st.session_state['general_conversation']:
            if message_type == 'user':
                st.markdown(f"""
                <div class="user-message">
                    <div class="message-content">{message_content}</div>
                    <img src="https://cdn-icons-png.flaticon.com/512/3135/3135715.png" alt="User" class="user-icon"/>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="bot-message">
                    <i class="fas fa-robot bot-icon"></i>
                    <div class="message-content">{message_content}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

# Document Analysis Tab
with tab_document_analysis:
    st.write("Téléchargez un document PDF pour obtenir des réponses spécifiques à son contenu.")
    
    handle_document_upload()

    if document_state["documents"]:
        embeddings = OpenAIEmbeddings(api_key=openai_key)
        vector_store = DocArrayInMemorySearch.from_documents(document_state["documents"], embeddings)
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(api_key=openai_key),
            chain_type="map_reduce",
            retriever=vector_store.as_retriever()
        )
        
        document_query = st.text_input("Posez une question sur votre document :")
        
        if document_query:
            document_response = qa_chain.run(document_query)
            
            if 'document_conversation' not in st.session_state:
                st.session_state['document_conversation'] = []

            st.session_state['document_conversation'].append(('user', document_query))
            st.session_state['document_conversation'].append(('bot', document_response))

            # Display the conversation in a box with a scrollbar
            with st.container():
                st.markdown("<div class='chat-box message-container'>", unsafe_allow_html=True)
                
                for message_type, message_content in st.session_state['document_conversation']:
                    if message_type == 'user':
                        st.markdown(f"""
                        <div class="user-message">
                            <div class="message-content">{message_content}</div>
                            <img src="https://cdn-icons-png.flaticon.com/512/3135/3135715.png" alt="User" class="user-icon"/>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="bot-message">
                            <i class="fas fa-robot bot-icon"></i>
                            <div class="message-content">{message_content}</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.write("Aucun document à traiter. Veuillez télécharger un document ou vérifiez les documents dans le répertoire pré-existant.")

# Document Generation Tab
with tab_doc_generation:
    st.write("Générez un test ou un quiz en français basé sur un document PDF existant ou téléchargez le vôtre, puis téléchargez-le sous forme de PDF.")
    
    # List PDFs in the pre-existing documents directory
    pdf_files = [f for f in os.listdir(PRE_EXISTING_DOCS_DIR) if f.endswith('.pdf')]

    # Option for user to upload their own PDF
    uploaded_file = st.file_uploader("Ou téléchargez votre propre document PDF", type=["pdf"])
    
    if uploaded_file:
        # Save the uploaded file in the pre-existing documents directory
        uploaded_pdf_path = os.path.join(PRE_EXISTING_DOCS_DIR, uploaded_file.name)
        with open(uploaded_pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"Le fichier '{uploaded_file.name}' a été téléchargé avec succès.")
        
        # Refresh the list of PDFs to include the newly uploaded file
        pdf_files.append(uploaded_file.name)

    if pdf_files:
        selected_pdf = st.selectbox("Choisissez un document PDF pour générer un contenu similaire:", pdf_files)
        
        if st.button("Générer"):
            if selected_pdf:
                # Load the selected PDF and extract its content
                pdf_path = os.path.join(PRE_EXISTING_DOCS_DIR, selected_pdf)
                loader = PyPDFLoader(pdf_path)
                pdf_documents = loader.load()

                # Combine the extracted content into a single string
                pdf_content = " ".join([doc.page_content for doc in pdf_documents])

                # Enhanced prompt to generate similar content with specific instructions
                user_prompt = (
                    f"Vous avez sélectionné le document intitulé '{selected_pdf}'. "
                    "Veuillez générer un document similaire en conservant la structure générale et les idées principales, "
                    "mais en reformulant le texte pour qu'il soit unique et amélioré. Assurez-vous que le contenu soit bien organisé, "
                    "clair et adapté à un public académique. Veuillez inclure des éléments similaires tels que les questions, "
                    "les guides d'étude ou tout autre format pertinent, tout en reformulant les phrases pour éviter le plagiat."
                    "\n\nContenu du document d'origine :\n"
                    f"{pdf_content}"
                )
                
                with st.spinner("Génération du contenu..."):
                    generated_content = generate_content(user_prompt)
                
                # Create PDF from generated content
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
                st.error("Veuillez sélectionner un document PDF.")
    else:
        st.warning("Aucun document PDF disponible dans le répertoire pré-existant.")
