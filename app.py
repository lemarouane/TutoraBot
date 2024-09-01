import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
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

# Initialize Streamlit
st.set_page_config(
    page_title="Outil QA Éducatif | TutoBot",
    page_icon="📚"
)

st.header('📚 TutoBot : Outil QA Éducatif')
st.subheader('Améliorez Votre Expérience d’Apprentissage')

st.success("Bienvenue sur TutoBot, un outil conçu pour offrir un soutien académique général et une analyse de documents.")

# Tabs for different functionalities
tab_general_chat, tab_document_analysis = st.tabs(["💬 Chat Général", "📄 Analyse de Document"])

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
