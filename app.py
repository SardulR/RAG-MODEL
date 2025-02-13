
import streamlit as st
import os
import tempfile
from multimodal_rag import MultimodalRAG
from evaluation import generate_evaluation_report
import logging
import requests
from urllib.parse import urlparse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the RAG system
@st.cache_resource
def initialize_rag():
    try:
        rag = MultimodalRAG()
        return rag
    except Exception as e:
        logger.error(f"Error initializing RAG: {e}")
        raise

def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def download_from_url(url, file_type):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            suffix = ".mp4" if file_type == "video" else ".mp3"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(response.content)
                return tmp.name
        return None
    except Exception as e:
        logger.error(f"Error downloading from URL: {e}")
        return None

# Page configuration
st.set_page_config(
    page_title="Multimodal RAG System",
    page_icon="🤖",
    layout="wide"
)

# Main title
st.title("🤖 Enhanced Multimodal RAG System")
st.write("Upload documents, media files, or provide URLs to extract information and ask questions.")

# Initialize RAG system
try:
    rag = initialize_rag()
    st.success("✅ RAG system initialized successfully!")
except Exception as e:
    st.error(f"Error initializing RAG system: {e}")
    st.stop()

# Sidebar for document and media upload
with st.sidebar:
    st.header("📄 Content Input")

    # Create tabs for different input methods
    input_tabs = st.tabs(["Documents", "Video", "Audio"])

    # Document Upload Tab
    with input_tabs[0]:
        uploaded_files = st.file_uploader(
            "Upload your documents",
            type=["txt", "pdf", "jpg", "jpeg", "png"],
            accept_multiple_files=True
        )

    # Video Input Tab
    with input_tabs[1]:
        video_option = st.radio("Choose video input method:", ["Upload File", "Provide URL"])

        if video_option == "Upload File":
            video_file = st.file_uploader("Upload video file", type=["mp4", "avi", "mov"])
            if video_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    tmp.write(video_file.getvalue())
                    st.session_state.video_path = tmp.name
        else:
            video_url = st.text_input("Enter video URL")
            if video_url and is_valid_url(video_url):
                if st.button("Extract Video Data"):
                    with st.spinner("Downloading video..."):
                        video_path = download_from_url(video_url, "video")
                        if video_path:
                            st.session_state.video_path = video_path
                            st.success("Video downloaded successfully!")
                        else:
                            st.error("Failed to download video")

    # Audio Input Tab
    with input_tabs[2]:
        audio_option = st.radio("Choose audio input method:", ["Upload File", "Provide URL"])

        if audio_option == "Upload File":
            audio_file = st.file_uploader("Upload audio file", type=["mp3", "wav", "m4a"])
            if audio_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                    tmp.write(audio_file.getvalue())
                    st.session_state.audio_path = tmp.name
        else:
            audio_url = st.text_input("Enter audio URL")
            if audio_url and is_valid_url(audio_url):
                if st.button("Extract Audio Data"):
                    with st.spinner("Downloading audio..."):
                        audio_path = download_from_url(audio_url, "audio")
                        if audio_path:
                            st.session_state.audio_path = audio_path
                            st.success("Audio downloaded successfully!")
                        else:
                            st.error("Failed to download audio")

    # Process Button for all content
    if st.button("Process All Content"):
        try:
            with st.spinner("Processing content..."):
                files_to_process = []

                # Add document files
                if uploaded_files:
                    temp_files = []
                    for uploaded_file in uploaded_files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                            tmp.write(uploaded_file.getvalue())
                            temp_files.append(tmp.name)
                    files_to_process.extend(temp_files)

                # Add video file if exists
                if hasattr(st.session_state, 'video_path'):
                    files_to_process.append(st.session_state.video_path)

                # Add audio file if exists
                if hasattr(st.session_state, 'audio_path'):
                    files_to_process.append(st.session_state.audio_path)

                if files_to_process:
                    # Process all files
                    rag.process_documents(files_to_process)

                    # Clean up temporary files
                    for temp_file in files_to_process:
                        try:
                            os.unlink(temp_file)
                        except:
                            pass

                    # Clear session state
                    if hasattr(st.session_state, 'video_path'):
                        del st.session_state.video_path
                    if hasattr(st.session_state, 'audio_path'):
                        del st.session_state.audio_path

                    st.success("✅ All content processed successfully!")
                else:
                    st.warning("No content to process. Please upload files or provide URLs first.")

        except Exception as e:
            st.error(f"Error processing content: {e}")

# Main area for querying
st.header("🔍 Ask Questions")
query = st.text_input("Enter your question:")

if query:
    if st.button("Submit Query"):
        try:
            with st.spinner("Processing query..."):
                # Get response from RAG system
                response = rag.query(query)

                # Display answer
                st.header("📝 Answer")
                st.write(response["answer"])

                # Display sources
                st.header("📚 Sources")
                for i, source in enumerate(response.get("sources", []), 1):
                    with st.expander(f"Source {i} (Score: {source['score']:.2f})"):
                        st.write(source["content"])
                        st.json(source["metadata"])

        except Exception as e:
            st.error(f"Error processing query: {e}")
            logger.error(f"Error processing query: {e}", exc_info=True)

# Footer
st.markdown("---")
st.markdown("### How to use")
st.markdown("""
1. Upload your content using the sidebar:
   - Documents (PDF, TXT, Images)
   - Video files or URLs
   - Audio files or URLs
2. Click 'Process All Content' to analyze everything
3. Ask questions about the processed content
4. View the AI-generated answers with source attribution
""")
