"""
Main Streamlit Application
Entry point for the PDF Summarizer UI
"""

import streamlit as st
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_integration.summarizer import LLMSummarizer
from database.models import get_db_manager
from utils.config import get_config
from utils.logger import setup_logger

logger = logging.getLogger(__name__)


class PDFSummarizerApp:
    """Main Streamlit application class."""

    def __init__(self, config):
        """Initialize the application with configuration."""
        self.config = config
        self.logger = setup_logger()
        self.db_manager = None
        self.summarizer = None

    def setup_page_config(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title=self.config.streamlit.page_title,
            page_icon=self.config.streamlit.page_icon,
            layout=self.config.streamlit.layout,
            initial_sidebar_state="expanded",
        )

    def setup_session_state(self):
        """Initialize session state variables"""
        if "current_summary" not in st.session_state:
            st.session_state.current_summary = None
        if "processing" not in st.session_state:
            st.session_state.processing = False
        if "uploaded_file" not in st.session_state:
            st.session_state.uploaded_file = None
        if "selected_page" not in st.session_state:
            st.session_state.selected_page = "🏠 Homepage"

    def initialize_components(self):
        """Initialize database and LLM components"""
        try:
            # Initialize database
            self.db_manager = get_db_manager()

            # Initialize LLM summarizer
            if self.config.groq.api_key:
                self.summarizer = LLMSummarizer(self.config)
            else:
                st.error("⚠️ Groq API key not found. Please set GROQ_API_KEY in your .env file.")
                st.info("1. Copy `.env.example` to `.env`")
                st.info("2. Add your Groq API key to the `.env` file")
                st.info("3. Restart the application")
                return False

            return True

        except Exception as e:
            st.error(f"❌ Failed to initialize components: {str(e)}")
            self.logger.error(f"Component initialization error: {e}")
            return False

    def run(self):
        """Run the main Streamlit application"""
        self.setup_page_config()
        self.setup_session_state()

        # Initialize components
        if not self.initialize_components():
            return

        # Main title
        st.title("📄 PDF Summarizer AI")
        st.markdown("Upload a PDF file to generate an AI-powered summary using Groq's LLM.")

        # Sidebar for navigation
        page = st.sidebar.selectbox(
            "Choose a page",
            ["🏠 Homepage", "📤 Upload & Summarize", "📚 Document History", "⚙️ Settings", "ℹ️ About"],
            index=["🏠 Homepage", "📤 Upload & Summarize", "📚 Document History", "⚙️ Settings", "ℹ️ About"].index(st.session_state.selected_page)
        )

        # Update selected page in session state
        st.session_state.selected_page = page

        if page == "🏠 Homepage":
            self.show_homepage()
        elif page == "📤 Upload & Summarize":
            self.show_upload_page()
        elif page == "📚 Document History":
            self.show_history_page()
        elif page == "⚙️ Settings":
            self.show_settings_page()
        elif page == "ℹ️ About":
            self.show_about_page()

    def show_homepage(self):
        """Show the homepage with README content"""
        st.header("🏠 Welcome to PDF Summarizer AI")

        # Load and display README content
        try:
            readme_path = Path("README.md")
            if readme_path.exists():
                with open(readme_path, "r", encoding="utf-8") as f:
                    readme_content = f.read()

                # Add custom styling for better display
                st.markdown("""
                <style>
                .readme-container {
                    background-color: #f8f9fa;
                    padding: 20px;
                    border-radius: 10px;
                    border: 1px solid #e1e5e9;
                    margin: 10px 0;
                }
                .readme-container h1 {
                    color: #1f77b4;
                    border-bottom: 2px solid #1f77b4;
                    padding-bottom: 10px;
                }
                .readme-container h2 {
                    color: #2ca02c;
                    border-bottom: 1px solid #2ca02c;
                    padding-bottom: 5px;
                }
                .readme-container h3 {
                    color: #ff7f0e;
                }
                .readme-container code {
                    background-color: #f1f1f1;
                    padding: 2px 4px;
                    border-radius: 3px;
                    font-family: 'Courier New', monospace;
                }
                .readme-container pre {
                    background-color: #f1f1f1;
                    padding: 15px;
                    border-radius: 5px;
                    overflow-x: auto;
                    border-left: 4px solid #1f77b4;
                }
                </style>
                """, unsafe_allow_html=True)

                # Display README content with custom styling
                st.markdown(f"""
                <div class="readme-container">
                {readme_content}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("📄 README.md file not found.")
                st.info("📋 Please ensure README.md exists in the project root directory.")

        except Exception as e:
            st.error(f"❌ Error reading README.md: {str(e)}")
            self.logger.error(f"Error reading README.md: {e}")

        # Add quick action buttons
        st.divider()
        st.subheader("🚀 Quick Actions")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📤 Upload PDF", key="homepage_upload", use_container_width=True):
                st.info("👆 Select '📤 Upload & Summarize' from the sidebar to get started")

        with col2:
            if st.button("📚 View History", key="homepage_history", use_container_width=True):
                st.info("👆 Select '📚 Document History' from the sidebar to view your documents")

        with col3:
            if st.button("⚙️ Settings", key="homepage_settings", use_container_width=True):
                st.info("👆 Select '⚙️ Settings' from the sidebar to configure the application")

        # Additional information
        st.divider()
        st.subheader("✨ Key Features")

        feature_col1, feature_col2 = st.columns(2)

        with feature_col1:
            st.markdown("""
            - 📄 **PDF Upload & Processing**
            - 🤖 **AI-Powered Summarization** (Groq Llama 3.3 70B)
            - 💾 **SQLite Database Storage**
            - 📚 **Document History Tracking**
            """)

        with feature_col2:
            st.markdown("""
            - 📤 **Export to PDF/Text**
            - 🔄 **Duplicate File Handling**
            - ⚡ **Fast Processing**
            - 🎯 **Multiple Summary Types**
            """)

        # System status
        st.divider()
        st.subheader("🔧 System Status")

        status_col1, status_col2, status_col3 = st.columns(3)

        with status_col1:
            st.metric("Database", "✅ Connected" if self.db_manager else "❌ Error")

        with status_col2:
            st.metric("LLM Service", "✅ Ready" if self.summarizer else "❌ Error")

        with status_col3:
            st.metric("API Key", "✅ Configured" if self.config.groq.api_key else "❌ Missing")

        # Database statistics
        if self.db_manager:
            try:
                documents = self.db_manager.get_all_documents()
                summaries = self.db_manager.get_all_summaries()

                st.divider()
                st.subheader("📊 Usage Statistics")

                stats_col1, stats_col2, stats_col3 = st.columns(3)

                with stats_col1:
                    st.metric("📄 Documents", len(documents))

                with stats_col2:
                    st.metric("📝 Summaries", len(summaries))

                with stats_col3:
                    if documents:
                        total_size = sum(doc.file_size for doc in documents)
                        st.metric("💾 Total Size", f"{total_size / (1024*1024):.1f} MB")
                    else:
                        st.metric("💾 Total Size", "0 MB")

            except Exception as e:
                self.logger.error(f"Error loading statistics: {e}")
                # Don't show error to user, just skip statistics

    def show_upload_page(self):
        """Show the upload and summarization page"""
        st.header("📤 Upload & Summarize")

        # File upload section
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload a PDF file to generate a summary"
        )

        if uploaded_file is not None:
            st.session_state.uploaded_file = uploaded_file

            # Display file information
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("File Name", uploaded_file.name)
            with col2:
                st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
            with col3:
                st.metric("File Type", uploaded_file.type)

            # Summary options
            st.subheader("🎯 Summary Options")

            col1, col2 = st.columns(2)
            with col1:
                summary_type = st.selectbox(
                    "Summary Type",
                    ["concise", "detailed", "bullet_points"],
                    help="Choose the type of summary you want"
                )

            with col2:
                temperature = st.slider(
                    "Creativity",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.3,
                    step=0.1,
                    help="Higher values make the summary more creative"
                )

            # Generate summary button
            if st.button("🚀 Generate Summary", type="primary"):
                if not st.session_state.processing:
                    self.process_pdf(uploaded_file, summary_type, temperature)
                else:
                    st.warning("Processing in progress...")

        # Display current summary if available
        if st.session_state.current_summary:
            self.display_summary(st.session_state.current_summary)

    def process_pdf(self, uploaded_file, summary_type, temperature):
        """Process the uploaded PDF and generate summary"""
        st.session_state.processing = True

        try:
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Step 1: Save file and create document record
            status_text.text("📁 Saving file...")
            progress_bar.progress(20)

            # Save uploaded file
            upload_path = self.config.get_upload_path(uploaded_file.name)
            with open(upload_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Create document record
            from pdf_processor.extractor import PDFExtractor
            from database.models import PDFDocument

            extractor = PDFExtractor()
            document = PDFDocument(
                filename=uploaded_file.name,
                original_filename=uploaded_file.name,
                file_path=upload_path,
                file_size=uploaded_file.size
            )

            # Step 2: Extract text
            status_text.text("📖 Extracting text from PDF...")
            progress_bar.progress(40)

            extraction_result = extractor.extract_text(upload_path)
            if not extraction_result["success"]:
                st.error(f"❌ Failed to extract text: {extraction_result.get('error', 'Unknown error')}")
                return

            # Update document with extracted info
            document.text_length = len(extraction_result["content"])
            document.page_count = extraction_result.get("page_count", 0)
            document.is_processed = True

            document_id = self.db_manager.save_document(document)

            # Step 3: Generate summary
            status_text.text("🤖 Generating summary with AI...")
            progress_bar.progress(70)

            summary_result = self.summarizer.summarize(
                extraction_result["content"],
                summary_type=summary_type,
                temperature=temperature
            )

            if not summary_result["success"]:
                st.error(f"❌ Failed to generate summary: {summary_result.get('error', 'Unknown error')}")
                return

            # Step 4: Save summary
            status_text.text("💾 Saving summary...")
            progress_bar.progress(90)

            from database.models import Summary
            summary = Summary(
                document_id=document_id,
                summary_text=summary_result["content"],
                summary_type=summary_type,
                processing_time=summary_result.get("processing_time", 0),
                model_used=summary_result.get("model_used", "unknown"),
                temperature=temperature
            )

            summary_id = self.db_manager.save_summary(summary)

            # Step 5: Complete
            progress_bar.progress(100)
            status_text.text("✅ Summary generated successfully!")

            # Store summary in session state
            st.session_state.current_summary = {
                "id": summary_id,
                "document_id": document_id,
                "content": summary_result["content"],
                "type": summary_type,
                "processing_time": summary_result.get("processing_time", 0),
                "model_used": summary_result.get("model_used", "unknown"),
                "document_info": {
                    "name": uploaded_file.name,
                    "size": uploaded_file.size,
                    "pages": extraction_result.get("page_count", 0)
                }
            }

            # Success message
            st.success("🎉 Summary generated successfully!")
            st.balloons()

        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            self.logger.error(f"PDF processing error: {e}")

        finally:
            st.session_state.processing = False

    def display_summary(self, summary_data):
        """Display the generated summary"""
        st.subheader("📄 Generated Summary")

        # Summary metadata
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Type", summary_data["type"].title())
        with col2:
            st.metric("Processing Time", f"{summary_data['processing_time']:.2f}s")
        with col3:
            st.metric("Model", summary_data["model_used"])

        # Summary content
        st.markdown("### Summary Content")
        st.write(summary_data["content"])

        # Export options
        st.subheader("💾 Export Summary")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("📄 Export as PDF", key="export_pdf"):
                self.export_summary(summary_data, "pdf")

        with col2:
            if st.button("📝 Export as Text", key="export_text"):
                self.export_summary(summary_data, "txt")

    def export_summary(self, summary_data, format_type):
        """Export summary to file"""
        try:
            from utils.exporters import export_summary

            # Create filename
            doc_name = summary_data["document_info"]["name"].replace(".pdf", "")
            filename = f"{doc_name}_summary.{format_type}"

            # Export summary
            export_path = self.config.get_export_path(filename)
            success = export_summary(summary_data, export_path, format_type)

            if success:
                st.success(f"✅ Summary exported to {filename}")

                # Provide download button
                with open(export_path, "rb") as file:
                    st.download_button(
                        label=f"📥 Download {format_type.upper()}",
                        data=file.read(),
                        file_name=filename,
                        mime="application/pdf" if format_type == "pdf" else "text/plain"
                    )
            else:
                st.error(f"❌ Failed to export summary as {format_type}")

        except Exception as e:
            st.error(f"❌ Export error: {str(e)}")
            self.logger.error(f"Export error: {e}")

    def show_history_page(self):
        """Show the document history page"""
        st.header("📚 Document History")

        try:
            # Get all documents
            documents = self.db_manager.get_all_documents()

            if not documents:
                st.info("No documents processed yet. Upload a PDF to get started!")
                return

            # Display documents
            for doc in documents:
                with st.expander(f"📄 {doc.original_filename}"):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write(f"**Upload Date:** {doc.upload_date.strftime('%Y-%m-%d %H:%M')}")
                        st.write(f"**File Size:** {doc.file_size / 1024:.1f} KB")
                        st.write(f"**Pages:** {doc.page_count}")
                        st.write(f"**Status:** {'✅ Processed' if doc.is_processed else '⏳ Pending'}")

                    with col2:
                        if doc.is_processed:
                            # Get summaries for this document
                            summaries = self.db_manager.get_summaries_for_document(doc.id)

                            if summaries:
                                st.write(f"**Summaries:** {len(summaries)}")

                                for summary in summaries:
                                    if st.button(f"View {summary.summary_type.title()} Summary",
                                              key=f"summary_{summary.id}"):
                                        st.session_state.current_summary = {
                                            "id": summary.id,
                                            "document_id": doc.id,
                                            "content": summary.summary_text,
                                            "type": summary.summary_type,
                                            "processing_time": summary.processing_time,
                                            "model_used": summary.model_used,
                                            "document_info": {
                                                "name": doc.original_filename,
                                                "size": doc.file_size,
                                                "pages": doc.page_count
                                            }
                                        }
                                        st.rerun()
                            else:
                                st.write("No summaries available")

                st.divider()

        except Exception as e:
            st.error(f"❌ Error loading history: {str(e)}")
            self.logger.error(f"History page error: {e}")

    def show_settings_page(self):
        """Show the settings page"""
        st.header("⚙️ Settings")

        st.subheader("🔧 Application Settings")

        # Display current configuration
        st.write("**Current Configuration:**")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Max File Size", f"{self.config.file_upload.max_file_size / (1024*1024):.0f} MB")
            st.metric("Chunk Size", str(self.config.text_processing.chunk_size))
            st.metric("Default Summary Type", self.config.summary.default_summary_type.title())

        with col2:
            st.metric("Temperature", str(self.config.summary.temperature))
            st.metric("Max History Entries", str(self.config.performance.max_history_entries))
            st.metric("Database", self.config.database.path)

        st.info("💡 To change these settings, modify the `.env` file and restart the application.")

    def show_about_page(self):
        """Show the about page"""
        st.header("ℹ️ About PDF Summarizer AI")

        st.markdown("""
        ### 🎯 **About**

        PDF Summarizer AI is a powerful application that uses advanced AI technology to generate concise, accurate summaries of PDF documents.

        ### 🚀 **Features**

        - **📤 Easy Upload**: Simply drag and drop your PDF files
        - **🤖 AI-Powered**: Uses Groq's Llama 3.3 70B model for high-quality summaries
        - **🎛️ Customizable**: Choose from different summary types and creativity levels
        - **📚 History**: Keep track of all your processed documents
        - **💾 Export**: Save summaries as PDF or text files
        - **🔒 Private**: All processing happens locally on your machine

        ### 🛠️ **Technology Stack**

        - **Frontend**: Streamlit
        - **AI/ML**: Groq API with Llama 3.3 70B
        - **PDF Processing**: PyMuPDF
        - **Database**: SQLite
        - **Language**: Python 3.11+

        ### 📋 **Version**

        **PDF Summarizer AI** v1.0.0

        ### 🔗 **Links**

        - [Groq AI](https://groq.com)
        - [Llama 3.3](https://ai.meta.com/llama/)
        - [Streamlit](https://streamlit.io)

        ---

        Made with ❤️ using AI technology
        """)

# Legacy function for backward compatibility
def run_app():
    """Legacy function to run the app"""
    config = get_config()
    app = PDFSummarizerApp(config)
    app.run()