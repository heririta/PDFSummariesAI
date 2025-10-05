# PDF Summarizer AI

A Python application that uses AI to summarize PDF documents using Streamlit, LangChain, and Groq API.

## Features

- Upload and process PDF files
- Extract text from PDFs using PyMuPDF
- Generate summaries using Groq's LLM (llama-3.3-70b-versatile)
- Store summaries and metadata in SQLite database
- View processing history
- Export summaries to text or PDF format

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and add your Groq API key
5. Run the application:
   ```bash
   streamlit run main.py
   ```

## Project Structure

```
PDFSummariesAI/
├── src/
│   ├── pdf_processor/     # PDF text extraction and processing
│   ├── llm_integration/   # LLM integration and summarization
│   ├── database/          # Database operations
│   ├── ui/               # Streamlit UI components
│   └── utils/            # Utility functions
├── tests/
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests
│   ├── contract/         # Contract tests
│   └── fixtures/         # Test fixtures
├── data/                 # Database and data files
├── static/              # Static assets
├── templates/           # HTML templates
└── logs/                # Application logs
```