from setuptools import setup, find_packages

setup(
    name="pdf-summarizer-ai",
    version="0.1.0",
    description="AI-powered PDF summarizer using Groq and LangChain",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "streamlit>=1.28.0",
        "langchain>=0.1.0",
        "langgraph>=0.0.40",
        "fitz>=1.23.0",
        "groq>=0.5.0",
        "python-dotenv>=1.0.0",
        "tiktoken>=0.5.0",
        "pypdf>=3.17.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ]
    },
    python_requires=">=3.8",
)