# PDF Summarizer AI Makefile

.PHONY: help install test run lint format clean dev-setup

help:
	@echo "Available commands:"
	@echo "  install      Install dependencies"
	@echo "  test         Run tests"
	@echo "  run          Run the application"
	@echo "  lint         Run linting"
	@echo "  format       Format code"
	@echo "  clean        Clean cache files"
	@echo "  dev-setup    Set up development environment"

install:
	pip install -r requirements.txt

dev-setup: install
	pip install -e .
	pre-commit install

test:
	pytest

test-cov:
	pytest --cov=src --cov-report=html

run:
	streamlit run main.py

lint:
	flake8 src/ tests/
	mypy src/

format:
	black src/ tests/
	isort src/ tests/

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/