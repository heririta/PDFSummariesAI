"""
Groq LLM Summarizer Module
Handles summarization using Groq API
"""

import requests
import json
import time
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class LLMSummarizer:
    """Handles text summarization using Groq API"""

    def __init__(self, config):
        """Initialize the summarizer with configuration"""
        self.config = config
        self.api_key = config.groq.api_key
        self.model = config.groq.model
        self.base_url = config.groq.base_url
        self.timeout = config.groq.timeout
        self.max_retries = config.groq.max_retries
        self.logger = logging.getLogger(__name__)

    def validate_api_key(self) -> bool:
        """Validate the Groq API key"""
        try:
            if not self.api_key:
                return False

            # Test API key with a simple request
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            # Test with a minimal request
            data = {
                "model": self.model,
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 1
            }

            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=10
            )

            return response.status_code == 200

        except Exception as e:
            self.logger.error(f"API key validation failed: {e}")
            return False

    def summarize(self, text: str, summary_type: str = "concise",
                  temperature: float = None, max_tokens: int = None) -> Dict[str, Any]:
        """
        Summarize text using Groq LLM

        Args:
            text: Text to be summarized
            summary_type: Type of summary (concise, detailed, bullet_points)
            temperature: Temperature for generation
            max_tokens: Maximum tokens in response

        Returns:
            Dictionary with summary results
        """
        try:
            if not text or not text.strip():
                return {
                    "success": False,
                    "error": "Text content cannot be empty",
                    "error_type": "empty_content"
                }

            if temperature is None:
                temperature = self.config.summary.temperature

            if max_tokens is None:
                max_tokens = self.config.summary.max_summary_length

            # Get appropriate prompt based on summary type
            prompt = self._get_summary_prompt(summary_type, text)

            # Prepare API request
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            data = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False
            }

            start_time = time.time()

            # Make API request with retries
            for attempt in range(self.max_retries + 1):
                try:
                    response = requests.post(
                        f"{self.base_url}/v1/chat/completions",
                        headers=headers,
                        json=data,
                        timeout=self.timeout
                    )

                    if response.status_code == 200:
                        result = response.json()
                        processing_time = time.time() - start_time

                        summary_content = result["choices"][0]["message"]["content"].strip()
                        token_count = result.get("usage", {}).get("total_tokens", 0)

                        return {
                            "success": True,
                            "content": summary_content,
                            "processing_time": processing_time,
                            "token_count": token_count,
                            "model_used": self.model,
                            "summary_type": summary_type,
                            "temperature": temperature,
                            "max_tokens": max_tokens
                        }
                    else:
                        error_msg = f"API error: {response.status_code} - {response.text}"
                        if attempt < self.max_retries:
                            time.sleep(2 ** attempt)  # Exponential backoff
                            continue
                        else:
                            return {
                                "success": False,
                                "error": error_msg,
                                "error_type": "api_error",
                                "retry_count": attempt
                            }

                except requests.exceptions.Timeout:
                    if attempt < self.max_retries:
                        time.sleep(2 ** attempt)
                        continue
                    else:
                        return {
                            "success": False,
                            "error": "Request timeout",
                            "error_type": "timeout",
                            "retry_count": attempt
                        }

                except Exception as e:
                    if attempt < self.max_retries:
                        time.sleep(2 ** attempt)
                        continue
                    else:
                        return {
                            "success": False,
                            "error": f"Request failed: {str(e)}",
                            "error_type": "request_error",
                            "retry_count": attempt
                        }

        except Exception as e:
            self.logger.error(f"Summarization error: {e}")
            return {
                "success": False,
                "error": f"Summarization failed: {str(e)}",
                "error_type": "general_error"
            }

    def _get_summary_prompt(self, summary_type: str, text: str) -> str:
        """Get the appropriate prompt for the summary type"""
        prompts = {
            "concise": f"Create a concise summary (2-3 sentences) that captures the main points:\n\n{text}",
            "detailed": f"Create a detailed summary (1-2 paragraphs) that covers the key information:\n\n{text}",
            "bullet_points": f"Create a bullet point summary with the main points:\n\n{text}"
        }

        return prompts.get(summary_type, prompts["concise"])

    def chunk_and_summarize(self, text: str, summary_type: str = "concise",
                           chunk_size: int = 4000, overlap: int = 400) -> Dict[str, Any]:
        """
        Summarize large text by chunking it first

        Args:
            text: Text to be summarized
            summary_type: Type of summary
            chunk_size: Size of each chunk
            overlap: Overlap between chunks

        Returns:
            Dictionary with summary results
        """
        try:
            # Simple chunking
            if len(text) <= chunk_size:
                return self.summarize(text, summary_type)

            chunks = []
            start = 0

            while start < len(text):
                end = min(start + chunk_size, len(text))
                chunk = text[start:end]
                chunks.append(chunk)

                if end >= len(text):
                    break

                start = end - overlap

            # Summarize each chunk
            chunk_summaries = []
            total_tokens = 0
            total_time = 0

            for i, chunk in enumerate(chunks):
                result = self.summarize(chunk, summary_type)

                if result["success"]:
                    chunk_summaries.append(result["content"])
                    total_tokens += result.get("token_count", 0)
                    total_time += result.get("processing_time", 0)
                else:
                    self.logger.warning(f"Failed to summarize chunk {i+1}")

            if not chunk_summaries:
                return {
                    "success": False,
                    "error": "Failed to summarize any chunks",
                    "error_type": "chunk_processing_failed"
                }

            # Combine chunk summaries
            combined_text = "\n\n".join(chunk_summaries)

            # Final summary of combined chunks
            if len(chunk_summaries) > 1:
                final_result = self.summarize(combined_text, summary_type)
                if final_result["success"]:
                    return {
                        "success": True,
                        "content": final_result["content"],
                        "chunks_used": len(chunks),
                        "chunk_summaries": chunk_summaries,
                        "total_tokens": total_tokens + final_result.get("token_count", 0),
                        "processing_time": total_time + final_result.get("processing_time", 0),
                        "model_used": self.model,
                        "summary_type": summary_type
                    }

            return {
                "success": True,
                "content": combined_text,
                "chunks_used": len(chunks),
                "chunk_summaries": chunk_summaries,
                "total_tokens": total_tokens,
                "processing_time": total_time,
                "model_used": self.model,
                "summary_type": summary_type
            }

        except Exception as e:
            self.logger.error(f"Chunk and summarize error: {e}")
            return {
                "success": False,
                "error": f"Chunk processing failed: {str(e)}",
                "error_type": "chunk_error"
            }

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            "model_name": self.model,
            "provider": "Groq",
            "max_tokens": 8192,  # Approximate for Llama 3.3
            "supports_streaming": True,
            "supports_function_calling": True
        }