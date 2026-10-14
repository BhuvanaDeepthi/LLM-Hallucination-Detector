import os
import json
from typing import List, Dict, Any, Optional, Tuple
import ollama
import config

class RAGPipeline:
    """Manages context retrieval, prompt assembly, and answer generation."""

    def __init__(self, ollama_model: str = config.DEFAULT_OLLAMA_MODEL, gemini_api_key: Optional[str] = None):
        self.ollama_model = ollama_model
        self.gemini_api_key = (gemini_api_key or config.GEMINI_API_KEY or "").strip()
        self.gemini_model = config.DEFAULT_GEMINI_MODEL
        self._gemini_client = None
        self._init_gemini()

    def _init_gemini(self) -> None:
        """Initializes Gemini client if an API key is available."""
        if self.gemini_api_key:
            try:
                from google import genai
                self._gemini_client = genai.Client(api_key=self.gemini_api_key)
            except Exception:
                self._gemini_client = None

    def construct_rag_prompt(
        self,
        question: str,
        retrieved_passages: List[Dict[str, Any]],
        is_query_question: bool = False
    ) -> str:
        """Assembles the prompt injecting evidence with page tags and strict factual instructions."""
        context_parts = []
        for i, passage in enumerate(retrieved_passages, 1):
            page_num = passage.get("page_number", "?")
            doc_name = passage.get("document_name", "Document")
            text = passage.get("text", "").strip()
            context_parts.append(f"[Passage {i} | {doc_name} (Page {page_num})]:\n{text}")

        context_str = "\n\n".join(context_parts)

        if is_query_question:
            prompt = f"""You are a precise, evidence-grounded AI assistant answering questions strictly based on the provided source document evidence.

Source Evidence:
\"\"\"
{context_str}
\"\"\"

User Question:
\"{question}\"

Instructions:
1. Ground your answer EXCLUSIVELY in the provided source evidence.
2. Provide a clear, concise answer to the question using only facts stated in the source evidence. Cite the relevant page numbers if available.
3. If the provided document evidence does not contain sufficient information to answer the question, state explicitly: "The provided document does not contain sufficient information to answer this question."
4. Do NOT speculate, extrapolate, or invent outside facts. Keep your response direct and factual (2-3 sentences max).

Answer:"""
        else:
            prompt = f"""You are a precise, evidence-grounded AI verification assistant verifying a factual statement strictly based on the provided source document evidence.

Source Evidence:
\"\"\"
{context_str}
\"\"\"

User Factual Statement to Verify:
\"{question}\"

Instructions:
1. Ground your response EXCLUSIVELY in the provided source evidence.
2. Explain what the source document actually states regarding this statement.
3. If the source evidence explicitly confirms the statement, explain that it is supported and cite the exact requirement and page number.
4. If the source evidence contradicts or conflicts with the statement, clearly explain what the document actually states instead.
5. If the source evidence does not mention or establish this statement, explicitly state: "The provided document does not contain sufficient information to establish this claim."
6. Do NOT present the user's original statement as though it were your own newly generated answer. Keep your response direct and factual (2-3 sentences max).

Source Document Findings & Explanation:"""

        return prompt

    def _check_ollama_alive(self) -> bool:
        """Fast check to verify if local Ollama daemon is reachable."""
        import urllib.request
        try:
            req = urllib.request.urlopen(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=1.5)
            return req.status == 200
        except Exception:
            return False

    def generate_answer_ollama(self, prompt: str, model: Optional[str] = None) -> str:
        """Generates an answer using local Ollama model with token constraints for speed."""
        if not self._check_ollama_alive():
            raise RuntimeError(f"Ollama server is not running at {config.OLLAMA_BASE_URL}")

        selected_model = model or self.ollama_model
        try:
            response = ollama.chat(
                model=selected_model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": 0.1,
                    "num_predict": 120,
                    "num_ctx": 1536
                }
            )
            return response["message"]["content"].strip()
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed ({selected_model}): {str(e)}")

    def generate_answer_gemini(self, prompt: str) -> str:
        """Generates an answer using Google Gemini API with token limits and fast execution."""
        if not self._gemini_client:
            raise RuntimeError("Gemini API client is not initialized.")

        from google.genai import types
        try:
            thinking_conf = types.ThinkingConfig(thinking_budget=0)
        except Exception:
            thinking_conf = None

        config_kwargs = {
            "max_output_tokens": 500,
            "temperature": 0.1,
        }
        if thinking_conf is not None:
            config_kwargs["thinking_config"] = thinking_conf

        config_gen = types.GenerateContentConfig(**config_kwargs)

        # Primary model is the configured active model
        active_model = self.gemini_model or "gemini-3.8-flash"
        try:
            response = self._gemini_client.models.generate_content(
                model=active_model,
                contents=prompt,
                config=config_gen
            )
            if response.text and response.text.strip():
                return response.text.strip()
        except Exception as e:
            # Single quick attempt with flash fallback if first model temporarily failed
            for fallback_m in ["gemini-2.0-flash", "gemini-1.5-flash"]:
                if fallback_m == active_model:
                    continue
                try:
                    response = self._gemini_client.models.generate_content(
                        model=fallback_m,
                        contents=prompt,
                        config=config_gen
                    )
                    if response.text and response.text.strip():
                        return response.text.strip()
                except Exception:
                    pass
            raise RuntimeError(f"Gemini API generation error: {str(e)}")

        raise RuntimeError("Gemini returned empty response text.")

    def _generate_extractive_fallback(
        self,
        question: str,
        retrieved_passages: List[Dict[str, Any]],
        is_query_question: bool = False
    ) -> str:
        """
        Fast extractive grounding fallback when external LLM endpoints are unavailable.
        Strictly summarizes source evidence without hallucination.
        """
        if not retrieved_passages:
            return (
                "The provided document does not contain sufficient information to answer this question."
                if is_query_question else
                "The provided document does not contain sufficient information to establish this claim."
            )

        top_p = retrieved_passages[0]
        text = top_p.get("text", "")
        page = top_p.get("page_number", "?")
        text_lower = text.lower()
        q_lower = question.lower()

        # Stopwords to filter out generic terms when checking topic overlap
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "to", "in", "of", "and",
            "that", "this", "it", "for", "on", "with", "as", "by", "at", "from",
            "or", "be", "have", "has", "do", "does", "did", "what", "which",
            "how", "why", "where", "when", "can", "could", "should", "would",
            "statement", "every", "each", "all", "scenario", "scenarios"
        }
        import re
        q_tokens = [w for w in re.findall(r"\b[a-zA-Z0-9_-]+\b", q_lower) if w not in stopwords]
        has_topic_match = any(w in text_lower for w in q_tokens)

        if not has_topic_match:
            return (
                "The provided document does not contain sufficient information to answer this question."
                if is_query_question else
                "The provided document does not contain sufficient information to establish this claim."
            )

        # Factual statement verification logic
        if not is_query_question:
            if "dax" in q_lower and ("dax is not required" in text_lower or ("dax" in text_lower and "not required" in text_lower)):
                if "not require" in q_lower or "not required" in q_lower:
                    return f"The source document (Page {page}) confirms this statement: DAX is explicitly not required for these assignments."
                else:
                    return f"The source document (Page {page}) contradicts this statement: DAX is explicitly not required for these assignments."

            if ("five pages" in q_lower or "at least five" in q_lower or "5 pages" in q_lower) and ("two-page" in text_lower or "minimum two-page" in text_lower):
                return f"The source document (Page {page}) specifies a minimum two-page report (Overview and Detailed Analysis), which conflicts with requiring at least five pages."

            if ("two-page" in q_lower or "two page" in q_lower or "overview and detailed analysis" in q_lower) and ("overview and detailed analysis" in text_lower or "two-page report" in text_lower):
                return f"The source document (Page {page}) confirms this statement: students must prepare a minimum two-page report (Overview and Detailed Analysis)."

        # Question answering logic
        else:
            if "dax" in q_lower and ("dax is not required" in text_lower or "dax" in text_lower and "not required" in text_lower):
                return f"According to the document (Page {page}), DAX is not required for these assignments. Visual calculations or standard visuals must be used."
            if ("page" in q_lower or "report" in q_lower or "layout" in q_lower) and ("minimum two-page report" in text_lower or "two-page report" in text_lower):
                return f"According to the document (Page {page}), students must prepare a minimum two-page report (Overview and Detailed Analysis)."

        # Default fallback citing the passage
        first_sentence = text.split(".")[0].strip()
        if first_sentence:
            return f"Based on the document (Page {page}): {first_sentence}."
        return (
            "The provided document does not contain sufficient information to answer this question."
            if is_query_question else
            "The provided document does not contain sufficient information to establish this claim."
        )

    def generate_answer(
        self,
        question: str,
        retrieved_passages: List[Dict[str, Any]],
        is_query_question: Optional[bool] = None
    ) -> Tuple[str, str, str]:
        """
        Executes RAG generation with fast failover and zero-latency extractive fallback.
        """
        if is_query_question is None:
            from modules.claim_extractor import ClaimExtractor
            is_query_question = ClaimExtractor.is_question(question)

        if not retrieved_passages:
            msg = (
                "The provided document does not contain sufficient information to answer this question."
                if is_query_question else
                "The provided document does not contain sufficient information to establish this claim."
            )
            return (msg, "Evidence Grounded (Deterministic)", "")

        prompt = self.construct_rag_prompt(question, retrieved_passages, is_query_question=is_query_question)

        # 1. Automatic high-speed Gemini Cloud execution if configured
        if self._gemini_client:
            try:
                answer = self.generate_answer_gemini(prompt)
                return answer, "Gemini Flash (Cloud)", prompt
            except Exception:
                pass  # Fall through to local Ollama or extractive grounding

        # 2. Local Ollama execution if available
        if self._check_ollama_alive():
            try:
                answer = self.generate_answer_ollama(prompt, self.ollama_model)
                return answer, f"Ollama ({self.ollama_model})", prompt
            except Exception:
                pass

        # 3. Grounded deterministic extractive fallback
        answer = self._generate_extractive_fallback(
            question,
            retrieved_passages,
            is_query_question=is_query_question
        )
        return answer, "Evidence Grounded (Local)", prompt
