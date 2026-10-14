import json
import re
from typing import List, Dict, Any, Optional
import ollama
import config

UNCERTAINTY_PHRASES = [
    "does not contain sufficient information",
    "does not provide enough information",
    "not enough information to",
    "cannot be determined from the",
    "does not specify whether",
    "does not specify",
    "does not mention",
    "no information provided",
    "is not specified in the document",
    "not establish this claim",
    "cannot establish this claim",
    "lacks sufficient information",
    "insufficient evidence",
    "does not state whether",
    "is not mentioned in the",
    "no evidence to support",
    "not explicitly mentioned"
]

class ClaimExtractor:
    """Extracts atomic, verifiable factual claims from an LLM-generated answer and user query."""

    def __init__(self, ollama_model: str = config.DEFAULT_OLLAMA_MODEL, gemini_api_key: Optional[str] = None):
        self.ollama_model = ollama_model
        self.gemini_api_key = (gemini_api_key or config.GEMINI_API_KEY or "").strip()
        self.gemini_model = config.DEFAULT_GEMINI_MODEL

    @staticmethod
    def is_meta_or_uncertainty_phrase(text: str) -> bool:
        """Identifies AI refusal or uncertainty phrases that should NOT be verified as factual claims."""
        if not text:
            return False
        text_lower = text.lower()
        for phrase in UNCERTAINTY_PHRASES:
            if phrase in text_lower:
                return True
        return False

    @staticmethod
    def extract_core_assertion_from_query(query: str) -> Optional[str]:
        """
        Extracts the underlying declarative assertion from a user question.
        Examples:
        - 'The assignment requires DAX measures for every scenario.' -> 'The assignment requires DAX measures for every scenario.'
        - 'The assignment requires DAX measures. Is this statement correct according to the document?' -> 'The assignment requires DAX measures.'
        - 'Does the assignment require DAX measures for every scenario?' -> 'The assignment requires DAX measures for every scenario.'
        - 'Is DAX required for every scenario?' -> 'DAX is required for every scenario.'
        """
        if not query or not query.strip():
            return None

        cleaned = query.strip()

        # Remove leading meta-prompts
        cleaned = re.sub(
            r"(?i)^(can you verify (if|whether)|is it true that|verify that|check if|does the document state (that)?)\s*",
            "",
            cleaned
        )

        # Convert "Does X require Y?" -> "X requires Y."
        match_does = re.match(r"(?i)^does\s+([a-zA-Z0-9\s_-]+?)\s+require\s+(.+?)(\?|$)", cleaned)
        if match_does:
            subject = match_does.group(1).strip()
            rest = match_does.group(2).strip()
            res = f"{subject} requires {rest}."
            return res[0].upper() + res[1:]

        # Convert "Is X required...?" -> "X is required..."
        match_is = re.match(r"(?i)^is\s+([a-zA-Z0-9\s_-]+?)\s+(required|mandatory|needed)\s*(.*?)(\?|$)", cleaned)
        if match_is:
            subj = match_is.group(1).strip()
            adj = match_is.group(2).strip()
            rest = match_is.group(3).strip()
            result = f"{subj} is {adj} {rest}.".strip()
            res = re.sub(r"\s+", " ", result)
            return res[0].upper() + res[1:]

        # Remove common verification suffixes or meta-questions at the end
        cleaned = re.sub(
            r"(?i)\.?\s*(is this statement correct|is this true|is that true|can you verify this|according to the (document|pdf)).*$",
            "",
            cleaned
        )
        cleaned = cleaned.strip(".?! \t\n")

        # If it looks like a declarative claim (at least 3 words and not a pure WH question)
        words = cleaned.split()
        if len(words) >= 3 and not cleaned.lower().startswith(("what", "who", "when", "where", "why", "how", "which")):
            res = cleaned + "."
            return res[0].upper() + res[1:]

        return None

    def _build_extraction_prompt(self, answer_text: str) -> str:
        return f"""Decompose the following text into a JSON array of independent, atomic, factual claims.
Guidelines:
1. Each claim must be a single self-contained factual statement.
2. Ignore meta-commentary, conversational filler, and statements of uncertainty (e.g. "The document does not mention...").
3. Only extract positive or negative factual claims made about the subject matter.

Text:
\"{answer_text}\"

Return ONLY a valid JSON array like:
[{{"claim": "First factual statement"}}, {{"claim": "Second factual statement"}}]
"""

    def _clean_json_response(self, raw_text: str) -> str:
        """Strips markdown code blocks, backticks, and extraneous text."""
        text = raw_text.strip()
        if "```" in text:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
            if match:
                text = match.group(1).strip()
            else:
                text = re.sub(r"```[a-zA-Z]*", "", text).replace("```", "").strip()

        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]

        return text

    def rule_based_extract(self, text: str) -> List[str]:
        """Instant sentence-level claim splitter that filters out meta-statements and uncertainty phrases."""
        if not text or not text.strip():
            return []

        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        claims = []
        META_PREFIXES = (
            "hello", "sure", "here is", "in summary", "based on",
            "the statement is", "the claim is", "this statement is", "this claim is",
            "according to the document", "the source document specifies", "the source document confirms",
            "the source document contradicts", "the source document states"
        )
        META_SUBSTRINGS = (
            "conflicts with requiring", "contradicts this statement", "confirms this statement",
            "explicitly supported by the source", "supported by the source"
        )

        for s in sentences:
            cleaned = s.strip()
            # Strip introductory clauses like "Based on the provided evidence, " or "According to the document, "
            cleaned = re.sub(
                r"^(based on (the )?(provided )?(evidence|document|text),?\s*|according to the (document|evidence|source),?\s*|the document (states|specifies|indicates) (that )?)",
                "",
                cleaned,
                flags=re.IGNORECASE
            ).strip()

            # Remove markdown formatting (**bold**, *italics*)
            cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
            cleaned = re.sub(r"\*([^*]+)\*", r"\1", cleaned)

            # Remove parenthetical page citations generated by LLMs e.g. (Page 2), (Page 2, Page 29)
            cleaned = re.sub(r"\s*\((?:pages?|source|see)\s*[^)]+\)", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()

            if cleaned:
                cleaned = cleaned[0].upper() + cleaned[1:]
                if not cleaned.endswith((".", "!", "?")):
                    cleaned += "."

            META_PREFIXES = (
                "hello", "sure", "here is", "in summary",
                "the statement is", "the claim is", "this statement is", "this claim is"
            )
            META_SUBSTRINGS = (
                "conflicts with requiring", "contradicts this statement", "confirms this statement",
                "explicitly supported by the source", "supported by the source"
            )

            if len(cleaned.split()) >= 3:
                clow = cleaned.lower()
                # Filter out conversational filler and meta-uncertainty phrases
                if not clow.startswith(META_PREFIXES) and not any(sub in clow for sub in META_SUBSTRINGS):
                    if not self.is_meta_or_uncertainty_phrase(cleaned):
                        claims.append(cleaned)
        return claims

    # Alias for compatibility with tests
    _rule_based_fallback = rule_based_extract

    @staticmethod
    def is_question(query: str) -> bool:
        """
        Determines whether a user input is an inquiry (question) or a direct factual claim/statement.
        """
        if not query or not query.strip():
            return False
        q = query.strip()
        if q.endswith("?"):
            return True
        first_word = q.split()[0].lower() if q.split() else ""
        first_word = re.sub(r"[^a-z]", "", first_word)
        question_words = {
            "what", "who", "whom", "whose", "which", "where", "when", "why", "how"
        }
        if first_word in question_words:
            return True
        aux_verbs = {"is", "are", "does", "do", "did", "can", "could", "would", "should", "will", "has", "have"}
        if first_word in aux_verbs and ("?" in q or len(q.split()) <= 8):
            return True
        return False

    def extract_claims(
        self,
        answer_text: str,
        user_query: Optional[str] = None,
        is_query_question: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        Extracts atomic claims from answer text and connects to the user query assertion.
        Returns a list of dicts: [{'claim_text': str, 'is_uncertainty_for_claim': bool, 'source': str}]
        Keeps user query assertion and generated answer claims clearly distinguished.
        """
        if not answer_text or not answer_text.strip():
            return []

        if is_query_question is None and user_query:
            is_query_question = self.is_question(user_query)

        answer_indicates_insufficient = self.is_meta_or_uncertainty_phrase(answer_text)
        core_assertion = self.extract_core_assertion_from_query(user_query) if user_query else None

        # Rule-based extraction splits into sentences and filters out meta-uncertainty phrases
        substantive_claims = self.rule_based_extract(answer_text)

        # 1. Fast path: If the answer indicates uncertainty and has NO substantive factual claims
        if answer_indicates_insufficient and not substantive_claims:
            if is_query_question:
                return [{
                    "claim_text": answer_text.strip(),
                    "is_uncertainty_for_claim": True,
                    "source": "generated_answer"
                }]
            else:
                fallback_label = core_assertion or (
                    user_query.strip("?. ") + "." if user_query else "The specified factual claim."
                )
                return [{
                    "claim_text": fallback_label,
                    "is_uncertainty_for_claim": True,
                    "source": "user_statement"
                }]

        raw_claims: List[str] = []

        # 2. Fast single-attempt Cloud JSON extraction if Gemini API key present and multiple complex claims
        if self.gemini_api_key and len(substantive_claims) > 2:
            try:
                from google import genai
                from google.genai import types
                client = genai.Client(api_key=self.gemini_api_key)
                prompt = self._build_extraction_prompt(answer_text)
                config_gen = types.GenerateContentConfig(
                    max_output_tokens=150,
                    temperature=0.1
                )
                res = client.models.generate_content(
                    model=self.gemini_model or "gemini-3.8-flash",
                    contents=prompt,
                    config=config_gen
                )
                if res.text and res.text.strip():
                    cleaned_json = self._clean_json_response(res.text.strip())
                    parsed = json.loads(cleaned_json)
                    if isinstance(parsed, list):
                        for item in parsed:
                            txt = item.get("claim", "") if isinstance(item, dict) else str(item)
                            if txt.strip() and not self.is_meta_or_uncertainty_phrase(txt):
                                raw_claims.append(txt.strip())
            except Exception:
                pass  # Fallthrough to rule-based claims

        # Fallback to rule-based claims if LLM didn't run or produce claims
        if not raw_claims:
            raw_claims = substantive_claims

        formatted_claims: List[Dict[str, Any]] = []

        # 3. If user entered a statement (not an open-ended question), add core assertion as primary claim
        if user_query and not is_query_question:
            statement_label = core_assertion or (user_query.strip("?. ") + ".")
            formatted_claims.append({
                "claim_text": statement_label,
                "is_uncertainty_for_claim": answer_indicates_insufficient,
                "source": "user_statement"
            })

        # 4. Add substantive claims from the answer (excluding duplicates of query assertion)
        for c in raw_claims:
            if core_assertion and c.lower().strip(". ") == core_assertion.lower().strip(". "):
                continue
            formatted_claims.append({
                "claim_text": c,
                "is_uncertainty_for_claim": False,
                "source": "generated_answer"
            })

        # If still empty for question, use cleaned first sentence of answer (never the user question)
        if not formatted_claims and answer_text:
            first_ans = answer_text.split(".")[0].strip()
            if len(first_ans.split()) >= 3 and not self.is_meta_or_uncertainty_phrase(first_ans):
                formatted_claims.append({
                    "claim_text": first_ans + ".",
                    "is_uncertainty_for_claim": False,
                    "source": "generated_answer"
                })
        elif not formatted_claims and user_query and not is_query_question:
            formatted_claims.append({
                "claim_text": core_assertion or user_query.strip("?. ") + ".",
                "is_uncertainty_for_claim": False,
                "source": "user_statement"
            })

        return formatted_claims
