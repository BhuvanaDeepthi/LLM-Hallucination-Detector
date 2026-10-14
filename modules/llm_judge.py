import json
import re
from typing import Dict, Any, Optional
import ollama
import config

class LLMJudge:
    """Second-stage LLM-as-a-Judge for evaluating ambiguous NLI cases using local Gemma (or Gemini)."""

    def __init__(self, ollama_model: str = config.DEFAULT_OLLAMA_MODEL):
        self.ollama_model = ollama_model
        self.gemini_api_key = config.GEMINI_API_KEY
        self.gemini_model = config.DEFAULT_GEMINI_MODEL

    def _build_judge_prompt(
        self,
        claim: str,
        evidence: str,
        nli_relation: str,
        nli_scores: Dict[str, float]
    ) -> str:
        return f"""You are an impartial, highly rigorous verification judge evaluating whether an AI claim is supported, contradicted, or unsupported by given document evidence.

Claim to Verify:
\"{claim}\"

Source Document Evidence:
\"{evidence}\"

Initial NLI Model Analysis:
- Preliminary Classification: {nli_relation}
- Entailment Score: {nli_scores.get('entailment', 0.0):.3f}
- Contradiction Score: {nli_scores.get('contradiction', 0.0):.3f}
- Neutral Score: {nli_scores.get('neutral', 0.0):.3f}

Evaluation Instructions:
1. Compare the claim strictly and exclusively against the provided source evidence.
2. Do NOT use outside knowledge or extrapolate assumptions.
3. Choose one of the following exact verdicts:
   - "Supported": The source evidence directly confirms the claim.
   - "Contradicted": The source evidence directly refutes or gives conflicting facts.
   - "Unsupported": The source evidence lacks sufficient information to either confirm or deny the claim.
4. Provide a brief 1-2 sentence rationale explaining the decision.
5. Return ONLY a valid JSON object with keys "verdict" and "rationale".

JSON Output Format:
{{
  "verdict": "Supported" | "Contradicted" | "Unsupported",
  "rationale": "Detailed explanation based strictly on the evidence text."
}}
"""

    def assess_claim(
        self,
        claim_result: Dict[str, Any],
        use_gemini: bool = False
    ) -> Dict[str, Any]:
        """
        Assesses a single claim result.
        Returns updated claim_result dictionary with judge_assessment and judge_used.
        Preserves original NLI result if judge is unavailable.
        """
        claim = claim_result.get("claim_text", "")
        evidence = claim_result.get("evidence_text", "")
        nli_relation = claim_result.get("nli_relation", "Neutral")
        nli_scores = {
            "entailment": claim_result.get("entailment_score", 0.0),
            "contradiction": claim_result.get("contradiction_score", 0.0),
            "neutral": claim_result.get("neutral_score", 0.0)
        }

        prompt = self._build_judge_prompt(claim, evidence, nli_relation, nli_scores)
        raw_output = ""

        # Attempt Gemini if requested
        if use_gemini and self.gemini_api_key:
            try:
                from google import genai
                client = genai.Client(api_key=self.gemini_api_key.strip())
                res = client.models.generate_content(model=self.gemini_model, contents=prompt)
                raw_output = res.text.strip()
            except Exception:
                raw_output = ""

        # Try local Ollama Gemma
        if not raw_output:
            try:
                res = ollama.chat(
                    model=self.ollama_model,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0.1}
                )
                raw_output = res["message"]["content"].strip()
            except Exception:
                # Judge unavailable: preserve initial NLI result cleanly
                claim_result["judge_used"] = False
                claim_result["judge_assessment"] = "LLM-as-Judge was unavailable; initial NLI verdict preserved."
                return claim_result

        # Parse JSON output
        try:
            cleaned = raw_output.strip()
            if "```" in cleaned:
                match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
                if match:
                    cleaned = match.group(1).strip()
            
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1:
                cleaned = cleaned[start:end + 1]

            parsed = json.loads(cleaned)
            judge_verdict = parsed.get("verdict", "").strip()
            rationale = parsed.get("rationale", "").strip()

            # Validate verdict is one of Supported, Contradicted, Unsupported
            valid_verdicts = [config.VERDICT_SUPPORTED, config.VERDICT_CONTRADICTED, config.VERDICT_UNSUPPORTED]
            matched_verdict = next((v for v in valid_verdicts if v.lower() == judge_verdict.lower()), None)

            if matched_verdict:
                claim_result["verdict"] = matched_verdict
                claim_result["judge_used"] = True
                claim_result["judge_assessment"] = rationale
            else:
                claim_result["judge_used"] = True
                claim_result["judge_assessment"] = rationale or f"Judge output: {raw_output[:120]}"

        except Exception:
            claim_result["judge_used"] = False
            claim_result["judge_assessment"] = "Judge response parsing failed; NLI verdict preserved."

        return claim_result
