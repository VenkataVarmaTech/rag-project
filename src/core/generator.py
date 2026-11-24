"""
Generation & Prompt Synthesis Module.
Constructs grounded prompts, enforces hallucination guardrails,
and synthesizes answers with source citations.
"""

import os
from typing import List, Dict, Any


class GroundedGenerator:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")

    def format_context(self, chunks: List[Dict[str, Any]]) -> str:
        formatted_blocks = []
        for i, c in enumerate(chunks, start=1):
            source = c.get("metadata", {}).get("source", "Unknown")
            page = c.get("metadata", {}).get("page", 1)
            score = c.get("rerank_score", c.get("score", 0.0))
            block = (
                f"[Source {i} | File: {source} (Page {page}) | Score: {score:.3f} | ID: {c['chunk_id']}]:\n"
                f"{c['text']}"
            )
            formatted_blocks.append(block)
        return "\n\n".join(formatted_blocks)

    def build_prompt(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        context_str = self.format_context(chunks)
        prompt = f"""You are a professional enterprise AI assistant.
Answer the user's question using ONLY the provided context snippets below.
Strict rules:
1. Base your answer STRICTLY on the facts present in the context.
2. If the context does not contain sufficient facts to answer, explicitly respond:
   "I cannot answer this question based on the provided documentation."
3. Cite the relevant Source numbers (e.g. [Source 1]) when stating facts.

==================== CONTEXT ====================
{context_str}
=================================================

USER QUESTION: {query}

GROUNDED ANSWER:"""
        return prompt

    def generate_answer(self, query: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generates an answer using an available LLM or provides a deterministic
        grounded synthesis from the retrieved chunks.
        """
        if not chunks:
            return {
                "answer": "I cannot answer this question based on the provided documentation. No relevant chunks were found.",
                "citations": [],
                "prompt_used": self.build_prompt(query, chunks)
            }

        prompt = self.build_prompt(query, chunks)

        # Check if external LLM client is available
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            try:
                from google import genai
                client = genai.Client(api_key=gemini_key)
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                )
                answer_text = response.text.strip()
            except Exception as e:
                answer_text = f"LLM Generation fallback due to API error: {e}"
        else:
            # Deterministic, grounded answer synthesis based on top matching sentences
            # This enables 100% offline testing, CI/CD, and fast evaluation
            top_chunk = chunks[0]
            answer_text = (
                f"Based on {top_chunk.get('metadata', {}).get('source', 'the documentation')} "
                f"[Source 1]:\n{top_chunk['text']}"
            )

        # Extract citations
        citations = []
        for i, c in enumerate(chunks, start=1):
            citations.append({
                "source_index": i,
                "document": c.get("metadata", {}).get("source", "doc"),
                "page": c.get("metadata", {}).get("page", 1),
                "chunk_id": c.get("chunk_id"),
                "relevance_score": c.get("rerank_score", c.get("score", 0.0)),
                "text_snippet": c["text"][:120] + "..." if len(c["text"]) > 120 else c["text"]
            })

        return {
            "answer": answer_text,
            "citations": citations,
            "prompt_used": prompt
        }

