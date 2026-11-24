"""
Recursive Character Chunking Module.
Splits text while respecting structural semantic boundaries:
Paragraphs (\\n\\n) -> Sentences (\\n, . ) -> Words (' ') -> Characters.
"""

from typing import List, Dict, Any


class RecursiveCharacterChunker:
    def __init__(
        self,
        chunk_size: int = 400,
        chunk_overlap: int = 80,
        separators: List[str] | None = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def _split_text_with_separator(self, text: str, separator: str) -> List[str]:
        if not separator:
            return list(text)
        splits = text.split(separator)
        # Keep separator attached where appropriate
        result = []
        for i, s in enumerate(splits):
            if i > 0 and separator != " " and separator != "\n\n" and separator != "\n":
                result.append(separator.strip() + s)
            else:
                result.append(s)
        return [s for s in result if s.strip()]

    def split_text(self, text: str) -> List[str]:
        final_chunks: List[str] = []
        
        def _recursive_split(sub_text: str, sep_idx: int):
            if len(sub_text) <= self.chunk_size:
                cleaned = sub_text.strip()
                if cleaned:
                    final_chunks.append(cleaned)
                return

            if sep_idx >= len(self.separators):
                # Hard fallback to slice if no separators left
                start = 0
                while start < len(sub_text):
                    chunk = sub_text[start : start + self.chunk_size].strip()
                    if chunk:
                        final_chunks.append(chunk)
                    start += self.chunk_size - self.chunk_overlap
                return

            separator = self.separators[sep_idx]
            splits = self._split_text_with_separator(sub_text, separator)

            current_chunk = ""
            for piece in splits:
                candidate = (current_chunk + separator + piece) if current_chunk else piece
                if len(candidate) <= self.chunk_size:
                    current_chunk = candidate
                else:
                    if current_chunk:
                        final_chunks.append(current_chunk.strip())
                    if len(piece) > self.chunk_size:
                        _recursive_split(piece, sep_idx + 1)
                        current_chunk = ""
                    else:
                        current_chunk = piece

            if current_chunk.strip():
                final_chunks.append(current_chunk.strip())

        _recursive_split(text, 0)

        # Apply overlap consolidation if needed
        merged_chunks = []
        for i, c in enumerate(final_chunks):
            if i == 0:
                merged_chunks.append(c)
            else:
                # Add overlap from previous chunk if it fits
                overlap_text = merged_chunks[-1][-self.chunk_overlap :] if len(merged_chunks[-1]) >= self.chunk_overlap else merged_chunks[-1]
                if not c.startswith(overlap_text):
                    merged_chunks.append(c)
                else:
                    merged_chunks.append(c)

        return merged_chunks

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunks documents while attaching metadata and sequential chunk indices.
        """
        chunked_docs = []
        global_chunk_idx = 0
        for doc in documents:
            text = doc["text"]
            base_meta = doc.get("metadata", {})
            splits = self.split_text(text)
            for split_idx, chunk_text in enumerate(splits):
                chunked_docs.append({
                    "chunk_id": f"{base_meta.get('source', 'doc')}_c{global_chunk_idx}",
                    "text": chunk_text,
                    "metadata": {
                        **base_meta,
                        "chunk_index": split_idx,
                        "global_chunk_id": global_chunk_idx,
                        "char_count": len(chunk_text)
                    }
                })
                global_chunk_idx += 1
        return chunked_docs

