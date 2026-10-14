from typing import List, Dict, Any
import re
import config

class TextChunker:
    """Splits document text into overlapping semantic passages while preserving page references."""

    def __init__(self, chunk_size: int = config.CHUNK_SIZE, chunk_overlap: int = config.CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_pages(self, pages: List[Dict[str, Any]], document_name: str) -> List[Dict[str, Any]]:
        """
        Chunks text on a per-page basis (preserving precise page attribution),
        subdividing long pages into overlapping passages of roughly chunk_size words.
        """
        all_chunks = []
        chunk_counter = 0

        for page in pages:
            page_num = page["page_number"]
            text = page["text"]

            if not text or not text.strip():
                continue

            words = text.split()
            if not words:
                continue

            # If page is shorter than chunk size, treat it as a single chunk
            if len(words) <= self.chunk_size:
                chunk_counter += 1
                all_chunks.append({
                    "chunk_id": f"{document_name}_p{page_num}_c{chunk_counter}",
                    "document_name": document_name,
                    "page_number": page_num,
                    "text": " ".join(words),
                    "word_count": len(words)
                })
                continue

            # Sliding window over words
            start_idx = 0
            while start_idx < len(words):
                end_idx = min(start_idx + self.chunk_size, len(words))
                chunk_words = words[start_idx:end_idx]
                chunk_text = " ".join(chunk_words)

                chunk_counter += 1
                all_chunks.append({
                    "chunk_id": f"{document_name}_p{page_num}_c{chunk_counter}",
                    "document_name": document_name,
                    "page_number": page_num,
                    "text": chunk_text,
                    "word_count": len(chunk_words)
                })

                # If we've reached the end of the page text
                if end_idx >= len(words):
                    break

                # Advance window by (chunk_size - overlap)
                step = max(1, self.chunk_size - self.chunk_overlap)
                start_idx += step

        return all_chunks
