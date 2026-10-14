import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pymupdf

class PDFProcessingError(Exception):
    """Custom exception raised when PDF extraction or validation fails."""
    pass

class PDFProcessor:
    """Handles PDF validation, metadata extraction, and page-level text extraction using PyMuPDF."""

    @staticmethod
    def compute_file_hash(file_bytes: bytes) -> str:
        """Computes SHA-256 hash of file content for unique identification."""
        return hashlib.sha256(file_bytes).hexdigest()

    @staticmethod
    def validate_pdf(file_bytes: bytes, filename: str) -> None:
        """
        Validates the PDF file integrity and format.
        Raises PDFProcessingError if the file is invalid.
        """
        if not filename.lower().endswith(".pdf"):
            raise PDFProcessingError(f"File '{filename}' does not have a .pdf extension.")
        
        if len(file_bytes) == 0:
            raise PDFProcessingError(f"Uploaded file '{filename}' is completely empty (0 bytes).")

        # Check standard PDF header %PDF
        if not file_bytes.startswith(b"%PDF"):
            raise PDFProcessingError(f"File '{filename}' does not start with standard PDF header (%PDF).")

    @classmethod
    def extract_text(cls, file_bytes: bytes, filename: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Extracts page-by-page text from PDF bytes.
        Returns:
            pages: List of dictionaries [{'page_number': int, 'text': str, 'char_count': int, 'word_count': int}]
            metadata: Summary metadata including page count, total words, whether scanned/empty.
        """
        cls.validate_pdf(file_bytes, filename)

        try:
            doc = pymupdf.open(stream=file_bytes, filetype="pdf")
        except Exception as e:
            raise PDFProcessingError(f"Failed to open PDF document '{filename}': {str(e)}")

        page_count = len(doc)
        if page_count == 0:
            doc.close()
            raise PDFProcessingError(f"PDF document '{filename}' contains 0 pages.")

        pages_data = []
        total_text_length = 0
        total_words = 0

        for page_idx in range(page_count):
            try:
                page = doc[page_idx]
                page_text = page.get_text("text") or ""
                # Clean up null bytes and excessive whitespace
                cleaned_text = page_text.replace("\x00", " ").strip()
                words = cleaned_text.split()
                
                pages_data.append({
                    "page_number": page_idx + 1,
                    "text": cleaned_text,
                    "char_count": len(cleaned_text),
                    "word_count": len(words)
                })
                total_text_length += len(cleaned_text)
                total_words += len(words)
            except Exception as e:
                pages_data.append({
                    "page_number": page_idx + 1,
                    "text": "",
                    "char_count": 0,
                    "word_count": 0,
                    "error": str(e)
                })

        doc.close()

        # Check for scanned or image-only PDF
        if total_text_length < 50:
            raise PDFProcessingError(
                f"Document '{filename}' appears to be a scanned image-only PDF or empty document. "
                f"Extracted only {total_text_length} characters across {page_count} page(s). "
                "This application requires text-based PDFs. Please upload a PDF with extractable text or OCR-processed text."
            )

        metadata = {
            "filename": filename,
            "page_count": page_count,
            "total_words": total_words,
            "total_chars": total_text_length,
            "file_hash": cls.compute_file_hash(file_bytes)
        }

        return pages_data, metadata
