import pytest
import pymupdf
from modules.pdf_processor import PDFProcessor, PDFProcessingError

def create_sample_pdf(text: str = "This is a test document with factual content about transformers.") -> bytes:
    """Creates a temporary in-memory PDF with text."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 72), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes

def test_validate_non_pdf():
    with pytest.raises(PDFProcessingError, match="does not have a .pdf extension"):
        PDFProcessor.validate_pdf(b"data", "test.txt")

def test_validate_empty_file():
    with pytest.raises(PDFProcessingError, match="completely empty"):
        PDFProcessor.validate_pdf(b"", "empty.pdf")

def test_validate_invalid_header():
    with pytest.raises(PDFProcessingError, match="does not start with standard PDF header"):
        PDFProcessor.validate_pdf(b"NOT_A_PDF_CONTENT", "corrupt.pdf")

def test_extract_valid_pdf():
    sample_text = (
        "The Attention Is All You Need paper introduced the Transformer architecture in 2017. "
        "It replaced recurrent neural networks with multi-head self-attention mechanisms. "
        "Transformers have become the foundation for modern large language models."
    )
    pdf_bytes = create_sample_pdf(sample_text)
    pages, metadata = PDFProcessor.extract_text(pdf_bytes, "transformer_paper.pdf")

    assert len(pages) == 1
    assert pages[0]["page_number"] == 1
    assert "Transformer architecture in 2017" in pages[0]["text"]
    assert metadata["page_count"] == 1
    assert metadata["total_words"] > 10

def test_extract_scanned_or_empty_pdf():
    # Empty page with no text
    doc = pymupdf.open()
    doc.new_page()
    empty_bytes = doc.tobytes()
    doc.close()

    with pytest.raises(PDFProcessingError, match="scanned image-only PDF or empty"):
        PDFProcessor.extract_text(empty_bytes, "blank.pdf")
