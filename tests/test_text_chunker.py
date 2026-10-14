import pytest
from modules.text_chunker import TextChunker

def test_chunk_short_page():
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    pages = [
        {"page_number": 1, "text": "This is a short single sentence on page one."}
    ]
    chunks = chunker.chunk_pages(pages, "sample.pdf")

    assert len(chunks) == 1
    assert chunks[0]["page_number"] == 1
    assert chunks[0]["document_name"] == "sample.pdf"
    assert "short single sentence" in chunks[0]["text"]

def test_chunk_long_page_overlap():
    chunker = TextChunker(chunk_size=10, chunk_overlap=3)
    # 25 words
    words = [f"word{i}" for i in range(25)]
    pages = [
        {"page_number": 2, "text": " ".join(words)}
    ]
    chunks = chunker.chunk_pages(pages, "multichunk.pdf")

    # Step is 10 - 3 = 7. Word range: 0-10, 7-17, 14-24, 21-25 -> 4 chunks
    assert len(chunks) >= 3
    for c in chunks:
        assert c["page_number"] == 2
        assert c["document_name"] == "multichunk.pdf"

    # Check overlap between first two chunks
    first_chunk_words = set(chunks[0]["text"].split())
    second_chunk_words = set(chunks[1]["text"].split())
    overlap = first_chunk_words.intersection(second_chunk_words)
    assert len(overlap) >= 2
