"""
Document Loader Module.
Handles multi-format document ingestion (TXT, Markdown, PDF) and metadata extraction.
"""

from pathlib import Path
from typing import List, Dict, Any
import pypdf


class DocumentLoader:
    """Loads and standardizes documents into structured dictionary payloads with metadata."""

    @staticmethod
    def load_txt(file_path: Path) -> List[Dict[str, Any]]:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        return [{
            "text": text,
            "metadata": {
                "source": file_path.name,
                "file_type": "txt",
                "page": 1
            }
        }]

    @staticmethod
    def load_markdown(file_path: Path) -> List[Dict[str, Any]]:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        return [{
            "text": text,
            "metadata": {
                "source": file_path.name,
                "file_type": "markdown",
                "page": 1
            }
        }]

    @staticmethod
    def load_pdf(file_path: Path) -> List[Dict[str, Any]]:
        pages_data = []
        reader = pypdf.PdfReader(str(file_path))
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                pages_data.append({
                    "text": text,
                    "metadata": {
                        "source": file_path.name,
                        "file_type": "pdf",
                        "page": page_num
                    }
                })
        return pages_data

    @classmethod
    def load_file(cls, file_path: str | Path) -> List[Dict[str, Any]]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        suffix = path.suffix.lower()
        if suffix in [".txt", ".text"]:
            return cls.load_txt(path)
        elif suffix in [".md", ".markdown"]:
            return cls.load_markdown(path)
        elif suffix == ".pdf":
            return cls.load_pdf(path)
        else:
            # Fallback to plain text read
            return cls.load_txt(path)

