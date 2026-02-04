"""Vector store management service for chatbot RAG."""

import json
import unicodedata
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from src.db.utility import load_json_file
from src.config.settings import get_settings


# Data file paths
BASE_DIR = Path(__file__).resolve().parent.parent
PROFESSORS_FILE_PATH = BASE_DIR / "data" / "professors.json"
UNIN_DATA_FILE_PATH = BASE_DIR / "data" / "unin_data.json"


def normalize_text(text: str) -> str:
    """
    Normalize text by removing diacritics and converting to ASCII.
    Handles Croatian characters like č, ć, š, ž, đ.

    Args:
        text: Text to normalize

    Returns:
        Normalized ASCII text
    """
    if not text:
        return ""
    # Normalize to NFD (decomposed form) and filter out combining characters
    nfd = unicodedata.normalize('NFD', text)
    return ''.join(char for char in nfd if unicodedata.category(char) != 'Mn')


def parse_professor_name(name: str) -> tuple[str, str]:
    """
    Parse professor name from "Last First" format to first and last names.
    Handles cases like "Ivančić Valenko Snježana" -> ("Snježana", "Ivančić Valenko")

    Args:
        name: Full name in "Last First" or "Last1 Last2 First" format

    Returns:
        Tuple of (first_name, last_name)
    """
    if not name:
        return "", ""

    parts = name.strip().split()
    if len(parts) == 0:
        return "", ""
    elif len(parts) == 1:
        return parts[0], parts[0]
    else:
        # Last part is first name, everything else is last name
        first_name = parts[-1]
        last_name = " ".join(parts[:-1])
        return first_name, last_name


def create_professor_documents_text(prof: dict) -> List[str]:
    """
    Create multiple searchable text variations for a professor.
    Each variation emphasizes different name formats for better semantic search.

    Args:
        prof: Professor dictionary

    Returns:
        List of formatted text content, one for each name variation
    """
    name = prof.get("name", "")
    first_name, last_name = parse_professor_name(name)

    # Create name variations
    name_variations = [
        name,  # Original: "Ivančić Valenko Snježana"
        f"{first_name} {last_name}",  # First Last: "Snježana Ivančić Valenko"
        normalize_text(name),  # Normalized: "Ivancic Valenko Snjezana"
        f"{normalize_text(first_name)} {normalize_text(last_name)}",  # Normalized First Last: "Snjezana Ivancic Valenko"
        last_name,  # Just last name: "Ivančić Valenko"
        normalize_text(last_name),  # Normalized last name: "Ivancic Valenko"
    ]

    # Remove duplicates and empty strings
    name_variations = [n for n in set(name_variations) if n.strip()]

    # Build contact info text (same for all variations)
    contact_parts = []
    if prof.get("title"):
        contact_parts.append(f"Title: {prof.get('title')}")
    if prof.get("email"):
        contact_parts.append(f"Email: {prof['email']}")
    if prof.get("phone"):
        contact_parts.append(f"Phone: {prof['phone']}")
    if prof.get("room"):
        contact_parts.append(f"Room: {prof['room']}")
        if prof.get("room_route"):
            contact_parts.append(f"Building: {prof['room_route']}")
    if prof.get("web"):
        contact_parts.append(f"Website: {prof['web']}")

    contact_info = "\n".join(contact_parts)

    # Create a document for each name variation
    documents = []
    for name_var in name_variations:
        text_parts = [
            f"Professor: {name_var}",
            f"Full name: {first_name} {last_name}",
            contact_info,
            f"\nOriginal data: {json.dumps(prof)}"
        ]
        documents.append("\n".join(text_parts))

    return documents


def create_room_document_text(room: dict) -> str:
    """
    Create searchable text content for a room document.

    Args:
        room: Room dictionary

    Returns:
        Formatted text content
    """
    text_parts = []

    # Add room number/name with variations
    if room.get("name"):
        text_parts.append(f"Room: {room['name']}")
        text_parts.append(f"Room name: {room['name']}")

    if room.get("number"):
        text_parts.append(f"Room number: {room['number']}")

    # Add room type/category
    if room.get("type"):
        text_parts.append(f"Type: {room['type']}")

    # Add building information
    if room.get("building"):
        text_parts.append(f"Building: {room['building']}")

    if room.get("floor"):
        text_parts.append(f"Floor: {room['floor']}")

    # Add description if available
    if room.get("description"):
        text_parts.append(f"Description: {room['description']}")

    # Store original JSON for reference
    text_parts.append(f"\nOriginal data: {json.dumps(room)}")

    return "\n".join(text_parts)


def load_data() -> tuple[list, list]:
    """
    Load professors and room data from JSON files.

    Returns:
        Tuple of (professors_data, rooms_data)
    """
    professors_data = load_json_file(str(PROFESSORS_FILE_PATH))
    rooms_data = load_json_file(str(UNIN_DATA_FILE_PATH))
    return professors_data, rooms_data


def create_documents(professors_data: list, rooms_data: list) -> List[Document]:
    """
    Convert JSON data to LangChain Document objects.
    Creates human-readable, searchable text content instead of raw JSON.
    Creates multiple documents per professor (one per name variation) for better matching.

    Args:
        professors_data: List of professor dictionaries
        rooms_data: List of room dictionaries

    Returns:
        List of Document objects with metadata
    """
    # Create multiple documents per professor for better name matching
    professor_docs = []
    for prof in professors_data:
        doc_texts = create_professor_documents_text(prof)
        for doc_text in doc_texts:
            professor_docs.append(
                Document(
                    page_content=doc_text,
                    metadata={
                        "type": "professor",
                        "name": prof.get("name", ""),
                        "id": prof.get("id", "")
                    }
                )
            )

    # Create documents for rooms with descriptive text
    room_docs = [
        Document(
            page_content=create_room_document_text(room),
            metadata={
                "type": "room",
                "name": room.get("name", ""),
                "id": room.get("id", "")
            }
        )
        for room in rooms_data
    ]

    # Combine all documents
    all_docs = professor_docs + room_docs
    return all_docs


def initialize_vector_store() -> tuple[FAISS, List[Document]]:
    """
    Initialize FAISS vector store with HuggingFace embeddings.

    Returns:
        Tuple of (FAISS vector store, list of documents)
    """
    settings = get_settings()

    # Load data
    professors_data, rooms_data = load_data()

    # Create documents
    documents = create_documents(professors_data, rooms_data)

    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create vector store
    vector_store = FAISS.from_documents(documents, embeddings)

    return vector_store, documents
