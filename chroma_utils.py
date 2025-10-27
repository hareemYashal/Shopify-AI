#!/usr/bin/env python3
"""
Shared ChromaDB client for consistent initialization across scripts
"""

import chromadb
from chromadb.config import Settings

# Shared persistent client instance
_chroma_client = None

def get_chroma_client(persist_path="./chroma_db"):
    """Get or create a shared ChromaDB client with consistent settings"""
    global _chroma_client
    
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(
            path=persist_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
    
    return _chroma_client

