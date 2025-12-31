"""
RAG pipeline orchestration module.

This module defines the Pipeline class, which coordinates the high-level
execution of a Retrieval-Augmented Generation (RAG) workflow. The pipeline
is responsible for invoking the following stages:

1. Localization: retrieve the input data source (e.g., clone a Git repo).
2. Chunking: parse and segment documents into semantically meaningful chunks.
3. Embedding: generate vector embeddings for the produced chunks.

"""

from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class Pipeline:
    """
    Orchestrates execution of a Retrieval-Augmented Generation (RAG) pipeline.

    Attributes:
        localizer: Component responsible for localizing and preparing the input data. 
            The only method currently supported is cloning a Git repository.
        chunker: Component responsible for chunking documents into
            embedding-ready text segments. The only chunking strategy currently supported
            is for Jupyter Notebooks.
        embedder: Component responsible for generating vector embeddings
            from document chunks.
    """

    localizer: any
    chunker: any
    embedder: any

    def run(self):
        try:
            logger.info("Begin execution of RAG pipeline.")
            self.localizer.localize()
            chunks = self.chunker.chunk()
            self.embedder.embed(chunks)
        except:
            logger.exception("Failed RAG pipeline execution.")
        finally:
            logger.info("End execution of RAG pipeline.")
            return
