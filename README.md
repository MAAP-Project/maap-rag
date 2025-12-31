# MAAP Retrieval-Augmented Generation (RAG)

Work in progress.

This project contains modules for building a MAAP RAG. The RAG may be used by LLM's to retrieve additional project context, improving accuracy and relevancy of responses. 

This RAG workflow is composed of the following stages:
- Localizing – bring source content into a local workspace
- Chunking – normalize & break source content into semantically similar chunks
- Embedding – convert chunks into numeric vectors
- Storing – persist embeddings in a vector store for retrieval by an LLM

These stages are orchestrated by a single pipeline module. The pipeline is indempotent, allowing for harmless reprocessing of RAG input data.
The pipeline may also be executed over different types of source data, different chunking strategies, and different embedding models. Support 
for each of these stages is limited, currently.

See `main.py` for a minimal driver script that demonstrates:
 - pipeline construction
 - pipeline execution

