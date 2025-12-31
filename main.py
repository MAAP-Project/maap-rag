'''
Driver script for the RAG pipeline.
'''

from rag_pipeline.core.pipeline import Pipeline
from rag_pipeline.localization.git_repo_localizer import GitRepoLocalizer
from rag_pipeline.chunking.notebook_chunker import NotebookChunker
from rag_pipeline.embedding.litellm_embedder import LiteLLMEmbedder
from rag_pipeline.core.logging import setup_logging

def main():
    setup_logging()

    pipeline = Pipeline(
        localizer=GitRepoLocalizer(branch="ogc", repo_url="https://github.com/MAAP-Project/maap-documentation.git"),
        chunker=NotebookChunker(),
        embedder=LiteLLMEmbedder(),
    )

    pipeline.run()
    # for embedding in pipeline.run():
    #     #TODO write embedding to vector store
    #     pass

if __name__ == "__main__":
    main()
