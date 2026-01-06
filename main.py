'''
Driver script for the RAG pipeline.
'''

from rag_pipeline.core.pipeline import Pipeline
from rag_pipeline.localization.git_repo_localizer import GitRepoLocalizer
from rag_pipeline.chunking.notebook_chunker import NotebookChunker
from rag_pipeline.embedding.litellm_embedder import LiteLLMEmbedder
from rag_pipeline.store.write_to_store import Store
from rag_pipeline.core.logging import setup_logging
from dotenv import load_dotenv


def main():
    setup_logging()

    pipeline = Pipeline(
        localizer=GitRepoLocalizer(branch="ogc", repo_url="https://github.com/MAAP-Project/maap-documentation.git"),
        chunker=NotebookChunker(),
        embedder=LiteLLMEmbedder(url="https://litellm.maap.xyz/api/v1/embeddings", model="amazon.titan-embed-text-v2:0"),
        store=Store()
    )

    pipeline.run()


if __name__ == "__main__":
    load_dotenv()
    main()
