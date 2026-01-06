from dataclasses import dataclass
import os
import logging
import requests
import json
from decimal import Decimal

logger = logging.getLogger(__name__)

@dataclass
class LiteLLMEmbedder:

    model: str
    url: str

    def embed(self, chunks):
        if not chunks:
            logger.warning("No chunks to embed. Skipping embedding.")
            return

        return self._litellm_embed(chunks)


    def _get_embedding_metrics(self, headers):
        # Cost of the embedding request (USD)
        cost = Decimal(headers.get("x-litellm-response-cost"))

        # Duration of request (s)
        duration = float(headers.get("x-litellm-response-duration-ms")) / 1000

        # Cumulative cost associated with user token (USD)
        user_cost = Decimal(headers.get("x-litellm-key-spend"))

        logger.info(
                "\n\n"
                "Embedding metrics\n"
                "-----------------\n"
                "Duration of embedding:                  %.6f s\n"
                "Cost of embedding:                     $%.6f\n"
                "Cumulative cost associated with token: $%.6f\n\n",
                cost,
                duration,
                user_cost,
)

    def _litellm_embed(self, chunks):
        # TODO mlucas: update to do asynchronous batch chunking for better performance
        # TODO: couple embedding to original chunk

        token = os.getenv("LITELLM_TOKEN")
        
        # Extract text to embed
        texts = [chunk.page_content for chunk in chunks]

        #TODO mlucas: for testing, use one chunk otherwise this will get expensive
        payload = json.dumps({
            "model": self.model,
            "input": [
                texts[0]
            ]
        })

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        response = requests.post(self.url, headers=headers, data=payload)
        response.raise_for_status()
        data = response.json()
        vectors = [d["embedding"] for d in data["data"]]

        self._get_embedding_metrics(response.headers)

        return vectors

