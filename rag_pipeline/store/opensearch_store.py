"""
"""

from dataclasses import dataclass
import logging
from pathlib import Path
import requests
import os

logger = logging.getLogger(__name__)

@dataclass
class OpenSearchStore:

    def write(self, embeddings):
        logger.info("TODO: implement write.")


    def confirm_store_connection():
        '''
        Confirm connection to store.
        '''
        url = os.getenv("OPENSEARCH_URL")
        try:
            requests.get(url)
            logger.info("Confirmed connection to %s", url)
            return True
        except Exception as e:
            #TODO mlucas: improve exception handling
            logger.error("Failed to confirm connection to vector store at %s: \n", url, e)
        return False
