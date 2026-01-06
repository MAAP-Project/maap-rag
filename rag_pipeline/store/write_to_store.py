"""
"""

from dataclasses import dataclass
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class Store:

    def write(self, embeddings):
        logger.info("TODO: implement write.")