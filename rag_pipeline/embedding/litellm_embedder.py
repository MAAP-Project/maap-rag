from dataclasses import dataclass
import subprocess
import sys
import os
import logging

logger = logging.getLogger(__name__)

@dataclass
class LiteLLMEmbedder:

    def embed(self, chunks):
        self._litellm_embed(chunks)


    def _litellm_embed(self, chunks):
        logger.info("TODO: not yet implmented")