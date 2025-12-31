"""
Git repository localization module.

This module defines the GitRepoLocalizer class, which is responsible for
localizing a remote Git repository on the local filesystem for use as
input to the Retrieval-Augmented Generation (RAG) pipeline.

Behavior:
- Creates (if necessary) and switches into a dedicated input directory.
- Clones the target repository if it does not exist locally.
- If the repository exists, fetches, checks out, and hard-resets to the
  specified branch.
- Logs all operations and surfaces errors with full stack traces.
"""

from dataclasses import dataclass
import subprocess
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class GitRepoLocalizer:
    branch: str
    repo_url: str


    def localize(self):
        self._clone_repo()


    def _clone_repo(self):
        self._create_input_dir()

        repo_name = self.repo_url.rstrip("/").split("/")[-1].replace(".git", "")
        repo_path = Path(os.getcwd()) / repo_name

        if not repo_path.exists():
            try:
                command = ["git", "clone", "--branch", self.branch, self.repo_url]
                logger.info("Cloning repo %s (branch=%s) into '%s'", self.repo_url, self.branch, os.getcwd())
                subprocess.check_call(command)
            except subprocess.CalledProcessError:
                logger.exception("Failed to clone git repo '%s', branch=%s to '%s'", self.repo_url, self.branch, os.getcwd())
                raise
        else:
            try:
                logger.info("Git repo %s (branch=%s) already exists at '%s'. Checking out latest changes.", self.repo_url, self.branch, os.getcwd())
                subprocess.check_call(["git", "fetch", "origin", self.branch], cwd=repo_path)
                subprocess.check_call(["git", "checkout", self.branch], cwd=repo_path)
                subprocess.check_call(["git", "reset", "--hard", f"origin/{self.branch}"], cwd=repo_path)
            except subprocess.CalledProcessError:
                logger.exception("Failed to update git repo '%s', branch=%s at '%s'", self.repo_url, self.branch, os.getcwd())
                raise


    def _create_input_dir(self):
        if not os.path.exists('input'):
            os.makedirs('input')
        os.chdir('input')
        logger.info("Created directory for RAG input data: '%s'. Current working directory: '%s'", 'input', os.getcwd())