"""
Jupyter Notebook chunking module.

This module defines the NotebookChunker class, which is responsible for
chunking Jupyter Notebooks into semantically meaningful units.

Behavior:
- Recursively searches the input data directory for all Jupyter Notebook files to chunk.
- For each Jupyter Notebook file found:
    - Loads all notebook cells, collecting cell metadata in addition to cell content.
    - Creates sections to group cells that are semantically similar (improves context for LLM).
    - Prepares chunks for embedding by creating langchain Documents.
- Logs all operations and surfaces errors with full stack traces.
"""

import nbformat
from pathlib import Path
import re
import logging
import uuid
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)", re.MULTILINE)

class NotebookChunker:

    def chunk(self):
        return self._chunk_notebooks()


    def _chunk_notebooks(self):
        chunks = []
        notebook_paths = self._get_notebook_paths()
        for notebook_path in notebook_paths:
            chunks.extend(self._chunk_notebook(notebook_path))
        return chunks


    def _get_notebook_paths(self):
        logger.info("Compiling list of notebooks found in '%s' directory...", 'input')
        notebook_paths = [str(p) for p in Path(".").rglob("*.ipynb")]
        logger.info("Found %s notebooks to chunk.", len(notebook_paths))
        return notebook_paths


    def _chunk_notebook(self, notebook_path):
        #TODO: mlucas metrics
        logger.info("Chunking notebook '%s' ...", notebook_path)
        cells = self._load_notebook_cells(notebook_path)
        sections = self._group_by_sections(cells)
        docs = self._sections_to_documents(sections, notebook_path)
        final_docs = self._split_documents(docs)
        return final_docs


    def _load_notebook_cells(self, path: Path):
        nb = nbformat.read(path, as_version=4)
        cells = []

        for idx, cell in enumerate(nb.cells):
            if cell.cell_type == "markdown":
                cells.append({
                    "type": "markdown",
                    "text": cell.source,
                    "index": idx
                })
            # TODO: mlucas Code cells may be particularly useful for BM25/keyword searches.
            elif cell.cell_type == "code":
                cells.append({
                    "type": "code",
                    "text": cell.source,
                    "index": idx
                })
        return cells


    def _group_by_sections(self, cells):
        """
        Groups cells by section to impove context for LLM. 
        
        Sections are created for each heading. If a cell contains a heading, a 
        new section is created and all following cells that contain no headings 
        will be added to this section. This strategy assumes that cells under
        the same headings are semantically similar.

        TODO: mlucas Will likely have to play around with this to see what level
        of heading provides the context that returns the most relevant results
        when used with the LLM. Or perhaps an alternative grouping strategy
        entirely.
        """
        sections = []
        current = {
            "heading": "ROOT",
            "level": 0,
            "cells": []
        }
        ref_current = current

        for cell in cells:
            if cell["type"] == "markdown":
                match = HEADING_RE.match(cell["text"].lstrip())
                if match:
                    sections.append(current)
                    current = {
                        "heading": match.group(2),
                        "cells": []
                    }
                    # Add current cell to newly-created section
                    current["cells"].append(cell)
                    ref_current = current
                    continue
                else:
                    # Cell has no heading so add it to current section
                    ref_current["cells"].append(cell)
        return sections
    

    def _sections_to_documents(self, sections, notebook_path):
        '''
        Create chunks for embedding.

        TODO: mlucas Metadata provided may help with filtering i.e. user queries LLM
        'show me the latest docs on topic X'
        '''
        docs = []

        for section in sections:
            markdown_parts = []
            code_parts = []

            for cell in section["cells"]:
                if cell["type"] == "markdown":
                    markdown_parts.append(cell["text"])
                elif cell["type"] == "code":
                    code_parts.append(cell["text"])

            if markdown_parts:
                docs.append(
                    Document(
                        page_content="\n\n".join(markdown_parts),
                        metadata={
                            "chunk_id": str(uuid.uuid4()),
                            "source": str(notebook_path),
                            "section": section["heading"],
                            "type": "markdown"
                        }
                    )
                )

            if code_parts:
                docs.append(
                    Document(
                        page_content="\n\n".join(code_parts),
                        metadata={
                            "chunk_id": str(uuid.uuid4()),
                            "source": str(notebook_path),
                            "section": section["heading"],
                            "type": "code",
                            "language": "python"
                        }
                    )
                )
        return docs


    def _split_documents(self, docs):
        '''
        If chunks generated are too large, split them.

        TODO: mlucas play with max chunk size and overlap. Rule-of-thumb for
        chunk overlap is ~15% of max chunk size.
        '''
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )
        final_docs = []
        for doc in docs:
            # Only split if needed
            if len(doc.page_content) > 1000:
                # TODO mlucas: add chunk_id to newly-created chunks
                final_docs.extend(splitter.split_documents([doc]))
            else:
                final_docs.append(doc)

        return final_docs