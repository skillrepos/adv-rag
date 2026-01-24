# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based educational workshop on Retrieval-Augmented Generation (RAG). It teaches how to build RAG systems combining vector databases (ChromaDB), graph databases (Neo4j), and local LLMs (Ollama with Llama 3.2) to create grounded AI applications.

## Technology Stack

- **Language:** Python 3
- **Vector Database:** ChromaDB (persistent, disk-based)
- **Graph Database:** Neo4j (Docker-based)
- **LLM Runtime:** Ollama (local inference at `http://localhost:11434`)
- **LLM Model:** Llama 3.2 (1b or 3b variants)
- **Embeddings:** Sentence-Transformers (`all-MiniLM-L6-v2`, 384-dimensional)
- **Framework:** LangChain for simplified RAG patterns

## Common Commands

### Environment Setup
```bash
# Python virtual environment setup
source scripts/pysetup.sh py_env

# Start Ollama and pull model
./scripts/startup_ollama.sh

# Set up Neo4j (choose dataset 1, 2, or 3)
cd neo4j && ./neo4j-setup.sh 3
```

### Indexing Documents
```bash
# Index PDF documents into ChromaDB
python tools/index_pdfs.py --pdf-dir ./data/knowledge_base_pdfs --chroma-path ./chroma_db

# Index source code into ChromaDB
python tools/index_code.py --code-dir ../ --chroma-path ./chroma_code_db
```

### Running Labs and Tools
```bash
# Interactive semantic search
python tools/search.py --target pdfs
python tools/search.py --target code

# Search with specific query
python tools/search.py --query "return policy" --target pdfs --top-k 5

# Run lab exercises
python code/lab1.py
python code/lab4.py
python code/lab6.py
python code/rag_code.py
```

## Architecture

### RAG Pipeline Pattern

The system implements a three-step RAG pattern:

1. **RETRIEVE**: Vector similarity search in ChromaDB using semantic embeddings
2. **AUGMENT**: Combine user question with retrieved context into an augmented prompt
3. **GENERATE**: Send to Ollama API for grounded, context-aware generation

### Multi-Database Strategy

- **ChromaDB** stores two separate collections:
  - `pdf_documents`: Document chunks from OmniTech PDFs in `./chroma_db`
  - `code_index`: Source code chunks in `./chroma_code_db`

- **Neo4j** stores structured relationships with three configurable datasets:
  - `data1/`: Person relationships (Lab 4)
  - `data2/`: Movie database (Lab 5)
  - `data3/`: OmniTech policies and relationships (Lab 6)

### Hybrid RAG (Lab 6)

Combines semantic search (ChromaDB) with graph queries (Neo4j) for comprehensive context retrieval.

## Key Files

| Path | Purpose |
|------|---------|
| `tools/index_pdfs.py` | Create ChromaDB index from PDFs (800-char chunks, 200-char overlap) |
| `tools/index_code.py` | Create ChromaDB index from source code (token-aware chunking) |
| `tools/search.py` | Semantic search tool for both databases |
| `code/rag_code.py` | Complete RAG system implementation |
| `code/lab4.py` | Graph RAG with Neo4j |
| `code/lab6.py` | Hybrid RAG combining semantic + graph search |
| `neo4j/neo4j-setup.sh` | Shell script to set up Neo4j Docker containers |

## Service Endpoints

- **Ollama API:** `http://localhost:11434/api/generate`
- **Neo4j Bolt:** `neo4j://localhost:7687`
- **Neo4j Browser:** `http://localhost:7474`
- **Neo4j Credentials:** `neo4j` / `neo4jtest` (development only)

## Critical Constraints

- **Embedding model consistency:** The embedding model (`all-MiniLM-L6-v2`) must match between indexing and search - changing it invalidates existing vectors
- **Ollama must be running:** LLM generation fails if Ollama service isn't started
- **ChromaDB uses PersistentClient:** Data survives session restarts at specified paths
- **Token-aware chunking:** Code indexer respects LLM context limits via tiktoken

## Workshop Labs

1. **Lab 1:** Basic prompt augmentation with static context
2. **Lab 2:** Vector database setup with ChromaDB
3. **Lab 3:** Complete RAG pipeline
4. **Lab 4:** Graph RAG with Neo4j Cypher queries
5. **Lab 5:** Simplified Graph RAG using LangChain
6. **Lab 6:** Hybrid RAG combining semantic + graph retrieval

See `labs.md` for detailed step-by-step instructions.
