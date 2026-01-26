#!/usr/bin/env python3
"""
Query Transformation and Re-ranking for RAG
================================================================================

TECHNIQUES COVERED:

1. QUERY TRANSFORMATION

2. RE-RANKING (Two-Stage Retrieval)

ENTERPRISE BENEFITS:

PREREQUISITES:
   - ChromaDB populated with OmniTech documents (run index_pdfs.py first)
   - Ollama running with llama3.2:3b model (ollama pull llama3.2:3b)

USAGE:
   python lab8.py
"""

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════════════════════

import requests
from typing import List, Dict, Tuple
from chromadb import PersistentClient
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "pdf_documents"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"

# ══════════════════════════════════════════════════════════════════════════════
# ADVANCED RAG CLASS
# ══════════════════════════════════════════════════════════════════════════════

class AdvancedRAG:
    """
    RAG system with query transformation and re-ranking capabilities.
    """

    def __init__(self):
        """Initialize the advanced RAG system."""
        print("Initializing Advanced RAG System...")

        # Connect to ChromaDB
        client = PersistentClient(
            path=CHROMA_PATH,
            settings=Settings(),
            tenant=DEFAULT_TENANT,
            database=DEFAULT_DATABASE
        )
        self.collection = client.get_collection(name=COLLECTION_NAME)
        print(f"  Connected to ChromaDB with {self.collection.count()} chunks")

    # ══════════════════════════════════════════════════════════════════════════
    # BASIC RETRIEVAL (Baseline for comparison)
    # ══════════════════════════════════════════════════════════════════════════

    def basic_retrieve(self, query: str, k: int = 3) -> List[Dict]:
        """
        Basic vector similarity search - our baseline.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )

        chunks = []
        if results['documents'] and len(results['documents'][0]) > 0:
            for i in range(len(results['documents'][0])):
                chunks.append({
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "score": 1.0 / (1.0 + results['distances'][0][i]),
                    "method": "basic"
                })
        return chunks

    # ══════════════════════════════════════════════════════════════════════════
    # QUERY TRANSFORMATION TECHNIQUES
    # ══════════════════════════════════════════════════════════════════════════

    # ──────────────────────────────────────────────────────────────────────────
    # TECHNIQUE 1: Query Expansion
    # ──────────────────────────────────────────────────────────────────────────

    def expand_query(self, query: str) -> str:
        """
        Expand the query with synonyms and related terms.
        """

        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": expansion_prompt,
                    "stream": False,
                    "options": {"temperature": 0.3}
                },
                timeout=60
            )
            if response.status_code == 200:
                expanded = response.json().get("response", "").strip()
                # Clean up - remove any explanatory text
                expanded = expanded.split('\n')[0]  # Take first line only
                return expanded if expanded else query
            return query
        except Exception as e:
            print(f"  Query expansion error: {e}")
            return query

    # ──────────────────────────────────────────────────────────────────────────
    # TECHNIQUE 2: Multi-Query Generation
    # ──────────────────────────────────────────────────────────────────────────

    def generate_multi_queries(self, query: str, n: int = 3) -> List[str]:
        """
        Generate multiple query variations from a single user query.
        """

        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": multi_query_prompt,
                    "stream": False,
                    "options": {"temperature": 0.5}
                },
                timeout=60
            )
            if response.status_code == 200:
                result = response.json().get("response", "").strip()
                queries = [q.strip() for q in result.split('\n') if q.strip()]
                # Always include original query
                if query not in queries:
                    queries.insert(0, query)
                return queries[:n]
            return [query]
        except Exception as e:
            print(f"  Multi-query generation error: {e}")
            return [query]

    # ──────────────────────────────────────────────────────────────────────────
    # TECHNIQUE 3: HyDE (Hypothetical Document Embedding)
    # ──────────────────────────────────────────────────────────────────────────

    def generate_hypothetical_answer(self, query: str) -> str:
        """
        Generate a hypothetical ideal answer (HyDE technique).
        """

        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": hyde_prompt,
                    "stream": False,
                    "options": {"temperature": 0.3}
                },
                timeout=60
            )
            if response.status_code == 200:
                return response.json().get("response", "").strip()
            return query
        except Exception as e:
            print(f"  HyDE generation error: {e}")
            return query

    # ══════════════════════════════════════════════════════════════════════════
    # RE-RANKING
    # ══════════════════════════════════════════════════════════════════════════

    def rerank_chunks(self, query: str, chunks: List[Dict], top_k: int = 3) -> List[Dict]:
        """
        Re-rank retrieved chunks using LLM-based relevance scoring.
        """
        if not chunks:
            return []

        # Score each chunk's relevance to the query
        scored_chunks = []
        for chunk in chunks:
            score = self._score_relevance(query, chunk['content'])
            chunk_copy = chunk.copy()
            chunk_copy['rerank_score'] = score
            scored_chunks.append(chunk_copy)

        # Sort by rerank score (descending)
        scored_chunks.sort(key=lambda x: x['rerank_score'], reverse=True)

        # Return top_k
        return scored_chunks[:top_k]

    def _score_relevance(self, query: str, document: str) -> float:
        """
        Score document relevance to query using LLM.
        """

        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": score_prompt,
                    "stream": False,
                    "options": {"temperature": 0.1}
                },
                timeout=30
            )
            if response.status_code == 200:
                result = response.json().get("response", "").strip()
                for char in result:
                    if char.isdigit():
                        return int(char) / 5.0  # Normalize to 0-1
                return 0.5
            return 0.0
        except Exception:
            return 0.0

    # ══════════════════════════════════════════════════════════════════════════
    # ADVANCED RETRIEVAL PIPELINES
    # ══════════════════════════════════════════════════════════════════════════

    def retrieve_with_expansion(self, query: str, k: int = 3) -> List[Dict]:
        """
        Retrieve using query expansion.
        """
        expanded_query = self.expand_query(query)
        print(f"    Expanded: '{query}' → '{expanded_query}'")
        chunks = self.basic_retrieve(expanded_query, k)
        for c in chunks:
            c['method'] = 'expansion'
        return chunks

    def retrieve_with_multi_query(self, query: str, k: int = 3) -> List[Dict]:
        """
        Retrieve using multiple query variations.
        """
        queries = self.generate_multi_queries(query, n=3)
        print(f"    Queries: {queries}")

        # Collect results from all queries
        all_chunks = {}  # Use dict to deduplicate by content hash
        for q in queries:
            results = self.basic_retrieve(q, k=k)
            for chunk in results:
                # Use first 100 chars as key for deduplication
                key = chunk['content'][:100]
                if key not in all_chunks:
                    chunk['method'] = 'multi_query'
                    all_chunks[key] = chunk
                else:
                    # Boost score for chunks found by multiple queries
                    all_chunks[key]['score'] = min(1.0, all_chunks[key]['score'] + 0.1)

        # Sort by score and return top k
        sorted_chunks = sorted(all_chunks.values(), key=lambda x: x['score'], reverse=True)
        return sorted_chunks[:k]

    def retrieve_with_hyde(self, query: str, k: int = 3) -> List[Dict]:
        """
        Retrieve using HyDE (Hypothetical Document Embedding).
        """
        hypothetical = self.generate_hypothetical_answer(query)
        print(f"    HyDE: '{query}' → '{hypothetical[:80]}...'")
        chunks = self.basic_retrieve(hypothetical, k)
        for c in chunks:
            c['method'] = 'hyde'
        return chunks

    def retrieve_with_reranking(self, query: str, k: int = 3) -> List[Dict]:
        """
        Retrieve using two-stage retrieval with re-ranking.
        """
        # Stage 1: Fast retrieval (get more candidates)
        candidates = self.basic_retrieve(query, k=k*2)
        print(f"    Stage 1: Retrieved {len(candidates)} candidates")

        # Stage 2: Re-rank
        reranked = self.rerank_chunks(query, candidates, top_k=k)
        print(f"    Stage 2: Re-ranked to top {len(reranked)}")
        for c in reranked:
            c['method'] = 'reranked'
        return reranked

    def retrieve_advanced(self, query: str, k: int = 3) -> List[Dict]:
        """
        Full advanced retrieval pipeline combining all techniques.
        """
        print("  Running full advanced pipeline...")

        # Step 1: Generate query variations
        queries = self.generate_multi_queries(query, n=2)

        # Step 2: Collect candidates from multiple approaches
        all_chunks = {}

        # Original query - basic
        for chunk in self.basic_retrieve(query, k=k):
            key = chunk['content'][:100]
            all_chunks[key] = chunk

        # Query variations
        for q in queries:
            for chunk in self.basic_retrieve(q, k=k):
                key = chunk['content'][:100]
                if key not in all_chunks:
                    all_chunks[key] = chunk

        # HyDE
        hypothetical = self.generate_hypothetical_answer(query)
        for chunk in self.basic_retrieve(hypothetical, k=k):
            key = chunk['content'][:100]
            if key not in all_chunks:
                all_chunks[key] = chunk

        print(f"    Collected {len(all_chunks)} unique candidates")

        # Step 3: Re-rank all candidates
        candidates = list(all_chunks.values())
        reranked = self.rerank_chunks(query, candidates, top_k=k)
        for c in reranked:
            c['method'] = 'advanced'

        return reranked

    # ══════════════════════════════════════════════════════════════════════════
    # ANSWER GENERATION
    # ══════════════════════════════════════════════════════════════════════════

    def generate_answer(self, question: str, chunks: List[Dict]) -> str:
        """Generate an answer from retrieved chunks."""
        if not chunks:
            return "I couldn't find relevant information to answer your question."

        context = "\n\n".join([
            f"[{c['metadata'].get('source', 'doc')}]\n{c['content']}"
            for c in chunks
        ])

        prompt = f"""Based on the following context, answer the question concisely.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3}
                },
                timeout=120
            )
            if response.status_code == 200:
                return response.json().get("response", "").strip()
            return f"Error: {response.status_code}"
        except Exception as e:
            return f"Error: {e}"

    # ══════════════════════════════════════════════════════════════════════════
    # COMPARISON DEMO
    # ══════════════════════════════════════════════════════════════════════════

    def compare_methods(self, query: str) -> Dict:
        """
        Compare all retrieval methods side-by-side.
        """
        print(f"\n{'='*60}")
        print(f"Comparing retrieval methods for:")
        print(f"  '{query}'")
        print('='*60)

        results = {}

        # Basic retrieval
        print("\n[1/5] Basic Retrieval...")
        basic_chunks = self.basic_retrieve(query)
        results['basic'] = {
            'chunks': basic_chunks,
            'answer': self.generate_answer(query, basic_chunks)
        }

        # Query expansion
        print("\n[2/5] Query Expansion...")
        expansion_chunks = self.retrieve_with_expansion(query)
        results['expansion'] = {
            'chunks': expansion_chunks,
            'answer': self.generate_answer(query, expansion_chunks)
        }

        # Multi-query
        print("\n[3/5] Multi-Query...")
        multi_chunks = self.retrieve_with_multi_query(query)
        results['multi_query'] = {
            'chunks': multi_chunks,
            'answer': self.generate_answer(query, multi_chunks)
        }

        # HyDE
        print("\n[4/5] HyDE...")
        hyde_chunks = self.retrieve_with_hyde(query)
        results['hyde'] = {
            'chunks': hyde_chunks,
            'answer': self.generate_answer(query, hyde_chunks)
        }

        # Re-ranking
        print("\n[5/5] Re-ranking...")
        rerank_chunks = self.retrieve_with_reranking(query)
        results['reranking'] = {
            'chunks': rerank_chunks,
            'answer': self.generate_answer(query, rerank_chunks)
        }

        return results


# ══════════════════════════════════════════════════════════════════════════════
# TERMINAL COLORS
# ══════════════════════════════════════════════════════════════════════════════

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
RESET = "\033[0m"
BOLD = "\033[1m"

METHOD_COLORS = {
    'basic': RED,
    'expansion': YELLOW,
    'multi_query': GREEN,
    'hyde': CYAN,
    'reranking': MAGENTA,
    'advanced': BLUE
}


# ══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE DEMO
# ══════════════════════════════════════════════════════════════════════════════

def run_demo():
    """
    Run interactive demo comparing query transformation and re-ranking.
    """
    print(f"\n{BOLD}Advanced RAG: Query Transformation & Re-ranking{RESET}")
    print("="*55)
    print("Compare different retrieval enhancement techniques")
    print("="*55)

    rag = AdvancedRAG()

    print(f"\n{BOLD}Methods:{RESET}")
    print(f"  {RED}BASIC{RESET}      - Standard vector search (baseline)")
    print(f"  {YELLOW}EXPANSION{RESET}  - Query expanded with synonyms")
    print(f"  {GREEN}MULTI-Q{RESET}    - Multiple query variations")
    print(f"  {CYAN}HYDE{RESET}       - Hypothetical document embedding")
    print(f"  {MAGENTA}RERANK{RESET}     - Two-stage with re-ranking")

    while True:
        print(f"\n{BOLD}Options:{RESET}")
        print("  1. Compare all methods on a query")
        print("  2. Try individual technique")
        print("  3. Exit")

        choice = input(f"\n{BOLD}Choose (1-3):{RESET} ").strip()

        if choice == "1":
            query = input(f"\n{BOLD}Enter query:{RESET} ").strip()
            if not query:
                continue

            results = rag.compare_methods(query)

            # Display comparison
            print(f"\n{'='*60}")
            print(f"{BOLD}COMPARISON RESULTS{RESET}")
            print('='*60)

            for method, data in results.items():
                color = METHOD_COLORS.get(method, RESET)
                print(f"\n{color}{BOLD}[{method.upper()}]{RESET}")

                # Show retrieved sources
                sources = [c['metadata'].get('source', 'unknown').split('/')[-1]
                          for c in data['chunks'][:2]]
                print(f"  Sources: {', '.join(sources)}")

                # Show answer preview
                answer = data['answer'][:200] + "..." if len(data['answer']) > 200 else data['answer']
                print(f"  Answer: {answer}")

        elif choice == "2":
            print(f"\n{BOLD}Techniques:{RESET}")
            print("  1. Query Expansion")
            print("  2. Multi-Query")
            print("  3. HyDE")
            print("  4. Re-ranking")

            tech = input(f"\n{BOLD}Choose technique (1-4):{RESET} ").strip()
            query = input(f"{BOLD}Enter query:{RESET} ").strip()

            if not query:
                continue

            if tech == "1":
                print(f"\n{YELLOW}{BOLD}Query Expansion:{RESET}")
                chunks = rag.retrieve_with_expansion(query)
                answer = rag.generate_answer(query, chunks)
            elif tech == "2":
                print(f"\n{GREEN}{BOLD}Multi-Query:{RESET}")
                chunks = rag.retrieve_with_multi_query(query)
                answer = rag.generate_answer(query, chunks)
            elif tech == "3":
                print(f"\n{CYAN}{BOLD}HyDE:{RESET}")
                chunks = rag.retrieve_with_hyde(query)
                answer = rag.generate_answer(query, chunks)
            elif tech == "4":
                print(f"\n{MAGENTA}{BOLD}Re-ranking:{RESET}")
                chunks = rag.retrieve_with_reranking(query)
                answer = rag.generate_answer(query, chunks)
            else:
                print("Invalid choice")
                continue

            print(f"\n{BOLD}Retrieved chunks:{RESET}")
            for i, c in enumerate(chunks[:3]):
                print(f"  [{i+1}] {c['metadata'].get('source', 'unknown').split('/')[-1]} (score: {c['score']:.3f})")

            print(f"\n{BOLD}Answer:{RESET}")
            print(f"  {answer}")

        elif choice == "3":
            print("\nGoodbye!")
            break

        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_demo()
