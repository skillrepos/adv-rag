#!/usr/bin/env python3
"""
Corrective RAG (CRAG) - Self-Correcting Retrieval-Augmented Generation
================================================================================

WHAT IS CRAG?

THE CRAG WORKFLOW:

WHY CRAG MATTERS FOR ENTERPRISE:

PREREQUISITES:
   - ChromaDB populated with OmniTech documents (run index_pdfs.py first)
   - Ollama running with llama3.2:3b model (ollama pull llama3.2:3b)

USAGE:
   python lab9.py
"""

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════════════════════

import os
import requests
from typing import List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum
from chromadb import PersistentClient
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "pdf_documents"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

# CRAG thresholds - tune these based on your quality requirements
RELEVANCE_THRESHOLD_HIGH = 0.7   # Above this = CORRECT (use retrieved docs)
RELEVANCE_THRESHOLD_LOW = 0.5   # Below this = INCORRECT (use web search only)
                                 # Between = AMBIGUOUS (refine docs + supplement)


# ══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

class RetrievalDecision(Enum):
    """
    CRAG decision categories based on retrieval quality evaluation.
    """
    CORRECT = "correct"
    INCORRECT = "incorrect"
    AMBIGUOUS = "ambiguous"


@dataclass
class CRAGResult:
    """
    Container for CRAG pipeline results with full audit trail.
    """
    question: str
    answer: str

    # Retrieval info
    initial_chunks: List[Dict]
    relevance_scores: List[float]
    decision: RetrievalDecision

    # Corrective actions taken
    web_search_used: bool = False
    web_results: List[Dict] = None
    refined_knowledge: str = ""

    # Final context used
    final_context: List[Dict] = None


# ══════════════════════════════════════════════════════════════════════════════
# CORRECTIVE RAG CLASS
# ══════════════════════════════════════════════════════════════════════════════

class CorrectiveRAG:
    """
    Corrective RAG (CRAG) implementation with self-correction capabilities.
    """

    def __init__(self):
        """Initialize the CRAG system."""
        print("Initializing Corrective RAG System...")

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
    # STEP 1: RETRIEVAL
    # ══════════════════════════════════════════════════════════════════════════

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """
        Retrieve candidate documents from vector store.
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
                    "score": 1.0 / (1.0 + results['distances'][0][i])
                })
        return chunks

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2: RETRIEVAL EVALUATION (The "C" in CRAG)
    # ══════════════════════════════════════════════════════════════════════════

    def evaluate_relevance(self, query: str, document: str) -> Tuple[float, str]:
        """
        Evaluate how relevant a single document is to the query.
        """

        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": eval_prompt,
                    "stream": False,
                    "options": {"temperature": 0.1}
                },
                timeout=30
            )
            if response.status_code == 200:
                result = response.json().get("response", "").strip()
                
                # Extract reasoning
                reasoning = "No reasoning provided"
                if "REASONING:" in result:
                    reasoning_part = result.split("REASONING:")[1].split("SCORE:")[0].strip()
                    reasoning = reasoning_part
                
                # Robust extraction using bracketed format for 1-10
                import re
                match = re.search(r'\[\[([0-9]|10)\]\]', result)
                if match:
                    return int(match.group(1)) / 10.0, reasoning
                
                # Fallback: look for any digit if the model ignored the brackets
                matches = re.findall(r'\d+', result)
                if matches:
                    score = int(matches[-1]) # Take last found digit (often the score)
                    if score > 10: score = 10
                    return score / 10.0, reasoning
                    
                return 0.2, reasoning
            return 0.0, "Connection error"
        except Exception as e:
            return 0.0, f"Error: {str(e)}"

    def evaluate_all_documents(self, query: str, chunks: List[Dict]) -> Tuple[List[float], List[str]]:
        """
        Evaluate relevance of all retrieved documents.

        Returns (list of relevance scores, list of reasoning strings).
        """
        scores = []
        reasonings = []
        for chunk in chunks:
            score, reasoning = self.evaluate_relevance(query, chunk['content'])
            scores.append(score)
            reasonings.append(reasoning)
        return scores, reasonings

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 3: DECISION MAKING
    # ══════════════════════════════════════════════════════════════════════════

    def make_decision(self, relevance_scores: List[float]) -> RetrievalDecision:
        """
        Decide corrective action based on document relevance scores.
        """
        if not relevance_scores:
            return RetrievalDecision.INCORRECT

        max_score = max(relevance_scores)
        avg_score = sum(relevance_scores) / len(relevance_scores)

        # Decision logic
        if max_score >= RELEVANCE_THRESHOLD_HIGH:
            # At least one highly relevant document - use retrieved docs
            return RetrievalDecision.CORRECT
        elif max_score < RELEVANCE_THRESHOLD_LOW:
            # No relevant documents - need external search
            return RetrievalDecision.INCORRECT
        else:
            # Partial relevance - refine and supplement
            return RetrievalDecision.AMBIGUOUS

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 4: CORRECTIVE ACTIONS
    # ══════════════════════════════════════════════════════════════════════════

    def simulate_web_search(self, query: str) -> List[Dict]:
        """
        Simulate web search for external knowledge.
        """

        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": web_prompt,
                    "stream": False,
                    "options": {"temperature": 0.5}
                },
                timeout=60
            )
            if response.status_code == 200:
                result = response.json().get("response", "").strip()
                # Parse into chunks
                web_chunks = []
                for i, part in enumerate(result.split('[Result')):
                    if part.strip():
                        content = part.split(']')[-1].strip() if ']' in part else part.strip()
                        if content:
                            web_chunks.append({
                                "content": content,
                                "metadata": {"source": f"web_search_{i}", "type": "web"},
                                "score": 0.8  # Web results get moderate confidence
                            })
                return web_chunks[:3]  # Return at most 3 results
            return []
        except Exception as e:
            print(f"  Web search error: {e}")
            return []

    def filter_relevant_documents(self, chunks: List[Dict], scores: List[float],
                                   threshold: float = 0.4) -> List[Dict]:
        """
        Filter documents to keep only those above relevance threshold.
        """
        filtered = []
        for chunk, score in zip(chunks, scores):
            if score >= threshold:
                chunk_copy = chunk.copy()
                chunk_copy['relevance_score'] = score
                filtered.append(chunk_copy)
        return filtered

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 5: KNOWLEDGE REFINEMENT
    # ══════════════════════════════════════════════════════════════════════════

    def refine_knowledge(self, query: str, chunks: List[Dict]) -> str:
        """
        Refine and extract key knowledge from documents.
        """
        if not chunks:
            return ""

        # Combine document content
        combined = "\n\n---\n\n".join([
            f"[{c['metadata'].get('source', 'doc')}]\n{c['content']}"
            for c in chunks
        ])

        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": refine_prompt,
                    "stream": False,
                    "options": {"temperature": 0.2}
                },
                timeout=60
            )
            if response.status_code == 200:
                return response.json().get("response", "").strip()
            return combined[:1500]  # Fallback to truncated original
        except Exception:
            return combined[:1500]

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 6: ANSWER GENERATION
    # ══════════════════════════════════════════════════════════════════════════

    def generate_answer(self, question: str, refined_knowledge: str,
                        decision: RetrievalDecision) -> str:
        """
        Generate the final answer using refined knowledge.
        """
        # Adapt prompt based on decision
        if decision == RetrievalDecision.CORRECT:
            confidence_note = "The following information is from our knowledge base and is reliable."
        elif decision == RetrievalDecision.INCORRECT:
            confidence_note = "The information below is from external sources as our knowledge base didn't contain relevant information."
        else:
            confidence_note = "The information combines our knowledge base with supplementary sources."

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
            return f"Error generating answer"
        except Exception as e:
            return f"Error: {e}"

    # ══════════════════════════════════════════════════════════════════════════
    # MAIN CRAG PIPELINE
    # ══════════════════════════════════════════════════════════════════════════

    def query(self, question: str) -> CRAGResult:
        """
        Execute the full CRAG pipeline.
        """
        print(f"\n{'='*60}")
        print(f"CRAG Query: {question}")
        print('='*60)

        # Step 1: Retrieve
        print("\n[1/6] Retrieving documents...")
        chunks = self.retrieve(question, k=5)
        print(f"      Retrieved {len(chunks)} documents")

        # Step 2: Evaluate relevance
        print("\n[2/6] Evaluating document relevance...")
        relevance_scores, reasonings = self.evaluate_all_documents(question, chunks)
        for i, (chunk, score) in enumerate(zip(chunks, relevance_scores)):
            source = chunk['metadata'].get('source', 'unknown').split('/')[-1]
            print(f"      Doc {i+1}: {source} - Relevance: {score:.2f}")

        # Step 3: Make decision
        print("\n[3/6] Making retrieval decision...")
        decision = self.make_decision(relevance_scores)
        print(f"      Decision: {decision.value.upper()}")

        # Step 4: Take corrective action
        web_results = []
        web_search_used = False

        if decision == RetrievalDecision.CORRECT:
            print("\n[4/6] Using retrieved documents (high relevance)")
            # Filter to only relevant documents
            filtered_chunks = self.filter_relevant_documents(chunks, relevance_scores, 0.5)
            final_chunks = filtered_chunks

        elif decision == RetrievalDecision.INCORRECT:
            print("\n[4/6] Retrieval insufficient - performing web search...")
            web_results = self.simulate_web_search(question)
            web_search_used = True
            print(f"      Retrieved {len(web_results)} web results")
            final_chunks = web_results  # Use only web results

        else:  # AMBIGUOUS
            print("\n[4/6] Partial relevance - refining + supplementing...")
            # Keep somewhat relevant docs
            filtered_chunks = self.filter_relevant_documents(chunks, relevance_scores, 0.3)
            # Also do web search
            web_results = self.simulate_web_search(question)
            web_search_used = True
            print(f"      Kept {len(filtered_chunks)} docs, added {len(web_results)} web results")
            final_chunks = filtered_chunks + web_results

        # Step 5: Refine knowledge
        print("\n[5/6] Refining knowledge...")
        refined_knowledge = self.refine_knowledge(question, final_chunks)
        print(f"      Extracted {len(refined_knowledge)} chars of refined knowledge")

        # Step 6: Generate answer
        print("\n[6/6] Generating answer...")
        answer = self.generate_answer(question, refined_knowledge, decision)

        # Build result
        result = CRAGResult(
            question=question,
            answer=answer,
            initial_chunks=chunks,
            relevance_scores=relevance_scores,
            decision=decision,
            web_search_used=web_search_used,
            web_results=web_results if web_results else None,
            refined_knowledge=refined_knowledge,
            final_context=final_chunks
        )

        return result

    # ══════════════════════════════════════════════════════════════════════════
    # COMPARISON WITH STANDARD RAG
    # ══════════════════════════════════════════════════════════════════════════

    def standard_rag_query(self, question: str) -> str:
        """
        Standard RAG for comparison (no correction).
        """
        chunks = self.retrieve(question, k=3)

        if not chunks:
            return "I couldn't find relevant information."

        context = "\n\n".join([c['content'] for c in chunks])

        prompt = f"""Answer based on the context.

CONTEXT:
{context[:2000]}

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
            return "Error"
        except Exception as e:
            return f"Error: {e}"

    def compare_with_standard(self, question: str) -> Dict:
        """
        Compare CRAG vs standard RAG on the same question.
        """
        print(f"\n{'='*60}")
        print("Comparing CRAG vs Standard RAG")
        print('='*60)

        # Standard RAG
        print("\n[Standard RAG]")
        standard_answer = self.standard_rag_query(question)

        # CRAG
        print("\n[Corrective RAG]")
        crag_result = self.query(question)

        return {
            'question': question,
            'standard_answer': standard_answer,
            'crag_result': crag_result
        }


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

DECISION_COLORS = {
    RetrievalDecision.CORRECT: GREEN,
    RetrievalDecision.INCORRECT: RED,
    RetrievalDecision.AMBIGUOUS: YELLOW
}


# ══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE DEMO
# ══════════════════════════════════════════════════════════════════════════════

def run_demo():
    """
    Run interactive CRAG demo.
    """
    print(f"\n{BOLD}Corrective RAG (CRAG) - Self-Correcting Retrieval{RESET}")
    print("="*55)
    print("RAG that knows when it doesn't know - and fixes it")
    print("="*55)

    crag = CorrectiveRAG()

    print(f"\n{BOLD}CRAG Decisions:{RESET}")
    print(f"  {GREEN}CORRECT{RESET}   - High relevance, use retrieved docs")
    print(f"  {YELLOW}AMBIGUOUS{RESET} - Partial relevance, refine + supplement")
    print(f"  {RED}INCORRECT{RESET} - Low relevance, use web search")

    while True:
        print(f"\n{BOLD}Options:{RESET}")
        print("  1. Run CRAG query")
        print("  2. Compare CRAG vs Standard RAG")
        print("  3. Exit")

        choice = input(f"\n{BOLD}Choose (1-3):{RESET} ").strip()

        if choice == "1":
            question = input(f"\n{BOLD}Enter question:{RESET} ").strip()
            if not question:
                continue

            result = crag.query(question)

            # Display results
            print(f"\n{'─'*60}")
            print(f"{BOLD}CRAG RESULT{RESET}")
            print('─'*60)

            # Decision
            decision_color = DECISION_COLORS.get(result.decision, RESET)
            print(f"\n{BOLD}Decision:{RESET} {decision_color}{result.decision.value.upper()}{RESET}")

            # Relevance scores
            print(f"\n{BOLD}Document Relevance:{RESET}")
            for i, (chunk, score) in enumerate(zip(result.initial_chunks, result.relevance_scores)):
                source = chunk['metadata'].get('source', 'unknown').split('/')[-1]
                bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
                color = GREEN if score >= 0.7 else YELLOW if score >= 0.5 else RED
                print(f"  {i+1}. [{bar}] {color}{score:.2f}{RESET} - {source}")
                if result.relevance_reasonings:
                     print(f"     {result.relevance_reasonings[i]}")

            # Web search
            if result.web_search_used:
                print(f"\n{CYAN}🌐 Web search was used to supplement knowledge{RESET}")

            # Answer
            print(f"\n{BOLD}Answer:{RESET}")
            print(f"  {result.answer}")

        elif choice == "2":
            question = input(f"\n{BOLD}Enter question:{RESET} ").strip()
            if not question:
                continue

            comparison = crag.compare_with_standard(question)

            # Display comparison
            print(f"\n{'='*60}")
            print(f"{BOLD}COMPARISON RESULTS{RESET}")
            print('='*60)

            print(f"\n{RED}{BOLD}[STANDARD RAG]{RESET}")
            print(f"  {comparison['standard_answer']}")

            result = comparison['crag_result']
            decision_color = DECISION_COLORS.get(result.decision, RESET)

            print(f"\n{GREEN}{BOLD}[CORRECTIVE RAG]{RESET} (Decision: {decision_color}{result.decision.value}{RESET})")
            if result.web_search_used:
                print(f"  {CYAN}(Used web search for supplemental info){RESET}")
            print(f"  {result.answer}")

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
