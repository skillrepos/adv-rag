#!/usr/bin/env python3
"""
RAG Evaluation and Quality Metrics
================================================================================


1. RETRIEVAL QUALITY - Are we finding the right documents?

2. ANSWER QUALITY - Is the generated answer accurate?

3. HALLUCINATION DETECTION - Is the LLM making things up?

WHY EVALUATION MATTERS FOR ENTERPRISE:

PREREQUISITES:
   - ChromaDB populated with OmniTech documents (run index_pdfs.py first)
   - Ollama running with llama3.2:3b model (ollama pull llama3.2:3b)

USAGE:
   python lab7.py
"""

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════════════════════

import json
import requests
from typing import List, Dict, Tuple
from dataclasses import dataclass
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
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EvaluationResult:
    """
    Container for evaluation metrics on a single RAG query.
    """
    question: str
    answer: str
    retrieved_chunks: List[Dict]

    # Retrieval Metrics (0.0 - 1.0)
    context_relevance: float = 0.0      # How relevant are retrieved chunks?
    retrieval_precision: float = 0.0     # What % of chunks are useful?

    # Answer Metrics (0.0 - 1.0)
    answer_groundedness: float = 0.0     # Is answer supported by context?
    answer_completeness: float = 0.0     # Does answer address the question?

    # Hallucination Detection
    hallucination_score: float = 0.0     # 0.0 = no hallucination, 1.0 = severe
    unsupported_claims: List[str] = None  # Claims not found in context

    def overall_score(self) -> float:
        """
        Calculate weighted overall quality score.
        """
        return (
            self.answer_groundedness * 0.40 +
            self.context_relevance * 0.30 +
            self.answer_completeness * 0.20 +
            self.retrieval_precision * 0.10
        )


# ══════════════════════════════════════════════════════════════════════════════
# TEST SUITE - Ground Truth Questions and Expected Answers
# ══════════════════════════════════════════════════════════════════════════════

TEST_CASES = [
    {
        "question": "What is the return window for standard products?",
        "expected_keywords": ["30 days", "thirty days", "30-day"],
        "expected_source": "Returns_Policy",
        "category": "policy"
    },
    {
        "question": "How do I reset my password?",
        "expected_keywords": ["forgot password", "reset", "email", "link"],
        "expected_source": "Account",
        "category": "account"
    },
    {
        "question": "What are the shipping costs for express delivery?",
        "expected_keywords": ["express", "shipping", "cost", "fee", "$"],
        "expected_source": "Shipping",
        "category": "shipping"
    },
    {
        "question": "Who do I contact for defective items?",
        "expected_keywords": ["support", "contact", "email", "phone", "defective"],
        "expected_source": "Returns",
        "category": "support"
    }
]


# ══════════════════════════════════════════════════════════════════════════════
# RAG EVALUATOR CLASS
# ══════════════════════════════════════════════════════════════════════════════

class RAGEvaluator:
    """
    Comprehensive RAG system evaluator.
    """

    def __init__(self):
        """Initialize the evaluator with database connection."""
        print("Initializing RAG Evaluator...")

        # Connect to ChromaDB
        client = PersistentClient(
            path=CHROMA_PATH,
            settings=Settings(),
            tenant=DEFAULT_TENANT,
            database=DEFAULT_DATABASE
        )
        self.collection = client.get_collection(name=COLLECTION_NAME)
        print(f"  Connected to ChromaDB with {self.collection.count()} chunks")

    # ──────────────────────────────────────────────────────────────────────────
    # CORE RAG FUNCTIONS
    # ──────────────────────────────────────────────────────────────────────────

    def retrieve(self, query: str, k: int = 3) -> List[Dict]:
        """
        Retrieve relevant document chunks for a query.
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

    def generate(self, question: str, context_chunks: List[Dict]) -> str:
        """
        Generate an answer using the LLM with retrieved context.
        """
        # Build context string from chunks
        context = "\n\n".join([
            f"[Source: {c['metadata'].get('source', 'unknown')}]\n{c['content']}"
            for c in context_chunks
        ])

        prompt = f"""Based on the following context, answer the question.
If the answer is not in the context, say "I don't have enough information."

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
    # METRIC 1: CONTEXT RELEVANCE
    # ══════════════════════════════════════════════════════════════════════════

    def evaluate_context_relevance(self, question: str, chunks: List[Dict]) -> float:
        """
        Evaluate how relevant retrieved chunks are to the question.
        """
        if not chunks:
            return 0.0

        # Prepare context summary for evaluation
        context_summary = "\n".join([
            f"Chunk {i+1}: {c['content'][:200]}..."
            for i, c in enumerate(chunks)
        ])

        # TODO: Add the evaluation prompt here
        eval_prompt = f"""Rate how relevant these retrieved document chunks are to answering the question.

QUESTION: {question}

RETRIEVED CHUNKS:
{context_summary}

Rate the overall relevance on a scale of 1-5:
1 = Completely irrelevant
2 = Slightly relevant
3 = Moderately relevant
4 = Highly relevant
5 = Perfectly relevant

Respond with ONLY a single number (1-5):"""

        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": eval_prompt,
                    "stream": False,
                    "options": {"temperature": 0.1}
                },
                timeout=60
            )
            if response.status_code == 200:
                result = response.json().get("response", "").strip()
                # Extract number from response
                for char in result:
                    if char.isdigit():
                        score = int(char)
                        return min(max(score, 1), 5) / 5.0  # Normalize to 0-1
                return 0.5  # Default if no number found
            return 0.0
        except Exception as e:
            print(f"  Relevance evaluation error: {e}")
            return 0.0

    # ══════════════════════════════════════════════════════════════════════════
    # METRIC 2: ANSWER GROUNDEDNESS
    # ══════════════════════════════════════════════════════════════════════════

    def evaluate_groundedness(self, answer: str, chunks: List[Dict]) -> Tuple[float, List[str]]:
        """
        Evaluate if the answer is grounded in (supported by) the context.
        """
        if not chunks or not answer:
            return 0.0, ["No context or answer provided"]

        # Combine all context
        full_context = "\n".join([c['content'] for c in chunks])

        # TODO: Add the groundedness evaluation prompt here
        eval_prompt = f"""Analyze if the ANSWER is fully supported by the CONTEXT.

CONTEXT:
{full_context[:2000]}

ANSWER:
{answer}

Respond in this exact format:
GROUNDEDNESS_SCORE: [1-5]
UNSUPPORTED_CLAIMS: [List any claims not supported, or "None"]

Where score means:
1 = Answer completely unsupported
3 = About half supported
5 = Fully grounded in context"""

        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": eval_prompt,
                    "stream": False,
                    "options": {"temperature": 0.1}
                },
                timeout=60
            )

            if response.status_code == 200:
                result = response.json().get("response", "")

                # Parse score
                score = 3  # Default
                for line in result.split('\n'):
                    if 'GROUNDEDNESS_SCORE' in line.upper():
                        for char in line:
                            if char.isdigit():
                                score = int(char)
                                break

                # Parse unsupported claims
                unsupported = []
                in_claims_section = False
                for line in result.split('\n'):
                    if 'UNSUPPORTED' in line.upper():
                        in_claims_section = True
                        claim_part = line.split(':')[-1].strip()
                        if claim_part and claim_part.lower() != 'none':
                            unsupported.append(claim_part)
                    elif in_claims_section and line.strip().startswith('-'):
                        unsupported.append(line.strip()[1:].strip())

                return min(max(score, 1), 5) / 5.0, unsupported
            return 0.0, ["Evaluation failed"]
        except Exception as e:
            return 0.0, [str(e)]

    # ══════════════════════════════════════════════════════════════════════════
    # METRIC 3: ANSWER COMPLETENESS
    # ══════════════════════════════════════════════════════════════════════════

    def evaluate_completeness(self, question: str, answer: str) -> float:
        """
        Evaluate if the answer completely addresses the question.
        """
        # TODO: Add the completeness evaluation prompt here
        eval_prompt = f"""Evaluate how completely this answer addresses the question.

QUESTION: {question}

ANSWER: {answer}

Rate completeness from 1-5:
1 = Does not address the question
3 = Moderately complete
5 = Fully complete

Respond with ONLY a single number (1-5):"""

        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": eval_prompt,
                    "stream": False,
                    "options": {"temperature": 0.1}
                },
                timeout=60
            )
            if response.status_code == 200:
                result = response.json().get("response", "").strip()
                for char in result:
                    if char.isdigit():
                        score = int(char)
                        return min(max(score, 1), 5) / 5.0
                return 0.5
            return 0.0
        except Exception as e:
            print(f"  Completeness evaluation error: {e}")
            return 0.0

    # ══════════════════════════════════════════════════════════════════════════
    # METRIC 4: RETRIEVAL PRECISION (Keyword-based)
    # ══════════════════════════════════════════════════════════════════════════

    def evaluate_retrieval_precision(self, chunks: List[Dict], expected_keywords: List[str]) -> float:
        """
        Evaluate retrieval precision using keyword matching.
        """
        if not chunks or not expected_keywords:
            return 0.0

        relevant_count = 0
        for chunk in chunks:
            content_lower = chunk['content'].lower()
            # Check if any expected keyword is in this chunk
            if any(kw.lower() in content_lower for kw in expected_keywords):
                relevant_count += 1

        return relevant_count / len(chunks)

    # ══════════════════════════════════════════════════════════════════════════
    # FULL EVALUATION PIPELINE
    # ══════════════════════════════════════════════════════════════════════════

    def evaluate_query(self, question: str, expected_keywords: List[str] = None) -> EvaluationResult:
        """
        Run full evaluation pipeline on a single query.
        """
        print(f"\n{'='*60}")
        print(f"Evaluating: {question}")
        print('='*60)

        # Step 1: Retrieve
        print("  [1/5] Retrieving context...")
        chunks = self.retrieve(question)
        print(f"       Retrieved {len(chunks)} chunks")

        # Step 2: Generate
        print("  [2/5] Generating answer...")
        answer = self.generate(question, chunks)
        print(f"       Answer: {answer[:100]}...")

        # Step 3: Evaluate context relevance
        print("  [3/5] Evaluating context relevance...")
        context_relevance = self.evaluate_context_relevance(question, chunks)
        print(f"       Context Relevance: {context_relevance:.2f}")

        # Step 4: Evaluate groundedness
        print("  [4/5] Evaluating groundedness...")
        groundedness, unsupported = self.evaluate_groundedness(answer, chunks)
        print(f"       Groundedness: {groundedness:.2f}")
        if unsupported and unsupported[0] != "None":
            print(f"       Unsupported claims: {unsupported}")

        # Step 5: Evaluate completeness
        print("  [5/5] Evaluating completeness...")
        completeness = self.evaluate_completeness(question, answer)
        print(f"       Completeness: {completeness:.2f}")

        # Step 6: Precision (if keywords provided)
        precision = 0.0
        if expected_keywords:
            precision = self.evaluate_retrieval_precision(chunks, expected_keywords)
            print(f"       Retrieval Precision: {precision:.2f}")

        # Build result
        result = EvaluationResult(
            question=question,
            answer=answer,
            retrieved_chunks=chunks,
            context_relevance=context_relevance,
            retrieval_precision=precision,
            answer_groundedness=groundedness,
            answer_completeness=completeness,
            hallucination_score=1.0 - groundedness,  # Inverse of groundedness
            unsupported_claims=unsupported if unsupported else []
        )

        print(f"\n  OVERALL SCORE: {result.overall_score():.2f}")
        return result

    # ══════════════════════════════════════════════════════════════════════════
    # TEST SUITE RUNNER
    # ══════════════════════════════════════════════════════════════════════════

    def run_test_suite(self, test_cases: List[Dict] = None) -> Dict:
        """
        Run evaluation on a suite of test cases.
        """
        if test_cases is None:
            test_cases = TEST_CASES

        print("\n" + "="*60)
        print("RUNNING RAG EVALUATION TEST SUITE")
        print("="*60)
        print(f"Test cases: {len(test_cases)}")

        results = []
        for i, test in enumerate(test_cases):
            print(f"\n[Test {i+1}/{len(test_cases)}]")
            result = self.evaluate_query(
                test['question'],
                test.get('expected_keywords', [])
            )
            results.append({
                'test_case': test,
                'evaluation': result
            })

        # Aggregate metrics
        avg_relevance = sum(r['evaluation'].context_relevance for r in results) / len(results)
        avg_groundedness = sum(r['evaluation'].answer_groundedness for r in results) / len(results)
        avg_completeness = sum(r['evaluation'].answer_completeness for r in results) / len(results)
        avg_precision = sum(r['evaluation'].retrieval_precision for r in results) / len(results)
        avg_overall = sum(r['evaluation'].overall_score() for r in results) / len(results)

        summary = {
            'total_tests': len(test_cases),
            'avg_context_relevance': avg_relevance,
            'avg_groundedness': avg_groundedness,
            'avg_completeness': avg_completeness,
            'avg_precision': avg_precision,
            'avg_overall_score': avg_overall,
            'results': results
        }

        return summary


# ══════════════════════════════════════════════════════════════════════════════
# TERMINAL COLORS
# ══════════════════════════════════════════════════════════════════════════════

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"


def color_score(score: float) -> str:
    """Return colored score string based on value."""
    if score >= 0.8:
        return f"{GREEN}{score:.2f}{RESET}"
    elif score >= 0.6:
        return f"{YELLOW}{score:.2f}{RESET}"
    else:
        return f"{RED}{score:.2f}{RESET}"


# ══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE DEMO
# ══════════════════════════════════════════════════════════════════════════════

def run_demo():
    """
    Run interactive RAG evaluation demo.
    """
    print(f"\n{BOLD}RAG Evaluation System{RESET}")
    print("="*50)
    print("Measure retrieval quality, answer accuracy, and hallucination")
    print("="*50)

    evaluator = RAGEvaluator()

    while True:
        print(f"\n{BOLD}Options:{RESET}")
        print("  1. Evaluate a single question")
        print("  2. Run full test suite")
        print("  3. Exit")

        choice = input(f"\n{BOLD}Choose (1-3):{RESET} ").strip()

        if choice == "1":
            question = input(f"\n{BOLD}Enter question:{RESET} ").strip()
            if question:
                result = evaluator.evaluate_query(question)

                # Display formatted results
                print(f"\n{BOLD}{'─'*50}{RESET}")
                print(f"{BOLD}EVALUATION RESULTS{RESET}")
                print(f"{'─'*50}")
                print(f"Question: {result.question}")
                print(f"\nAnswer: {result.answer}")
                print(f"\n{BOLD}Metrics:{RESET}")
                print(f"  Context Relevance:  {color_score(result.context_relevance)}")
                print(f"  Answer Groundedness: {color_score(result.answer_groundedness)}")
                print(f"  Answer Completeness: {color_score(result.answer_completeness)}")
                print(f"  Hallucination Risk:  {color_score(1.0 - result.hallucination_score)}")
                print(f"\n  {BOLD}OVERALL SCORE: {color_score(result.overall_score())}{RESET}")

                if result.unsupported_claims and result.unsupported_claims[0] not in ["None", "Evaluation failed"]:
                    print(f"\n{YELLOW}⚠ Potential unsupported claims:{RESET}")
                    for claim in result.unsupported_claims:
                        print(f"  • {claim}")

        elif choice == "2":
            summary = evaluator.run_test_suite()

            # Display summary
            print(f"\n{BOLD}{'='*60}{RESET}")
            print(f"{BOLD}TEST SUITE SUMMARY{RESET}")
            print(f"{'='*60}")
            print(f"Total Tests: {summary['total_tests']}")
            print(f"\n{BOLD}Average Scores:{RESET}")
            print(f"  Context Relevance:  {color_score(summary['avg_context_relevance'])}")
            print(f"  Groundedness:       {color_score(summary['avg_groundedness'])}")
            print(f"  Completeness:       {color_score(summary['avg_completeness'])}")
            print(f"  Retrieval Precision: {color_score(summary['avg_precision'])}")
            print(f"\n  {BOLD}OVERALL SYSTEM SCORE: {color_score(summary['avg_overall_score'])}{RESET}")

            # Quality assessment
            overall = summary['avg_overall_score']
            if overall >= 0.8:
                print(f"\n{GREEN}✓ System performing well - ready for production{RESET}")
            elif overall >= 0.6:
                print(f"\n{YELLOW}⚠ System needs improvement - review low-scoring areas{RESET}")
            else:
                print(f"\n{RED}✗ System needs significant work - not production ready{RESET}")

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
