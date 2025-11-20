#!/usr/bin/env python3
"""
Enhanced RAG Implementation with performance monitoring
"""

import os
import re
import logging
import time
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
import pypdf

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("rag-system")

class KnowledgeBase:
    """Enhanced RAG system with performance monitoring"""

    def __init__(self, pdf_directory: str = "./knowledge_base_pdfs"):
        """Initialize the knowledge base with ChromaDB"""
        self.pdf_directory = pdf_directory
        self.chroma_client = None
        self.collection = None
        self.documents = []
        self.performance_metrics = {
            "load_time": 0,
            "search_times": [],
            "total_searches": 0
        }
        self.initialize_database()
        self.load_documents()

    def initialize_database(self):
        """Initialize ChromaDB vector database with timing"""
        start_time = time.time()
        logger.info("Initializing ChromaDB...")

        self.chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))

        # Delete existing collection if it exists
        try:
            self.chroma_client.delete_collection("knowledge_docs")
        except:
            pass

        # Create new collection
        self.collection = self.chroma_client.create_collection("knowledge_docs")

        init_time = time.time() - start_time
        logger.info(f"ChromaDB initialized successfully in {init_time:.3f} seconds")

    def load_pdf_document(self, file_path: str, file_id: str) -> Optional[Dict]:
        """Load a single PDF document with timing"""
        try:
            start_time = time.time()
            logger.info(f"Loading PDF: {file_path}")

            # Extract text from PDF
            with open(file_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                text = ""
                page_count = len(pdf_reader.pages)

                for page_num, page in enumerate(pdf_reader.pages):
                    text += page.extract_text() + " "
                    if (page_num + 1) % 10 == 0:
                        logger.debug(f"  Processed {page_num + 1}/{page_count} pages")

            # Clean up extracted text
            text = re.sub(r'\s+', ' ', text.strip())

            # Determine category from filename with enhanced logic
            category = "general"
            file_lower = file_id.lower()

            category_mappings = {
                "returns": ["return", "refund", "exchange"],
                "shipping": ["shipping", "delivery", "ship"],
                "account": ["password", "account", "login", "auth"],
                "troubleshooting": ["troubleshoot", "problem", "issue", "fix"],
                "payment": ["payment", "pay", "card", "billing"],
                "support": ["support", "help", "contact"]
            }

            for cat, keywords in category_mappings.items():
                if any(keyword in file_lower for keyword in keywords):
                    category = cat
                    break

            document = {
                "id": file_id,
                "text": text,
                "category": category,
                "source": os.path.basename(file_path),
                "page_count": page_count,
                "char_count": len(text)
            }

            load_time = time.time() - start_time
            logger.info(f"Successfully loaded {file_id}: {len(text)} chars, {page_count} pages in {load_time:.2f}s")

            return document

        except Exception as e:
            logger.error(f"Failed to load PDF {file_path}: {e}")
            return None

    def load_documents(self):
        """Load all PDF documents from the directory with timing"""
        start_time = time.time()

        # Check if PDF directory exists
        if not os.path.exists(self.pdf_directory):
            logger.warning(f"PDF directory not found: {self.pdf_directory}")
            logger.info("Using sample documents instead")
            self.load_sample_documents()
            return

        # Load all PDF files
        pdf_files = [f for f in os.listdir(self.pdf_directory) if f.endswith('.pdf')]
        logger.info(f"Found {len(pdf_files)} PDF files to load")

        for i, filename in enumerate(pdf_files, 1):
            file_path = os.path.join(self.pdf_directory, filename)
            file_id = filename.replace('.pdf', '')

            logger.info(f"Loading document {i}/{len(pdf_files)}: {filename}")

            document = self.load_pdf_document(file_path, file_id)
            if document:
                self.documents.append(document)

                # Add to ChromaDB with metadata
                self.collection.add(
                    documents=[document["text"]],
                    metadatas=[{
                        "category": document["category"],
                        "source": document["source"],
                        "page_count": str(document["page_count"]),
                        "char_count": str(document["char_count"])
                    }],
                    ids=[document["id"]]
                )

        self.performance_metrics["load_time"] = time.time() - start_time
        logger.info(f"Loaded {len(self.documents)} documents in {self.performance_metrics['load_time']:.2f} seconds")

    def load_sample_documents(self):
        """Load sample documents if PDFs are not available"""
        sample_docs = [
            {
                "id": "policy_returns",
                "text": "Return Policy: Items can be returned within 30 days of purchase with original receipt. Products must be in original condition and packaging. Refunds are processed within 5-7 business days. Exchanges are available for different sizes or colors of the same product. Return shipping is free for defective items.",
                "category": "returns",
                "source": "sample_data"
            },
            {
                "id": "policy_shipping",
                "text": "Shipping Information: Standard shipping takes 3-5 business days within the US ($5.99). Express shipping available in 1-2 business days ($15.99). Free shipping on orders over $50. International shipping takes 7-14 business days ($19.99). Tracking information is provided for all orders.",
                "category": "shipping",
                "source": "sample_data"
            },
            {
                "id": "account_password",
                "text": "Account & Password Help: To reset your password, click 'Forgot Password' on the login page. Enter your email address and check your inbox for reset instructions. Passwords must be at least 8 characters with one uppercase letter and one number. For account issues, contact support@example.com",
                "category": "account",
                "source": "sample_data"
            },
            {
                "id": "troubleshoot_power",
                "text": "Device Power Troubleshooting: If your device won't turn on, first check the power cable connection. Try a different outlet. Hold the power button for 10 seconds to force reset. Check if the battery needs charging - plug in for at least 30 minutes. If issues persist, the device may need service.",
                "category": "troubleshooting",
                "source": "sample_data"
            }
        ]

        for doc in sample_docs:
            doc["page_count"] = 1
            doc["char_count"] = len(doc["text"])
            self.documents.append(doc)

            self.collection.add(
                documents=[doc["text"]],
                metadatas=[{
                    "category": doc["category"],
                    "source": doc["source"],
                    "page_count": "1",
                    "char_count": str(doc["char_count"])
                }],
                ids=[doc["id"]]
            )

        logger.info(f"Loaded {len(sample_docs)} sample documents")

    def search(self, query: str, max_results: int = 3) -> List[Dict]:
        """Search the knowledge base for relevant documents with timing"""
        start_time = time.time()
        self.performance_metrics["total_searches"] += 1

        try:
            logger.info(f"Searching for: '{query}'")

            # Query ChromaDB
            results = self.collection.query(
                query_texts=[query],
                n_results=max_results
            )

            # Format results
            formatted_results = []
            if results['documents'] and len(results['documents'][0]) > 0:
                for i in range(len(results['documents'][0])):
                    document = results['documents'][0][i]
                    metadata = results['metadatas'][0][i]
                    distance = results['distances'][0][i] if 'distances' in results else 0

                    # Calculate relevance score (inverse of distance)
                    score = 1.0 / (1.0 + distance)

                    formatted_results.append({
                        "id": results['ids'][0][i],
                        "content": document,
                        "category": metadata.get('category', 'unknown'),
                        "source": metadata.get('source', 'unknown'),
                        "score": score,
                        "metadata": metadata
                    })

            search_time = time.time() - start_time
            self.performance_metrics["search_times"].append(search_time)

            logger.info(f"Search completed in {search_time:.3f} seconds, found {len(formatted_results)} results")

            return formatted_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def get_statistics(self) -> Dict:
        """Get enhanced statistics about the knowledge base"""
        stats = {
            "total_documents": len(self.documents),
            "categories": {},
            "total_characters": 0,
            "total_pages": 0,
            "performance": {
                "load_time": self.performance_metrics["load_time"],
                "total_searches": self.performance_metrics["total_searches"],
                "avg_search_time": 0
            }
        }

        for doc in self.documents:
            category = doc["category"]
            if category not in stats["categories"]:
                stats["categories"][category] = 0
            stats["categories"][category] += 1
            stats["total_characters"] += doc.get("char_count", len(doc["text"]))
            stats["total_pages"] += doc.get("page_count", 1)

        # Calculate average search time
        if self.performance_metrics["search_times"]:
            stats["performance"]["avg_search_time"] = sum(self.performance_metrics["search_times"]) / len(self.performance_metrics["search_times"])

        return stats

# Main execution for testing
if __name__ == "__main__":
    # Initialize knowledge base
    kb = KnowledgeBase("../knowledge_base_pdfs")

    # Show enhanced statistics
    stats = kb.get_statistics()
    print("\n" + "="*60)
    print("Knowledge Base Statistics")
    print("="*60)
    print(f"Total Documents: {stats['total_documents']}")
    print(f"Total Characters: {stats['total_characters']:,}")
    print(f"Total Pages: {stats['total_pages']}")
    print(f"\nLoad Time: {stats['performance']['load_time']:.2f} seconds")
    print("\nDocuments by Category:")
    for category, count in stats['categories'].items():
        print(f"  {category}: {count}")

    # Test search with timing
    test_queries = [
        "How can I return a product?",
        "What are the shipping costs?",
        "Reset my password"
    ]

    print("\n" + "="*60)
    print("Testing Search Performance")
    print("="*60)

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = kb.search(query, max_results=2)

        for i, result in enumerate(results, 1):
            print(f"  Result {i}: {result['category']} (Score: {result['score']:.3f})")

    # Show performance summary
    final_stats = kb.get_statistics()
    print("\n" + "="*60)
    print("Performance Summary")
    print("="*60)
    print(f"Total Searches: {final_stats['performance']['total_searches']}")
    print(f"Average Search Time: {final_stats['performance']['avg_search_time']:.3f} seconds")