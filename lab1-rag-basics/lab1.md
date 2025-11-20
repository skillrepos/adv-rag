# Lab 1: Building a Basic RAG (Retrieval-Augmented Generation) System
## Understanding the foundation of AI-powered knowledge retrieval
## Session lab - Lab 1 of 7
## Revision 1.0 - 2025-11-19

**Notes:**
1. We will be running in a GitHub Codespace using Ollama to serve llama3.2:3b as the model.
2. Substitute the appropriate key combinations for your operating system where needed.
3. To copy and paste in the codespace, you may need to use keyboard commands - CTRL-C and CTRL-V.

</br></br></br>

**Lab 1: Building a Basic RAG System**

**Purpose: In this lab, we'll create a basic RAG (Retrieval-Augmented Generation) system that can read PDF documents, store them in a vector database, and retrieve relevant information based on queries.**

1. First, let's navigate to our lab directory and start setting up the basic RAG implementation.

```
cd lab1-rag-basics
```
<br><br>

2. Let's verify that Ollama is running and that we have the llama3.2:3b model available:

```
ollama list
```

If Ollama is not running, start it with:
```
ollama serve &
```
<br><br>

3. Now, let's examine our basic RAG implementation. We have a completed version and a skeleton version. Use the diff command to see the differences:

```
code -d rag_complete.py rag_skeleton.py
```
<br><br>

4. Once you have the diff view open, merge the code segments from the complete file (left side) into the skeleton file (right side) by clicking the arrow pointing right in the middle bar for each difference. Start with the imports section, then the document loading function, and finally the search functionality.

![Side-by-side merge](../images/merge-example.png)
<br><br>

5. After merging all the changes, close the diff view by clicking the "X" in the tab. Now let's test our basic RAG system:

```
python rag_skeleton.py
```
<br><br>

6. The system should load the PDF documents and create the vector database. Let's now create a simple test script to query the knowledge base. Create a new file:

```
code test_rag.py
```
<br><br>

7. Paste the following code into the test_rag.py file:

```python
from rag_skeleton import KnowledgeBase

# Initialize the knowledge base
kb = KnowledgeBase()

# Test queries
queries = [
    "How do I return a product?",
    "What are the shipping options?",
    "How can I reset my password?"
]

print("Testing RAG System\n" + "="*50)
for query in queries:
    print(f"\nQuery: {query}")
    results = kb.search(query, max_results=2)

    for i, result in enumerate(results, 1):
        print(f"Result {i}: {result['content'][:200]}...")
        print(f"Category: {result['category']}, Score: {result['score']:.2f}")
```
<br><br>

8. Save the file (CTRL/CMD + S) and run the test:

```
python test_rag.py
```
<br><br>

9. You should see search results for each query. Notice how the system finds relevant documents based on the query content. Let's add some performance monitoring to our RAG system. Use the diff tool again:

```
code -d rag_enhanced.py rag_skeleton.py
```
<br><br>

10. Merge in the enhancements that add timing information and better logging. After merging, close the diff view.
<br><br>

11. Now let's test the enhanced version with a simple benchmark:

```
python benchmark_rag.py
```
<br><br>

12. You should see timing information for document loading and search operations. This gives us a baseline for our RAG system performance.

![RAG Performance](../images/rag-performance.png)
<br><br>

13. Finally, let's verify that our knowledge base persists correctly. Run the persistence test:

```
python test_persistence.py
```
<br><br>

14. The test should show that documents are correctly stored and can be retrieved even after restarting the system.

<p align="center">
**[END OF LAB]**
</p>
</br></br>

**Key Takeaways:**
- You've built a basic RAG system that can load PDF documents
- The system uses ChromaDB as a vector database for similarity search
- Documents are chunked and embedded for efficient retrieval
- The system can find relevant information based on semantic similarity

**Next Lab Preview:**
In Lab 2, we'll build a standalone agent that can use this RAG system to answer questions using the Ollama LLM.