# Advanced RAG
## Advanced Techniques for Leveraging your Data with GenAI
## Session labs 
## Revision 1.11 - 01/26/26

**Follow the startup instructions in the README.md file IF NOT ALREADY DONE!**

**NOTE: To copy and paste in the codespace, you may need to use keyboard commands - CTRL-C and CTRL-V. Chrome may work best for this.**

**Lab 1 - Working with Vector Databases**

**Purpose: In this lab, we’ll get a reminder on how to use vector databases for storing supporting data and doing similarity searches.**

1. For this lab and the following ones, we'll be using files in the *code* subdirectory. Change to that directory.

```
cd code
```

<br><br>

2. We have several data files that we'll be using that are for a ficticious company. The files are located in the *data/knowledge_base_pdfs* directory. [knowledge base pdfs](./data/knowledge_base_pdfs) You can browse them via the explorer view. Here's a [direct link](./data/knowledge_base_pdfs/OmniTech_Returns_Policy_2024.pdf) to an example one if you want to open it and take a look at it.

![PDF data file](./images/ragv2-15.png?raw=true "PDF data file") 

<br><br>

3. In our repository, we have some simple tools built around a popular vector database called Chroma. There are two files which will create a vector db (index) for the *.py files in our repo and another to do the same for the pdfs in our knowledge base. You can look at the files either via the usual "code <filename>" method or clicking on [**tools/index_code.py**](./tools/index_code.py) or [**tools/index_pdfs.py**](./tools/index_pdfs.py).

```
code ../tools/index_code.py
code ../tools/index_pdfs.py
```

<br><br>

4. Let's create a vector database of our local code files (python and bash). Run the program to index those. **This may run for a while before you see things happening.** You'll see the program loading the embedding model that will turn the code chunks into numeric represenations in the vector database and then it will read and index our *.py files. It will create a new local vector database in *./chroma_code_db*.

```
python ../tools/index_code.py
```

![Running code indexer](./images/arag3.png?raw=true "Running code indexer")

![Running code indexer](./images/arag4.png?raw=true "Running code indexer")

<br><br>

5. To help us do easy/simple searches against our vector databases, we have another tool at [**tools/search.py**](./tools/search.py). This tool connects to the ChromaDB vector database we create, and, using cosine similarity metrics, finds the top "hits" (matching chunks) and prints them out. You can open it and look at the code in the usual way if you want. No changes are needed to the code.

```
code ../tools/search.py
```

This tool takes a *--target* argument when you run it with a value of either "code" or "pdfs" to indicate which vector database to search.
You can also pass search queries directly on the command line with the *--query* argument. Or you can just start it and type in the queries, hit return, and get results. To exit in that mode, type "exit".

<br><br>

6. Now, let's run the search tool against the vector database we built in step 4. You can run it with phrases related to our coding like any of the ones shown below. You can run the commands with separate invocations of the tool as shown here, or just run it and enter them in interactive mode.  Notice the top hits and their respective cosine similarity values. Are they close? Farther apart?

```
  python ../tools/search.py --query "convert text to vectors" --target code
  python ../tools/search.py --query "tokenize sentences" --target code
  python ../tools/search.py --query "convert text to numbers" --target code
```

![Running search](./images/arag5.png?raw=true "Running search")

<br><br>

7.  Let's create a vector database based off of the PDF files. Just run the indexer for the pdf file.

```
python ../tools/index_pdfs.py --pdf-dir ../data/knowledge_base_pdfs
```

![Indexing PDFs](./images/arag6.png?raw=true "Indexing PDFs")

<br><br>

8. We can run the same search tool to find the top hits for information about the company policies. Below are some prompts you can try here. Notice the cosine similarity values on each - are they close? Farther apart?  When done, just type "exit".

```
  python ../tools/search.py --query "track my shipment" --target pdfs
  python ../tools/search.py --query "forgot my login credentials" --target pdfs
  python ../tools/search.py --query "exchange damaged item" --target pdfs
```

![PDF search](./images/arag7.png?raw=true "PDF search")

<br><br>

9. Keep in mind that this is not trying to intelligently answer your prompts at this point. This is a simple semantic search to find related chunks. In lab 2, we'll add in the LLM to give us better responses. 

<br>
<p align="center">
<b>[END OF LAB]</b>
</p>
</br></br>


**Lab 2: Building a Complete RAG System with Vector DBs**

**Purpose: In this lab, we'll create a complete RAG (Retrieval-Augmented Generation) system that uses content from our vector db and an LLM to generate intelligent, grounded answers.**

1. You should still be in the *code* subdirectory. We're going to build a TRUE RAG system that combines vector search with LLM generation. This is different from Lab 1 - instead of just finding similar chunks, we'll use those chunks as context for an LLM to generate complete answers. First, let's examine our complete RAG implementation. We have a completed version and a skeleton version. Use the diff command to see the differences:

```
code -d ../extra/rag_complete.txt rag_code.py
```

![Diff](./images/aia-1-42.png?raw=true "Diff")

<br><br>

2. Once you have the diff view open, take a moment to look at the structure in the complete version on the left. Notice the three main methods: `retrieve()` for finding chunks, `build_prompt()` for augmenting with context, and `generate()` for calling the LLM. These are the three steps of RAG.

- Lines 95-157: `retrieve()` - semantic search in ChromaDB
- Lines 159-209: `build_prompt()` - combining context with the question
- Lines 211-273: `generate()` - calling Ollama's Llama 3.2 model

<br><br>

3. Now, merge the code segments from the complete file (left side) into the skeleton file (right side) by hovering over the middle bar and clicking the arrow pointing right in the middle bar for each difference. Start with the comments section at the top, then work your way down through the class methods.

![Merge](./images/aia-1-43.png?raw=true "Merge")

<br><br>

5. After merging all the changes, double-check that there are no remaining diffs (red blocks on the side). Then close the diff view by clicking the "X" in the tab. 

![Completed](./images/aia-1-44.png?raw=true "Completed")

<br><br>

6. Now let's run our complete RAG system:

```
python rag_code.py
```

The system will connect to the vector database we created in Lab 2 and check if Ollama is running.

<br><br>

7. You should see knowledge base statistics showing how many chunks are indexed, and a check that Ollama is running with the llama3.2:1b model. If you see any errors about Ollama not running, check that with "ollama list".  If Ollama doesn't respond, try "ollama serve &".

![Running](./images/arag8.png?raw=true "Running")

<br><br>

8. Now you'll be at a prompt to ask questions. Try this first question:

```
How can I return a product?
```

Watch what happens - the system will show you the three RAG steps in the logs:
- **[RETRIEVE]** Finding relevant chunks in the vector database
- **[AUGMENT]** Building a prompt with context
- **[GENERATE]** Querying Llama 3.2 to generate an answer

<br><br>

9. After a few seconds, you'll see an ANSWER section with the LLM-generated response, followed by a SOURCES section showing which PDFs and pages were used. Notice how the answer is much more complete and natural than just showing search results.

![Answer](./images/arag9.png?raw=true "Answer")

<br><br>

10. Try a few more questions to see RAG in action:

```
What are the shipping costs?
How do I reset my password?
What should I do if my device won't turn on?
```

For each question, notice how the system retrieves relevant chunks and generates a complete answer based on that context.

![Answer](./images/ragv2-12.png?raw=true "Answer")

<br><br>

11. Now try asking a question that's NOT in the PDFs to see how RAG handles it:

```
What's the CEO's favorite color?
```

![Answer](./images/aia-1-48.png?raw=true "Answer")

Notice how the system should say it doesn't have that information (rather than making something up). This is the "grounding" benefit of RAG - answers are based on actual documents.

<br><br>

12. When you're done experimenting, type `quit` to exit the system.

<br><br>


**Key Takeaways:**
- You've built a TRUE RAG system that combines vector search with LLM generation
- RAG has three steps: Retrieve relevant chunks, Augment the prompt with context, Generate answers with an LLM
- The system uses ChromaDB for semantic search and Ollama/Llama 3.2 for generation
- RAG answers are grounded in your documents - reducing hallucination compared to pure LLM queries
- The system can cite sources, showing which documents and pages were used

<p align="center">
<b>[END OF LAB]</b>
</p>
</br></br>


**Lab 3 - Implementing Graph RAG with Frameworks**

**Purpose: In this lab, we'll see how to simply implement Graph RAG by leveraging frameworks and using LLMs to help generate queries.**


1. To do this lab, we need a graph database. We'll use a docker image for this that is already populated with data for us. Change to the neo4j directory and run the script command below. This will take a few minutes to build and start. Be sure to add the "&" to run this in the background.

(When it is ready, you will see messages like the ones shown below. Just hit *Enter* and you can change back to the *workspaces/adv-rag* subdirectory. 

```
cd /workspaces/adv-rag/neo4j
./neo4j-setup.sh 2 &
cd ..
```

![neo4j ready](./images/arag13.png?raw=true "neo4j ready")

<br><br>

2. This graph database is prepopulated with a large set of nodes and relationships related to movies. This includes actors and directors associated with movies, as well as the movie's genre, imdb rating, etc. You can take a look at the graph nodes by running the following commands in the terminal. **You should be in the "root" directory (/workspaces/adv-rag) when you run these commands.**

```
npm i -g http-server
http-server
```
<br><br>

3. After a moment, you should see a pop-up dialog. Click on the "Make public" button to make the port public. 

![running local web server](./images/rag24.png?raw=true "running local web server")

<br>

Then go to the *PORTS* tab and, in the second column under *Forwarded Address*, find the row for port 8080. Click on the *globe* icon.

<br>

![running local web server](./images/arag40.png?raw=true "running local web server")

<br>

You may be asked to click a button to continue to see the codespace. If so, click to accept. Then, it should take you to the graph. It will take a minute or two to load and then you can zoom in by using your mouse (roll wheel) to see more details.

<br>

![loading nodes](./images/rag25.png?raw=true "loading nodes")
![graph nodes](./images/rag26.png?raw=true "graph nodes")


<br><br>

4. When done, you can stop the *http-server* process with *Ctrl-C*. Now, let's go back and create a file to use the langchain pieces and the llm to query our graph database. Change back to the *genai* directory and create a new file named lab3.py.
   
```
cd code
code lab3.py
```

<br><br>

5. First, add the imports from *langchain* that we need. Put the following lines in the file you just created.
```
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_ollama import OllamaLLM
```

<br><br>

6. Next, let's add the connection to the graph database. Add the following to the file.
```
graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password="neo4jtest",
    enhanced_schema=False,
)
```

<br><br>


7. Now, let's create the chain instance that will allow us to leverage the LLM to help create the Cypher query and help frame the answer so it makes sense. We'll use Ollama and our llama3 model for both the LLM to create the Cypher queries and the LLM to help frame the answers.

```
chain = GraphCypherQAChain.from_llm(
    cypher_llm=OllamaLLM(model="llama3.2:3b", temperature=0),
    qa_llm=OllamaLLM(model="llama3.2:3b", temperature=0),
    graph=graph,
    verbose=True,
    allow_dangerous_requests=True,
)
```

<br><br>

8. Finally, let's add the code loop to take in a query and invoke the chain. After you've added this code, save the file.

```
while True:
    query = input("\nQuery: ")
    if query == "exit":
        break
    if query.strip() == "":
        continue
    response = chain.invoke({"query": query})
    print(response["result"])
```

<br><br>

9. Now, run the code.
```
python lab3.py
```

<br><br>

10. You can prompt it with queries related to the info in the graph database, like below. (Notes: Actual output length may be limited by the framework. Because of the small model we are using, other queries may or may n ot work well.)
```
Which movies are comedies?
```

![querying the graph](./images/arag14.png?raw=true "querying the graph")

<br><br>

<br>
<p align="center">
<b>[END OF LAB]</b>
</p>
</br></br>

**Lab 4 - Hybrid RAG: Semantic Search + Knowledge Graph**

**Purpose: In this lab, we'll build a hybrid RAG system that combines semantic search (ChromaDB) with knowledge graph traversal (Neo4j) to get both precision and context in our answers.**

1. RAG works because semantic search understands meaning. But for precise facts (timeframes, contacts, relationships), a knowledge graph provides structured answers. Combining both gives us the best of both worlds.

For this lab, a knowledge graph has been pre-built from the OmniTech documents. It contains:
- **Entities**: Products, Policies, TimeFrames, Contacts, Conditions, Fees, ShippingMethods, Documents
- **Relationships**: APPLIES_TO, HAS_TIMEFRAME, HANDLES, REQUIRES_CONDITION, HAS_FEE, USES_SHIPPING, CONTAINS

You can view it [here](./neo4j/data3/omnitech_policies.csv) if interested.

<br><br>
  
2. First, let's create the Neo4j graph database with the OmniTech knowledge graph. Run the commands (similar to lab 3) below.

```
cd /workspaces/adv-rag/neo4j
./neo4j-setup.sh 3 &
```

Wait for the message indicating Neo4j is ready (about 30-60 seconds). The script will:
- Build a Docker image with the OmniTech schema
- Start Neo4j container on ports 7474 (web) and 7687 (Bolt)
- Load the schema

When done, you will see a message ending with "Then run:    MATCH (n) RETURN count(n);". This is informational and you can just hit *Enter/Return* to get back to the prompt.

![building graph db](./images/arag15.png?raw=true "building graph db")

<br><br>

3. Change back to the code directory. Then we'll build out the hybrid RAG system as lab4.py with the diff and merge process that we've used before. The second command below will start up the editor session.
   
```
cd /workspaces/adv-rag/code
code -d ../extra/lab4-changes.txt lab4.py
```

<br><br>

4. What you'll see here is that most of the merges are comment sections explaining what the code does (plus some for the prompt, etc.). You can review and merge them as we've done before. After looking over the change, hover over the middle section and click the arrow to merge. Continue with this process until there are no more differences. Then click on the "X" in the tab at the top to close and save your changes.

![merge and save](./images/arag16.png?raw=true "merge and save")

<br><br>
 

5. Now let's run the hybrid RAG demo with the command below (in the *code* directory). This will then be waiting for you to type in a query.

```
python lab4.py
```

![running](./images/arag17.png?raw=true "running")

<br><br>

6. Let's try a basic query for the return policy. Type in the query below and hit *Enter/Return*.

```
What is the return window for Pro-Series equipment and who do I contact?
```

<br><br>

![running query](./images/arag18.png?raw=true "running query")

7. Watch the output - the demo asks the same question using three different methods:

You'll see:
- **METHOD 1: SEMANTIC** - Finds document chunks with similar meaning
- **METHOD 2: GRAPH** - Traverses Neo4j relationships via Cypher
- **METHOD 3: HYBRID** - Combines both for precision + context

Each method shows:
- What it retrieved (chunks vs graph nodes)
- The LLM-generated answer based on that context

<br><br>

You can compare the results:

| Method | What it found | Strength |
|--------|---------------|----------|
| SEMANTIC | Document chunks mentioning Pro-Series | Good context, handles vocabulary mismatch |
| GRAPH | Pro_Series → Pro_Series_Return → 14_Days | Precise facts via Cypher traversal |
| HYBRID | Graph facts + Document context | Combines both worlds |

![multiple answers](./images/arag19.png?raw=true "multiple answers")

<br><br>

8. Let's try another query that may benefit more from having the graph db involved. Enter the one below.
   
```
Who handles defective items?
```

<br><br>

9. Notice again the variations in the responses. Typically, because of the direct mapping, the *HYBRID* and *GRAPH* responses will have the best information.

![2nd query](./images/arag21.png?raw=true "2nd query")

<br><br>
    
10. Discussion Points:
- **Semantic search** (ChromaDB) understands MEANING - handles "money back" → "refund"
- **Graph search** (Neo4j) understands STRUCTURE - traverses entity relationships
- **Cypher queries** navigate: `Product → Policy → TimeFrame → Contact`
- **Hybrid** combines both: graph precision + semantic context
- This mirrors production RAG architectures used by enterprises

<br><br>


**Key Takeaway:**
> Semantic search understands MEANING. Graph search understands STRUCTURE. Together they provide comprehensive, accurate answers.

<br>
<p align="center">
<b>[END OF LAB]</b>
</p>
</br></br>

**Lab 5 - RAG Evaluation and Quality Metrics**

**Purpose: In this lab, we'll learn how to evaluate RAG system quality - a critical concern for enterprise deployments where accuracy, reliability, and answer traceability are paramount.**

1. You should still be in the *code* subdirectory. We're going to build a RAG evaluation system that measures retrieval quality, answer accuracy, and detects potential hallucinations. First, let's examine our evaluation implementation. We have a completed version and a skeleton version. Use the diff command to see the differences:

```
code -d ../extra/lab5_eval_complete.txt lab5.py
```

![diff](./images/arag22.png?raw=true "diff")

<br><br>

2. Once you have the diff view open, take a moment to look at the structure in the complete version on the left. Notice the key evaluation metrics:
   - **Context Relevance**: How relevant are retrieved chunks to the question?
   - **Answer Groundedness**: Is the answer supported by the context?
   - **Answer Completeness**: Does the answer address all parts of the question?
   - **Hallucination Detection**: Is the LLM making unsupported claims?

<br><br>

3. Now, merge the code segments from the complete file (left side) into the skeleton file (right side) by clicking the arrow pointing right in the middle bar for each difference. Start with the docstrings and comments at the top, then work your way down through the evaluation methods.

<br><br>

4. After merging all the changes, double-check that there are no remaining diffs (red blocks on the side). Then close the diff view by clicking the "X" in the tab.

<br><br>

5. Now let's run our RAG evaluation system:

```
python lab5.py
```

The system will connect to the vector database we created earlier and present you with options.

<br><br>

6. You should see a menu with options to evaluate a single question or run a full test suite. Select option **1** to evaluate a single question. Enter a question like:

```
How do I reset my password?
```

![question](./images/arag25.png?raw=true "question")

<br><br>

7. Watch the evaluation process - the system will:
   - **[1/5]** Retrieve relevant context chunks
   - **[2/5]** Generate an answer using the LLM
   - **[3/5]** Evaluate context relevance (LLM-as-judge)
   - **[4/5]** Check answer groundedness (is it supported by context?)
   - **[5/5]** Assess answer completeness

<br><br>

8. After evaluation, you'll see color-coded scores:
   - **GREEN (0.8+)**: Excellent quality
   - **YELLOW (0.6-0.8)**: Acceptable, room for improvement
   - **RED (below 0.6)**: Needs attention

Notice the **OVERALL SCORE** which weights the metrics based on enterprise priorities (groundedness is most important at 40%).

![run](./images/arag24.png?raw=true "run")

<br><br>

9. Try a few more questions to see how scores vary:

```
What is the return policy for products?
What are the shipping costs?
Who is the CEO of OmniTech?
```

Notice how the last question (about the CEO) should show lower groundedness if that information isn't in the documents.

![run](./images/arag26.png?raw=true "run")

<br><br>

**Steps 10-12 are optional and may take longer than lab time allows.**

<br>

10. Now select option **2** to run the full test suite. This runs evaluation on a predefined set of questions with expected keywords - simulating automated regression testing.

<br><br>

11. After the test suite completes, you'll see aggregate metrics for your entire RAG system. This is how enterprises monitor RAG quality:
   - Track metrics over time
   - Set quality thresholds for production readiness
   - Compare different RAG configurations


![system test](./images/arag27.png?raw=true "system test")

<br><br>

12. Discussion Points:
   - **Why is groundedness critical?** Hallucinated answers can cause real business damage
   - **LLM-as-judge approach**: Using one LLM to evaluate another LLM's output
   - **Automated testing**: Test suites enable continuous quality monitoring
   - **Enterprise compliance**: Evaluation metrics provide audit trails for regulated industries

<br><br>

<br>
<p align="center">
<b>[END OF LAB]</b>
</p>
</br></br>

**Lab 6 - Query Transformation and Re-ranking**

**Purpose: In this lab, we'll implement advanced retrieval techniques that dramatically improve RAG quality - query transformation (expansion, multi-query, HyDE) and two-stage retrieval with re-ranking.**

1. You should still be in the *code* subdirectory. We're going to build an advanced RAG system that transforms user queries for better retrieval and re-ranks results for higher precision. Use the diff command to examine the implementation:

```
code -d ../extra/lab6_rerank_complete.txt lab6.py
```

![diff and merge](./images/arag28.png?raw=true "diff and merge")

<br><br>

2. Once you have the diff view open, look at the key techniques in the complete version:
   - **Query Expansion**: Add synonyms and related terms
   - **Multi-Query**: Generate multiple query variations
   - **HyDE**: Generate hypothetical answers to search for
   - **Re-ranking**: Score and reorder retrieved chunks

<br><br>

3. Merge all the code segments from the complete file into the skeleton file, starting from the top. Pay attention to the prompt templates used for each transformation technique.

<br><br>

4. After merging, close the diff view. Now let's run the advanced RAG demo:

```
python lab6.py
```

![run](./images/arag29.png?raw=true "run")

<br><br>

5. You'll see a menu explaining the different retrieval methods. Each has a color code:
   - **RED (BASIC)**: Standard vector search (baseline)
   - **YELLOW (EXPANSION)**: Query expanded with synonyms
   - **GREEN (MULTI-Q)**: Multiple query variations
   - **CYAN (HYDE)**: Hypothetical document embedding
   - **MAGENTA (RERANK)**: Two-stage with re-ranking

<br><br>

6. Select option **1** to compare all methods on a query. Enter a short, ambiguous query:

```
money back
```

This is intentionally vague - notice how the different methods handle vocabulary mismatch (documents might say "refund" or "return" instead of "money back").

<br><br>

7. Watch the output as each method processes the query:
   - **Query Expansion** adds synonyms like "refund", "reimbursement"
   - **Multi-Query** generates variations like "refund process", "return policy"
   - **HyDE** generates an ideal answer and searches for similar content
   - **Re-ranking** retrieves more candidates then scores them precisely

<br><br>

8. Compare the answers from each method. Notice how:
   - Basic search might miss relevant documents
   - Expanded queries find more related content
   - HyDE often finds the most relevant passages
   - Re-ranking improves precision (relevant docs ranked higher)

![response](./images/arag30.png?raw=true "response")

<br><br>

9. Now select option **2** to try individual techniques. Choose **3 (HyDE)** and enter:

```
how long to return
```

![response](./images/arag32.png?raw=true "response")

Notice how HyDE generates a hypothetical answer like "Customers may return products within 30 days..." and uses THAT to search - bridging the gap between question-style and answer-style text.

<br><br>

10. Try the **Re-ranking** technique (option 4) with:

```
shipping options and costs
```

![response](./images/arag31.png?raw=true "response")

Notice how re-ranking retrieves 6 candidates (2x the final count) and then scores each one's relevance to return only the top 3 most relevant.

<br><br>

11. Discussion Points:
   - **Query-document mismatch**: Users ask questions, but documents contain answers
   - **HyDE insight**: Searching with answer-like text finds answer-containing documents
   - **Re-ranking trade-off**: More compute for higher precision
   - **Combining techniques**: Production systems often use multiple approaches
   - **Enterprise value**: Better retrieval = better answers from same knowledge base

<p align="center">
**[END OF LAB]**
</p>
</br></br>

**Lab 7 - Corrective RAG (CRAG)**

**Purpose: In this lab, we'll implement Corrective RAG (CRAG), an advanced technique where the system evaluates its own retrieval quality and takes corrective action when results are insufficient - including falling back to web search.**

1. You should still be in the *code* subdirectory. We're going to build a self-correcting RAG system that "knows when it doesn't know" and takes corrective action. Use the diff command to examine the implementation:

```
code -d ../extra/lab7_crag_complete.txt lab7.py
```

<br><br>

2. Once you have the diff view open, look at the CRAG workflow in the complete version:
   - **Retrieval Grader**: Evaluates relevance of each retrieved document
   - **Decision Logic**: CORRECT / AMBIGUOUS / INCORRECT based on scores
   - **Corrective Actions**: Web search fallback, document filtering
   - **Knowledge Refinement**: Extract only relevant information
   - **Answer Generation**: Generate with confidence-appropriate prompts

![response](./images/arag33.png?raw=true "response")

<br><br>

3. Merge all the code segments from the complete file into the skeleton file. Pay special attention to the evaluation prompts and decision thresholds.

<br><br>

4. After merging, close the diff view. Now let's run the CRAG demo:

```
python lab7.py
```

<br><br>

5. You'll see a menu with options and the CRAG decision legend:
   - **GREEN (CORRECT)**: High relevance - use retrieved documents
   - **YELLOW (AMBIGUOUS)**: Partial relevance - refine + supplement with web
   - **RED (INCORRECT)**: Low relevance - fall back to web search

![run](./images/arag34.png?raw=true "run")

<br><br>

6. Select option **1** to run a CRAG query. First, try a question that SHOULD be in the knowledge base:

```
What is the return policy for products?
```

Watch the 6-step CRAG pipeline execute:
- Retrieves 5 documents (more than typical RAG)
- Evaluates each document's relevance (0.0-1.0)
- Makes a decision (likely CORRECT for this query)
- Uses filtered documents without web search
- Refines knowledge and generates answer

<br><br>

7. Notice the visual relevance bars showing each document's score. Documents above 0.7 are considered highly relevant (green), 0.4-0.7 are ambiguous (yellow), and below 0.4 are irrelevant (red).

![known](./images/arag35.png?raw=true "known")

<br><br>

8. Now try a question that's likely NOT in the knowledge base:

```
What is the current stock price of OmniTech?
```

Watch the system detect low relevance and trigger web search (simulated). This is CRAG in action - it "knows when it doesn't know."

![known](./images/arag36.png?raw=true "known")

<br><br>

9. Now select option **2** to compare CRAG vs Standard RAG on a question. Enter:

```
How do I contact support for warranty issues?
```

![known](./images/arag37.png?raw=true "known")

Compare the answers - CRAG should provide a more complete response by intelligently filtering or supplementing the context.

![known](./images/arag38.png?raw=true "known")

<br><br>

10. Try the comparison with a question outside the knowledge base:

```
What are the latest AI developments in 2026?
```

![unknown](./images/arag39.png?raw=true "unknown")

Notice how Standard RAG might hallucinate or give a vague answer, while CRAG recognizes the retrieval failure and seeks external information.

<br><br>

12. Discussion Points:
   - **Self-awareness**: CRAG "knows when it doesn't know" - critical for enterprise trust
   - **Graceful degradation**: Falls back to external search rather than hallucinating
   - **Relevance thresholds**: Tunable parameters (0.7/0.4) for your quality requirements
   - **Audit trail**: CRAGResult tracks every decision for compliance/debugging
   - **Production patterns**: Real systems use actual web search APIs (Google, Bing, Tavily)
   - **Cost trade-off**: More LLM calls for evaluation, but better answer quality

<br>
<p align="center">
<b>[END OF LAB]</b>
</p>
</br></br>


<p align="center">
<b>For educational use only by the attendees of our workshops.</b>
</p>

<p align="center">
<b>(c) 2026 Tech Skills Transformations and Brent C. Laster. All rights reserved.</b>
</p>

