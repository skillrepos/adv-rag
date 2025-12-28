# Gen AI: Understanding and Using RAG
## Making LLMs smarter by pairing your data with Gen AI
## Session labs 
## Revision 4.0 - 12/28/25

**Follow the startup instructions in the README.md file IF NOT ALREADY DONE!**

**NOTE: To copy and paste in the codespace, you may need to use keyboard commands - CTRL-C and CTRL-V. Chrome may work best for this.**

**Lab 1 - Grounding data by augmenting prompts**

**Purpose: In this lab, we’ll see a basic example of augmenting a prompt by retrieving context from a data file.**

1. In our repository, we have a set of Python programs to help us illustrate and work with concepts in the labs. The first set are in the *code* subdirectory. Go to the *TERMINAL* tab in the bottom part of your codespace and change into that directory.

```
cd code
```
<br><br>

2. For this lab, we will simulate "retrieving" data from the Troubleshooting and Security manuals for a fictitious company called "Omnitech". Run the command below to create a text-based context file using information from the provided documents.

```
cat <<EOF > ../data/omnitech_context.txt
OmniTech Force Restart: Press and hold the Power button for exactly 10 seconds.
OmniTech Password Policy (v5.2): Accounts created after Jan 1, 2024, must be 8+ chars and cannot contain dictionary words like 'omnitech'.
OmniTech Holiday Returns: Items bought Nov 1 - Dec 25 can be returned until Jan 31.
EOF
```
<br><br>


3. Open the starter script [lab1.py](./genai/lab1.py) in the editor.

```
code lab1.py
```

<br><br>


4. Modify the prompt on line 17 to ask a highly specific question about OmniTech's internal procedures. This tests the LLM's "base" knowledge:

```python
"prompt": "How long do I need to hold the power button to force restart an OmniTech device, and what is the return deadline for a gift bought on December 10th?:\n\n",
```

![prompt 1](./images/ragv2-1.png?raw=true "prompt 1") 

<br><br>

5. Save the file (CTRL+S or CMD+S) and run it. Observe the result: The AI will likely give a generic answer (like "usually 5-10 seconds") or admit it doesn't have enough information or details.

```bash
python lab1.py
```


![first run](./images/ragv2-3.png?raw=true "first run") 

<br><br>

6. Now, let’s add the "Retrieval" step. At the top of your script (after the imports), add the code to read the OmniTech context you created:

```python
# Read the proprietary OmniTech documentation snippet
with open("../data/omnitech_context.txt", "r") as file:
    omnitech_info = file.read()
```

![read context](./images/ragv2-7.png?raw=true "read context") 

<br><br>

7. Update the prompt to include this context. Change the "prompt" line to use an f-string that injects the documentation:

```python
"prompt": f"Using the OmniTech Manuals below, answer the user question.\n\nManuals: {omnitech_info}\n\nQuestion: How long do I need to hold the power button to force restart an OmniTech device, and what is the return deadline for a gift bought on December 10th?:\n\n",
```

![prompt 2](./images/ragv2-4.png?raw=true "prompt 2") 


<br><br>

8. Save your changes (CTRL+S or CMD+S).

<br><br>

9. Run the script again:

```bash
python lab1.py
```

<br><br>

10. Verify the success: Notice how the AI now provides the exact "10 seconds" requirement and the "January 31" holiday deadline found in the context.

![run 2](./images/ragv2-6.png?raw=true "run 2") 

<br><br>

11. Discussion Point: Why didn't we just "train" the model on this data? Training is expensive and slow. By simply "attaching" the relevant page of the manual to the prompt, we updated the model's knowledge in milliseconds.

<br><br>
    
<p align="center">
**[END OF LAB]**
</p>
</br></br>


**Lab 2 - Working with Vector Databases**

**Purpose: In this lab, we’ll learn about how to use vector databases for storing supporting data and doing similarity searches.**

1. For this lab and the following ones, we'll be using files in the *rag* subdirectory. Change to that directory.

```
cd data
```

<br><br>

2. We have several data files that we'll be using that are for a ficticious company. The files are located in the *data/knowledge_base_pdfs* directory. [knowledge base pdfs](./data/knowledge_base_pdfs) You can browse them via the explorer view. Here's a [direct link](./data/knowledge_base_pdfs/OmniTech_Returns_Policy_2024.pdf) to an example one if you want to open it and take a look at it.

![PDF data file](./images/rag2v-15.png?raw=true "PDF data file") 

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

![Running code indexer](./images/aia-1-35.png?raw=true "Running code indexer")

![Running code indexer](./images/aia-1-36.png?raw=true "Running code indexer")

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

![Running search](./images/ragv2-8.png?raw=true "Running search")

<br><br>

7.  Let's create a vector database based off of the PDF files. Just run the indexer for the pdf file.

```
python ../tools/index_pdfs.py --pdf-dir ../data/knowledge_base_pdfs
```

![Indexing PDFs](./images/ragv2-9.png?raw=true "Indexing PDFs")

<br><br>

8. We can run the same search tool to find the top hits for information about the company policies. Below are some prompts you can try here. Notice the cosine similarity values on each - are they close? Farther apart?  When done, just type "exit".

```
  python ../tools/search.py --query "track my shipment" --target pdfs
  python ../tools/search.py --query "forgot my login credentials" --target pdfs
  python ../tools/search.py --query "exchange damaged item" --target pdfs
```

![PDF search](./images/ragv2-10.png?raw=true "PDF search")

<br><br>

8. Keep in mind that this is not trying to intelligently answer your prompts at this point. This is a simple semantic search to find related chunks. In lab 3, we'll add in the LLM to give us better responses. 

<br>
<p align="center">
<b>[END OF LAB]</b>
</p>
</br></br>


**Lab 3: Building a Complete RAG System**

**Purpose: In this lab, we'll create a complete RAG (Retrieval-Augmented Generation) system that retrieves relevant context from our vector database and uses an LLM to generate intelligent, grounded answers.**

1. You should still be in the *rag* subdirectory. We're going to build a TRUE RAG system that combines vector search with LLM generation. This is different from Lab 2 - instead of just finding similar chunks, we'll use those chunks as context for an LLM to generate complete answers. First, we need to bring down a smaller model to use with these labs. Use the Ollama command below:

```
ollama pull llama3.2:1b
```

<br><br>

2. Now let's examine our complete RAG implementation. We have a completed version and a skeleton version. Use the diff command to see the differences:

```
code -d ../extra/rag_complete.txt rag_code.py
```

![Diff](./images/aia-1-42.png?raw=true "Diff")

<br><br>

3. Once you have the diff view open, take a moment to look at the structure in the complete version on the left. Notice the three main methods: `retrieve()` for finding chunks, `build_prompt()` for augmenting with context, and `generate()` for calling the LLM. These are the three steps of RAG.

- Lines 95-157: `retrieve()` - semantic search in ChromaDB
- Lines 159-209: `build_prompt()` - combining context with the question
- Lines 211-273: `generate()` - calling Ollama's Llama 3.2 model

<br><br>

4. Now, as you've done before, merge the code segments from the complete file (left side) into the skeleton file (right side) by clicking the arrow pointing right in the middle bar for each difference. Start with the comments section at the top, then work your way down through the class methods.

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

![Running](./images/ragv2-13.png?raw=true "Running")

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

![Answer](./images/ragv2-11.png?raw=true "Answer")

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

**Lab 4 - Implementing Graph RAG with Neo4j**

**Purpose: In this lab, we'll see how to implement Graph RAG by querying a Neo4j database and using Ollama to generate responses.**

1. For this lab, we'll need a neo4j instance running. We'll use a docker image for this that is already populated with data for us. There is a shell script named [**neo4j/neo4j-setup.sh**](./neo4j/neo4j-setup.sh) that you can run to start the neo4j container running. Change to the neo4j directory, set an environment variable for the DOCKER version and run the script. This will take a few minutes to build and start. Afterwards you can change back to the *code* subdirectory. Be sure to include the "&" to run this in the background.

```
cd /workspaces/rag/neo4j

export DOCKER_API_VERSION=1.43

./neo4j-setup.sh 1 &

```

<br><br>

2. When done, you may see an "INFO Started" or just a "naming to docker.io/library/neo4j:custom" message. The container should then be running. You can just hit *Enter* and do a *docker ps* command to verify you see a "neo4j:custom" container with "Up # seconds" in the STATUS column.

```
docker ps
```
![container check](./images/ragv2-14.png?raw=true "container check")

<br><br>

3. For the next steps, make sure you're back in the *code* directory. In here, we have a simple Python program to interact with the graph database and query it. The file name is lab4.py. Open the file either by clicking on [**code/lab4.py**](./code/lab4.py) or by entering the *code* command below in the codespace's terminal.

```
cd ../code
code lab4.py
```

<br><br>

4. You can look around this file to see how it works. It simply connects to the graph database, does a Cypher query (see the function *query_graph* on line 6), and returns the results. For this one, the graph db was initialized with information that *Ada Lovelace, a Mathematician, worked with Alan Turing, a Computer Scientist*.

<br><br>

5. When done looking at the code, go ahead and execute the program using the command below. When it's done, you'll be able to see the closest match from the knowledge base data file to the query.
```
python lab4.py
```
![running lab4 file](./images/rag21.png?raw=true "running lab4 file")

<br><br>

5. Now, let's update the code to pass the retrieved answer to an LLM to expand on. We'll be using the llama3 model that we setup with Ollama previously. For simplicity, the changes are already in a file in [**extra/lab4-changes.txt**](./extra/lab4-changes.txt) To see and merge the differences, we'll use the codespace's built-in diff/merge functionality. Run the command below.

```
code -d /workspaces/rag/extra/lab4-changes.txt /workspaces/rag/code/lab4.py
```

<br><br>

6. Once you have this screen up, take a look at the added functionality in the *lab4-changes.txt* file. Here we are passing the answer collected from the knowledge base onto the LLM and asking it to expand on it. To merge the changes, you can click on the arrows between the two files (#1 and #2 in the screenshot) and then close the diff window via the X in the upper corner (#3 in the screenshot).

![lab 4 diff](./images/rag22.png?raw=true "lab 4 diff")

<br><br>

7. Now, you can go ahead and run the updated file again to see what the LLM generates using the added context. Note: This will take several minutes to run.  (If you happen to get an error about not being able to establish a connection, your ollama server may not be running any longer.  If that's the case, you can restart it via the command *"ollama serve &"* and then rerun the python command again.)

```
python lab4.py
```

<br><br>

8. After the run is complete, you should see additional data from the LLM related to the additional context with an interesting result!

![lab output 4](./images/rag23.png?raw=true "lab output 4")

<p align="center">
**[END OF LAB]**
</p>
</br></br>

**Lab 5 - Simplifying RAG with Frameworks and LLMs**

**Purpose: In this lab, we'll see how to simplify Graph RAG by leveraging frameworks and using LLMs to help generate queries.**

1. In our last lab, we hardcoded Cypher queries and worked more directly with the Graph database. Let's see how we can simplify this.

<br><br>

2. First, we need a different graph database. Again, we'll use a docker image for this that is already populated with data for us. Change to the neo4j directory and run the script, but note the different parameter ("2" instead of "1"). This will take a few minutes to build and start. Be sure to add the "&" to run this in the background.

(When it is ready, you may see a "*INFO  [neo4j/########] successfully initialized:*" message or one that says "naming to docker.io/library/neo4j:custom".) Just hit *Enter* and you can change back to the *workspaces/rag* subdirectory. 

```
cd /workspaces/rag/neo4j
./neo4j-setup.sh 2 &
cd ..
``` 

<br><br>

3. This graph database is prepopulated with a large set of nodes and relationships related to movies. This includes actors and directors associated with movies, as well as the movie's genre, imdb rating, etc. You can take a look at the graph nodes by running the following commands in the terminal. **You should be in the "root" directory (/workspaces/rag) when you run these commands.**

```
npm i -g http-server
http-server
```

<br><br>

3. After a moment, you should see a pop-up dialog that you can click on to open a browser to see some of the nodes in the graph. It will take a minute or two to load and then you can zoom in by using your mouse (roll wheel) to see more details.

![running local web server](./images/rag24.png?raw=true "running local web server")
![loading nodes](./images/rag25.png?raw=true "loading nodes")
![graph nodes](./images/rag26.png?raw=true "graph nodes")


<br><br>

4. When done, you can stop the *http-server* process with *Ctrl-C*. Now, let's go back and create a file to use the langchain pieces and the llm to query our graph database. Change back to the *genai* directory and create a new file named lab5.py.
```
cd code
code lab5.py
```

<br><br>

5. First, add the imports from *langchain* that we need. Put the following lines in the file you just created.
```
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_ollama import OllamaLLM
```

<br><br>

6. Now, let's add the connection to the graph database. Add the following to the file.
```
graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password="neo4jtest",
    enhanced_schema=False,
)
```

<br><br>


7. Next, let's create the chain instance that will allow us to leverage the LLM to help create the Cypher query and help frame the answer so it makes sense. We'll use Ollama and our llama3 model for both the LLM to create the Cypher queries and the LLM to help frame the answers.
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
python lab5.py
```

<br><br>

10. You can prompt it with queries related to the info in the graph database, like:
```
Who starred in Star Trek : Generations?
Which movies are comedies?
```


<p align="center">
**[END OF LAB]**
</p>
</br></br>

**Lab 6 - Hybrid RAG: Semantic Search + Knowledge Graph**

**Purpose: In this lab, we'll build a hybrid RAG system that combines semantic search (ChromaDB) with knowledge graph traversal (Neo4j) to get both precision and context in our answers.**

1. RAG works because semantic search understands meaning. But for precise facts (timeframes, contacts, relationships), a knowledge graph provides structured answers. Combining both gives us the best of both worlds.

For this lab, a knowledge graph has been pre-built from the OmniTech documents. It contains:
- **Entities**: Products, Policies, TimeFrames, Contacts, Conditions, Fees, ShippingMethods, Documents
- **Relationships**: APPLIES_TO, HAS_TIMEFRAME, HANDLES, REQUIRES_CONDITION, HAS_FEE, USES_SHIPPING, CONTAINS

You can view it [here](./neo4j/data3/omnitech_policies.csv) if interested.

<br><br>
  
2. First, let's create the Neo4j graph database with the OmniTech knowledge graph. Run the commands (similar to lab 5) below.

```
cd /workspaces/rag/neo4j
./neo4j-setup.sh 3 &
```

Wait for the message indicating Neo4j is ready (about 30-60 seconds). The script will:
- Build a Docker image with the OmniTech schema
- Start Neo4j container on ports 7474 (web) and 7687 (Bolt)
- Auto-initialize the knowledge graph via APOC

When done, you will see a message ending with "Then run:    MATCH (n) RETURN count(n);". This is informational and you can just hit *Enter/Return* to get back to the prompt.

![building graph db](./images/ragv2-17.png?raw=true "building graph db")

<br><br>

3. Change back to the code directory. Then we'll build out the hybrid RAG system as lab6.py with the diff and merge process that we've used before. The second command below will start up the editor session.
   
```
cd /workspaces/rag/code
code -d ../extra/lab6-changes.txt lab6.py
```

![building hybrid code](./images/ragv2-18.png?raw=true "building hybrid code")

<br><br>

4. What you'll see here is that most of the merges are comment sections explaining what the code does (plus some for the prompt, etc.). You can review and merge them as we've done before. After looking over the change, hover over the middle section and click the arrow to merge. Continue with this process until there are no more differences. Then click on the "X" in the tab at the top to close and save your changes.

![merge and save](./images/ragv2-19.png?raw=true "merge and save")

<br><br>
 

5. Now let's run the hybrid RAG demo with the command below (in the *code* directory). This will then be waiting for you to type in a query.

```
python lab6.py
```

![running](./images/ragv2-20.png?raw=true "running")

<br><br>

6. Let's try a basic query for the return policy. Type in the query below and hit *Enter/Return*.

```
What is the return window for Pro-Series equipment and who do I contact?
```

![running query](./images/ragv2-22.png?raw=true "running query")

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

![multiple answers](./images/ragv2-21.png?raw=true "multiple answers")

<br><br>

8. Let's try another query that may benefit more from having the graph db involved. Enter the one below.
   
```
Who handles defective items?
```

![2nd query](./images/ragv2-24.png?raw=true "2nd query")

<br><br>


9. Notice again the variations in the responses. Typically, because of the direct mapping, the *HYBRID* and *GRAPH* responses will have the best information.

![2nd query](./images/ragv2-23.png?raw=true "2nd query")

<br><br>
    
Notice the `HybridRAG` class connects to both databases:
- **ChromaDB** for semantic search (lines 46-53)
- **Neo4j** for graph search (lines 55-67)

<br><br>

10. Discussion Points:
- **Semantic search** (ChromaDB) understands MEANING - handles "money back" → "refund"
- **Graph search** (Neo4j) understands STRUCTURE - traverses entity relationships
- **Cypher queries** navigate: `Product → Policy → TimeFrame → Contact`
- **Hybrid** combines both: graph precision + semantic context
- This mirrors production RAG architectures used by enterprises

<br><br>

11. [OPTIONAL] You can also visualize the knowledge graph. Start a local web server:

```
cd /workspaces/rag/neo4j/data3/public
npx http-server
```

Click the pop-up to open the browser and see the graph visualization. You can move around using the mouse and also zoom in and out. 

**NOTE**: If you can't open the page, you may need to go back to the codespace, go to the *PORTS* tab (next to *TERMINAL*), right-click, and set the *Visibility* field to *Public*. See screenshot below.

![make port public](./images/ragv2-26.png?raw=true "make port public")

After that, refresh the page and try again. You may still have to click through another page to allow access.

<br><br>

**Key Takeaway:**
> Semantic search understands MEANING. Graph search understands STRUCTURE. Together they provide comprehensive, accurate answers.

<p align="center">
**[END OF LAB]**
</p>
</br></br>

<p align="center">
**[END OF LAB]**
</p>
</br></br>


<p align="center">
<b>For educational use only by the attendees of our workshops.</b>
</p>

<p align="center">
<b>(c) 2026 Tech Skills Transformations and Brent C. Laster. All rights reserved.</b>
</p>

