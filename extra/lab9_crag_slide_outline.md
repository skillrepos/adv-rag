# Slide Outline: Lab 9 - Corrective RAG (CRAG)
## Self-Correcting Retrieval-Augmented Generation

---

### Slide 1: The Problem with Standard RAG
**Title:** When RAG Doesn't Know It Doesn't Know

**Bullet Points:**
- Standard RAG always uses retrieved documents - even if irrelevant
- No mechanism to detect retrieval failures
- Results in:
  - Hallucinated answers based on unrelated context
  - Confident-sounding but wrong responses
  - User trust erosion
- Enterprise impact: Wrong answers can cause real damage

**Suggested Visual:** Side-by-side comparison showing "What user asked" vs "Irrelevant docs retrieved" vs "Hallucinated answer"

---

### Slide 2: Introducing CRAG
**Title:** Corrective RAG - Self-Healing Retrieval

**Bullet Points:**
- CRAG adds self-correction loop to standard RAG
- Key insight: Evaluate retrieval BEFORE generation
- If retrieval quality is poor → take corrective action
- "Know when you don't know" - then fix it
- Published research: "Corrective Retrieval Augmented Generation" (2024)

**Suggested Visual:** CRAG logo/diagram with feedback loop arrow

---

### Slide 3: The CRAG Workflow
**Title:** Six-Step Self-Correcting Pipeline

**Bullet Points:**
1. **Retrieve** - Get candidate documents (more than usual)
2. **Evaluate** - Grade each document's relevance
3. **Decide** - CORRECT / AMBIGUOUS / INCORRECT
4. **Correct** - Take action based on decision
5. **Refine** - Extract relevant knowledge
6. **Generate** - Produce grounded answer

**Suggested Visual:** Flowchart showing the 6 steps with decision diamond at step 3 branching into three paths

---

### Slide 4: The Retrieval Grader
**Title:** Evaluating Document Relevance

**Bullet Points:**
- Core innovation: LLM evaluates each retrieved document
- Question: "Does this document help answer the query?"
- Scoring: 1-5 scale normalized to 0.0-1.0
- Evaluates:
  - Topic alignment
  - Information usefulness
  - Answer potential
- Runs on EACH retrieved document

**Suggested Visual:** Document with overlay showing relevance score and evaluation criteria checkboxes

**Code Pattern:**
```python
def evaluate_relevance(query, document):
    # LLM rates: "Does this help answer the query?"
    # Returns 0.0 (irrelevant) to 1.0 (highly relevant)
```

---

### Slide 5: Decision Categories
**Title:** Three Paths Based on Relevance

**Bullet Points:**

| Decision | Condition | Action |
|----------|-----------|--------|
| **CORRECT** | max_score ≥ 0.7 | Use retrieved docs (filtered) |
| **AMBIGUOUS** | 0.4 ≤ max_score < 0.7 | Refine + supplement with web |
| **INCORRECT** | max_score < 0.4 | Fall back to web search |

- Thresholds are tunable per use case
- Conservative = higher thresholds (more web fallback)
- Aggressive = lower thresholds (trust retrieval more)

**Suggested Visual:** Traffic light diagram (green/yellow/red) with threshold values

---

### Slide 6: CORRECT Path
**Title:** High Confidence - Use Retrieved Documents

**Bullet Points:**
- Triggered when: At least one doc scores ≥ 0.7
- Action: Filter to keep only relevant documents
- Skip web search - knowledge base has the answer
- Fastest path through CRAG
- Most common for well-indexed knowledge bases

**Suggested Visual:** Green arrow flowing straight through pipeline, bypassing web search

---

### Slide 7: INCORRECT Path
**Title:** Low Confidence - External Search Fallback

**Bullet Points:**
- Triggered when: All docs score < 0.4
- Action: Discard retrieved docs, use web search
- Acknowledges: "Our knowledge base doesn't have this"
- Critical for:
  - Out-of-domain questions
  - Incomplete knowledge bases
  - Current events / recent information
- Prevents hallucination from irrelevant context

**Suggested Visual:** Red arrow bypassing document store, going directly to web search icon

---

### Slide 8: AMBIGUOUS Path
**Title:** Partial Confidence - Combine Sources

**Bullet Points:**
- Triggered when: 0.4 ≤ max_score < 0.7
- Action: Keep somewhat-relevant docs + supplement with web
- Best of both worlds:
  - Preserve partial knowledge base info
  - Fill gaps with external search
- Most sophisticated path
- Requires knowledge fusion

**Suggested Visual:** Yellow arrow splitting into two paths that merge back together

---

### Slide 9: Knowledge Refinement
**Title:** Extract Only What Matters

**Bullet Points:**
- After corrective action, refine the context
- Purpose:
  - Remove irrelevant portions of documents
  - Eliminate redundancy across sources
  - Focus on query-relevant information
- Creates concise, targeted context for generation
- Reduces noise → better answers

**Suggested Visual:** Funnel diagram showing raw docs → refinement → concentrated knowledge

**Code Pattern:**
```python
refined = refine_knowledge(query, chunks)
# Extracts only query-relevant facts
```

---

### Slide 10: Web Search Integration
**Title:** Fallback to External Knowledge

**Bullet Points:**
- When internal retrieval fails, search the web
- Production implementations use:
  - Google Search API
  - Bing Search API
  - Tavily (AI-optimized search)
  - DuckDuckGo
- Scrape and process top results
- Apply same knowledge refinement

**Suggested Visual:** Icons of search engines with arrows pointing to document processing

---

### Slide 11: Confidence-Aware Generation
**Title:** Adapting the Prompt to the Source

**Bullet Points:**
- Generation prompt adapts based on decision:
  - **CORRECT**: "This is from our reliable knowledge base"
  - **INCORRECT**: "This is from external sources"
  - **AMBIGUOUS**: "This combines internal and external sources"
- Helps LLM calibrate confidence
- Enables appropriate hedging in answers
- Transparency for the user

**Suggested Visual:** Three prompt templates color-coded by decision type

---

### Slide 12: The Audit Trail
**Title:** Full Transparency for Enterprise

**Bullet Points:**
- CRAGResult captures everything:
  - Original question and final answer
  - All retrieved documents with scores
  - Decision made and why
  - Whether web search was used
  - Refined knowledge used for generation
- Essential for:
  - Debugging failures
  - Compliance audits
  - Quality monitoring

**Suggested Visual:** CRAGResult object diagram showing all captured fields

---

### Slide 13: CRAG vs Standard RAG
**Title:** Side-by-Side Comparison

**Bullet Points:**

| Aspect | Standard RAG | CRAG |
|--------|--------------|------|
| Retrieval check | None | Per-document evaluation |
| Bad retrieval | Uses anyway | Detects and corrects |
| Web fallback | No | Yes, when needed |
| Hallucination risk | Higher | Lower |
| LLM calls | 1 | 2+ (eval + generate) |
| Latency | Lower | Higher |

**Suggested Visual:** Two pipeline diagrams side by side showing the difference

---

### Slide 14: When CRAG Shines
**Title:** Ideal Use Cases

**Bullet Points:**
- **Incomplete knowledge bases**: Gracefully handles gaps
- **Evolving information**: Falls back for current events
- **High-stakes applications**: Healthcare, legal, finance
- **Customer-facing systems**: Can't afford wrong answers
- **Hybrid knowledge needs**: Internal + public information
- **Compliance requirements**: Full audit trail

**Suggested Visual:** Icons representing each use case with checkmarks

---

### Slide 15: Tuning CRAG Thresholds
**Title:** Configuring for Your Use Case

**Bullet Points:**
- Two key thresholds:
  - `RELEVANCE_THRESHOLD_HIGH` (default 0.7) → CORRECT
  - `RELEVANCE_THRESHOLD_LOW` (default 0.4) → INCORRECT
- Higher thresholds = More conservative, more web fallback
- Lower thresholds = Trust retrieval more
- Tune based on:
  - Knowledge base completeness
  - Cost of wrong answers
  - Latency budget

**Suggested Visual:** Slider diagram showing threshold adjustment effects

---

### Slide 16: Performance Considerations
**Title:** The Cost of Self-Correction

**Bullet Points:**
- Trade-offs:
  - More LLM calls (N+1 for N documents)
  - Higher latency (evaluation step)
  - Potentially more tokens (web search results)
- Mitigations:
  - Batch evaluation calls
  - Use smaller/faster model for grading
  - Cache common query evaluations
  - Async evaluation pipeline

**Suggested Visual:** Cost/benefit balance scale

---

### Slide 17: Production Architecture
**Title:** CRAG in the Real World

**Bullet Points:**
- Evaluation model: Can use smaller, faster LLM
- Web search: Tavily, Serper, or custom API
- Caching: Cache evaluations for repeated queries
- Monitoring: Track decision distribution over time
- A/B testing: Compare CRAG vs standard RAG
- Fallback chain: Web → cached answers → "I don't know"

**Suggested Visual:** Production architecture diagram with caching layer and monitoring

---

### Slide 18: Lab 9 Summary
**Title:** Key Takeaways

**Bullet Points:**
- **Self-awareness is key**: Systems must know their limits
- **Evaluate before generate**: Check retrieval quality first
- **Graceful fallback**: Web search beats hallucination
- **Three decisions**: CORRECT / AMBIGUOUS / INCORRECT
- **Knowledge refinement**: Extract only what matters
- **Audit everything**: Enterprise needs demand transparency
- **Tune thresholds**: Adjust for your quality requirements

**Suggested Visual:** Summary icons for each key point

---

## APPENDIX: Additional Slides (Optional)

### Slide A1: CRAG Research Origins
**Title:** Academic Foundation

**Bullet Points:**
- Paper: "Corrective Retrieval Augmented Generation" (2024)
- Authors: Yan et al.
- Key contributions:
  - Retrieval evaluator concept
  - Web search augmentation
  - Knowledge refinement decomposition
- Builds on Self-RAG and other reflective approaches

---

### Slide A2: Related Techniques
**Title:** The Self-Correcting RAG Family

**Bullet Points:**
- **Self-RAG**: Generates reflection tokens during output
- **FLARE**: Forward-Looking Active Retrieval
- **CRAG**: Corrective action based on retrieval quality
- **Adaptive RAG**: Adjusts retrieval strategy dynamically
- All share theme: RAG systems that improve themselves

---

### Slide A3: Implementation Checklist
**Title:** Deploying CRAG in Production

**Bullet Points:**
- [ ] Implement retrieval grader with appropriate prompt
- [ ] Configure decision thresholds for use case
- [ ] Integrate web search API (with rate limiting)
- [ ] Add knowledge refinement step
- [ ] Capture full audit trail (CRAGResult)
- [ ] Set up monitoring for decision distribution
- [ ] A/B test against baseline RAG
- [ ] Tune thresholds based on production data
