# Slide Outline: Labs 7 & 8 - Advanced RAG Techniques
## RAG Evaluation & Query Transformation/Re-ranking

---

## SECTION 1: RAG Evaluation and Quality Metrics (Lab 7)

### Slide 1: Why RAG Evaluation Matters for Enterprise
**Title:** Measuring RAG Quality: The Enterprise Imperative

**Bullet Points:**
- Production RAG systems need reliability guarantees
- Regulatory compliance requires answer traceability
- A/B testing different configurations needs metrics
- Continuous improvement requires measurable baselines
- Customer trust depends on consistent accuracy

**Suggested Visual:** Dashboard mockup showing RAG quality metrics over time (line chart trending upward with quality scores)

---

### Slide 2: The RAG Quality Framework
**Title:** Four Dimensions of RAG Quality

**Bullet Points:**
- **Retrieval Quality** - Are we finding the right documents?
  - Context Relevance: Do chunks relate to the question?
  - Retrieval Precision: % of chunks actually useful

- **Answer Quality** - Is the response accurate?
  - Groundedness: Answer supported by context?
  - Completeness: All aspects addressed?

- **Hallucination Risk** - Is the LLM making things up?
  - Factual consistency with source material
  - Unsupported claims detection

**Suggested Visual:** 2x2 matrix diagram with the four quality dimensions, color-coded by importance

---

### Slide 3: Context Relevance Evaluation
**Title:** Measuring Retrieval Quality

**Bullet Points:**
- Goal: Verify retrieved chunks relate to the question
- Approach: "LLM-as-judge" - use AI to rate relevance
- Scale: 1-5 rating normalized to 0.0-1.0 score
- Handles semantic similarity (not just keyword match)
- Robust to paraphrasing and synonyms

**Suggested Visual:** Diagram showing query → retrieved chunks → LLM judge → relevance scores

**Code Snippet Example:**
```python
# LLM evaluates: "Rate relevance 1-5"
# Returns normalized score 0.0-1.0
context_relevance = evaluate_context_relevance(question, chunks)
```

---

### Slide 4: Answer Groundedness - Detecting Hallucinations
**Title:** Is the Answer Grounded in Facts?

**Bullet Points:**
- Critical metric: Can every claim be traced to source?
- LLM analyzes: "Is this supported by the context?"
- Identifies specific unsupported claims
- Inverse = hallucination risk score
- Enterprise priority: Groundedness weighted at 40%

**Suggested Visual:** Side-by-side comparison showing "grounded answer" (with source highlights) vs "hallucinated answer" (with red warning flags)

---

### Slide 5: The LLM-as-Judge Pattern
**Title:** Using AI to Evaluate AI

**Bullet Points:**
- Pattern: Use one LLM to evaluate another's output
- Benefits:
  - Understands semantic meaning (not just overlap)
  - Handles complex evaluation criteria
  - More interpretable than embedding similarity
- Structured prompts ensure consistent scoring
- Temperature=0.1 for deterministic evaluation

**Suggested Visual:** Flow diagram: Question → RAG System → Answer → Evaluator LLM → Scores

---

### Slide 6: Automated Test Suites
**Title:** Regression Testing for RAG Systems

**Bullet Points:**
- Define test cases with ground truth expectations
- Run automated evaluation on each question
- Aggregate metrics for system-level assessment
- Track changes over time (version comparison)
- Set quality gates for production deployment

**Suggested Visual:** Table showing test cases with expected keywords, actual scores, and pass/fail status

**Example Test Case:**
```python
{
    "question": "What is the return window?",
    "expected_keywords": ["30 days", "return"],
    "category": "policy"
}
```

---

### Slide 7: Enterprise Evaluation Scorecard
**Title:** Weighted Overall Quality Score

**Bullet Points:**
- Not all metrics equally important
- Enterprise weighting reflects priorities:
  - Groundedness: 40% (can't make things up)
  - Context Relevance: 30% (right docs retrieved)
  - Completeness: 20% (answer the full question)
  - Precision: 10% (efficiency of retrieval)
- Quality thresholds:
  - GREEN (0.8+): Production ready
  - YELLOW (0.6-0.8): Needs improvement
  - RED (<0.6): Not production ready

**Suggested Visual:** Colored gauge/dashboard showing overall score with component breakdown

---

## SECTION 2: Query Transformation and Re-ranking (Lab 8)

### Slide 8: The Query-Document Gap
**Title:** Why Basic RAG Retrieval Falls Short

**Bullet Points:**
- Users ask questions; documents contain answers
- Vocabulary mismatch: "refund" vs "money back"
- Ambiguous queries: "broken device" - warranty? repair? return?
- Short queries lack context for good matching
- Vector similarity alone may miss relevant content

**Suggested Visual:** Venn diagram showing query terms vs document terms with small overlap, labeled "The Gap"

---

### Slide 9: Query Transformation Overview
**Title:** Bridging the Query-Document Gap

**Bullet Points:**
- Transform user queries for better retrieval
- Three main techniques:
  1. **Query Expansion** - Add synonyms
  2. **Multi-Query** - Generate variations
  3. **HyDE** - Search with hypothetical answers
- Each addresses different aspects of the gap
- Can be combined for maximum effectiveness

**Suggested Visual:** Flowchart showing original query branching into three transformation paths

---

### Slide 10: Query Expansion
**Title:** Adding Synonyms and Related Terms

**Bullet Points:**
- Use LLM to identify synonyms and related terms
- Expands search coverage without losing focus
- Example:
  - Input: "refund"
  - Output: "refund return money back reimbursement"
- Helps with vocabulary mismatch
- Simple, fast, low overhead

**Suggested Visual:** Word cloud centered on "refund" with related terms radiating outward

**Code Pattern:**
```python
expanded = expand_query("refund")
# → "refund return money back reimbursement"
```

---

### Slide 11: Multi-Query Generation
**Title:** Multiple Perspectives on One Question

**Bullet Points:**
- Generate N variations of the user's query
- Each variation may match different relevant docs
- Aggregate results from all variations
- Boost scores for docs found by multiple queries
- Powerful for ambiguous or conversational queries

**Suggested Visual:** Single user question splitting into 3 arrows, each pointing to different document clusters

**Example:**
```
Input: "how to get money back"
Queries:
  1. "how to get money back"
  2. "refund process and policy"
  3. "return item for reimbursement"
```

---

### Slide 12: HyDE - Hypothetical Document Embedding
**Title:** The Most Powerful Query Transformation

**Bullet Points:**
- Key insight: Questions ≠ Answers semantically
- Process:
  1. Generate hypothetical ideal answer
  2. Use THAT answer as search query
  3. Find documents similar to hypothetical
- Bridges question-style to answer-style text
- Often finds most relevant passages

**Suggested Visual:** Two-step diagram: Question → LLM → Hypothetical Answer → Vector Search → Real Documents

**Example:**
```
Question: "What is the refund policy?"
Hypothetical: "Customers may return products within
30 days for a full refund. Items must be..."
→ Search finds documents containing actual policy
```

---

### Slide 13: Two-Stage Retrieval with Re-ranking
**Title:** Fast Recall + Precise Ranking

**Bullet Points:**
- Problem: Vector search is fast but sometimes imprecise
- Solution: Two stages
  - Stage 1: Fast retrieval of MORE candidates (e.g., 10)
  - Stage 2: Accurate re-scoring of candidates
  - Return only top K after re-ranking
- Re-ranker can use cross-attention (more compute per doc)
- Dramatically improves precision@k

**Suggested Visual:** Funnel diagram: Many candidates → Re-ranker → Fewer, better results

---

### Slide 14: Re-ranking with LLM Scoring
**Title:** LLM-Based Relevance Scoring

**Bullet Points:**
- For each candidate, ask LLM: "How relevant is this?"
- 1-5 scale: Irrelevant → Perfect match
- More accurate than vector distance alone
- Trade-off: Slower, but only run on K candidates
- Production systems may use specialized cross-encoders

**Suggested Visual:** Table showing 6 candidates with original scores, LLM relevance scores, and final ranking

**Code Pattern:**
```python
candidates = basic_retrieve(query, k=6)
reranked = rerank_chunks(query, candidates, top_k=3)
```

---

### Slide 15: Combining Techniques
**Title:** The Full Advanced Pipeline

**Bullet Points:**
- Best results: Combine multiple approaches
- Full pipeline:
  1. Generate multi-queries + HyDE
  2. Retrieve from all variations
  3. Deduplicate results
  4. Re-rank combined candidates
  5. Return top K
- Trade-off: Latency vs quality
- Configurable based on use case requirements

**Suggested Visual:** Complex flowchart showing all techniques merging into a single re-ranking stage

---

### Slide 16: Method Comparison
**Title:** Which Technique When?

**Bullet Points:**

| Technique | Best For | Trade-off |
|-----------|----------|-----------|
| Basic | Simple, direct queries | Baseline |
| Expansion | Vocabulary mismatch | Low overhead |
| Multi-Query | Ambiguous queries | Medium latency |
| HyDE | Complex questions | Higher latency |
| Re-ranking | Precision-critical | Per-doc compute |
| Combined | Maximum quality | Highest latency |

**Suggested Visual:** Comparison table or decision tree for choosing techniques

---

### Slide 17: Enterprise Impact
**Title:** Business Value of Advanced Retrieval

**Bullet Points:**
- Better answers from SAME knowledge base
- Handles ambiguous user queries gracefully
- Reduces "I don't know" responses
- Improves user satisfaction and trust
- Lower support escalation rates
- Competitive advantage in AI-powered products

**Suggested Visual:** Before/after metrics showing improvement in answer quality, user satisfaction

---

### Slide 18: Lab Summary
**Title:** Key Takeaways

**Bullet Points:**
- **Evaluation is essential** - You can't improve what you don't measure
- **Groundedness prevents hallucination** - Critical for enterprise trust
- **Query transformation bridges gaps** - Between how users ask and docs answer
- **HyDE is powerful** - Generates answer-like text to find answers
- **Re-ranking improves precision** - More compute on fewer docs
- **Combine for best results** - Multi-technique pipelines for production

**Suggested Visual:** Summary icons for each key point

---

## APPENDIX: Additional Slides (Optional)

### Slide A1: Production Considerations
**Title:** Deploying Advanced RAG in Production

**Bullet Points:**
- Latency budgets determine technique selection
- Cache transformed queries for common patterns
- A/B test different pipeline configurations
- Monitor evaluation metrics in production
- Set up alerts for quality degradation

---

### Slide A2: Tools and Frameworks
**Title:** RAG Evaluation Ecosystem

**Bullet Points:**
- RAGAS (Retrieval-Augmented Generation Assessment)
- LangSmith / LangChain evaluation
- TruLens
- Custom evaluation with local LLMs (this lab)
- Enterprise observability platforms

---

### Slide A3: Future Directions
**Title:** What's Next in RAG

**Bullet Points:**
- Agentic RAG with planning
- Self-reflective RAG (auto-correction)
- Multi-modal retrieval
- Fine-tuned re-rankers
- Learned query transformations
