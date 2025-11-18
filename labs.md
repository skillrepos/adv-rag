# Enterprise AI Accelerator - Day 0
## AI Concepts and Readiness
## Session labs 
## Revision 1.1 - 11/17/25
## (c) 2025 Tech Skills Transformations

**Follow the startup instructions in the README.md file IF NOT ALREADY DONE!**

**NOTES**
- To copy and paste in the codespace, you may need to use keyboard commands - CTRL-C and CTRL-V. Chrome may work best for this.
- If your codespace has to be restarted, run these commands again!
  ```
  ollama serve &
  ```

<br><br>

# Lab 1 - AI vs ML vs Generative AI vs Data Science: Understanding the Differences Through Practice

**Purpose: Learn the key differences between Data Science, Artificial Intelligence (AI), Machine Learning (ML), Deep Learning (DL), and Generative AI by seeing how each discipline approaches the same business problem. This hands-on lab uses practical examples and a local language model to demonstrate when and why you'd use each approach.

<br><br>

## The Scenario: E-Commerce Customer Analysis

You work for an online store and have this customer data:

```
Customer Purchase History (Last 6 Months):
- Customer A: 5 purchases, avg $45, categories: electronics, books
- Customer B: 2 purchases, avg $120, categories: clothing, accessories
- Customer C: 12 purchases, avg $25, categories: books, home goods
- Customer D: 1 purchase, $200, category: electronics
- Customer E: 8 purchases, avg $35, categories: books, toys

Recent behavior: Customer C browsed "kitchen gadgets" 3 times this week
```

We'll explore how **Data Science**, **Traditional AI**, **Machine Learning**, **Deep Learning**, and **Generative AI** each approach this data differently.

<br><br>

## 1. Data Science Approach - "What Happened?"

**Data Science** focuses on extracting insights from data using statistics, visualization, and analysis. It answers: **"What happened and why?"**

**Set color to Blue for Data Science:**
```bash
ds-color
```

Ask the model to perform a data science analysis:

```bash
ollama run llama3.2:3b "You are a data scientist analyzing e-commerce data.

Dataset:
- Customer A: 5 purchases, avg \$45, categories: electronics, books
- Customer B: 2 purchases, avg \$120, categories: clothing, accessories
- Customer C: 12 purchases, avg \$25, categories: books, home goods
- Customer D: 1 purchase, \$200, category: electronics
- Customer E: 8 purchases, avg \$35, categories: books, toys

Perform descriptive statistical analysis:
1. Calculate: Average purchase frequency, average order value, most popular category
2. Identify: Customer segments (high/medium/low engagement)
3. Find: Patterns and correlations

Provide statistical summary in under 100 words."
```

**Observe:** Data Science focuses on **descriptive and diagnostic analytics** - understanding what the data tells us about past behavior. 

**Key Concept:** Data Science uses statistical methods, data cleaning, visualization, and analysis to extract insights from existing data.

<br><br>

## 2. Traditional AI Approach - "Apply Logic Rules"

**Traditional AI** (also called symbolic AI or rule-based AI) uses human-defined rules and logic. It answers: **"What should we do based on predefined rules?"

**Set color to Magenta for Traditional AI:**
```bash
ai-color
```

Ask the model to create rule-based logic:

```bash
ollama run llama3.2:3b "You are designing a traditional rule-based AI system for customer classification.

Create explicit IF-THEN rules to categorize customers:

Rules format:
IF [condition] THEN [action/classification]

Consider:
- Purchase frequency (high: 8+, medium: 3-7, low: 1-2)
- Average order value (high: \$100+, medium: \$30-99, low: <\$30)
- Category preferences

Create 5 specific rules for customer classification and recommended actions. Be precise and deterministic."
```

**Observe:** Traditional AI uses **explicit, human-coded rules**. It's deterministic, transparent, but inflexible. 

**Key Concept:** Traditional AI relies on human experts to define all the rules. Good for well-defined problems with clear logic.

<br><br>

## 3. Machine Learning Approach - "What Will Happen?"

**Machine Learning** learns patterns from data automatically and makes predictions. It answers: **"What is likely to happen next?"

**Set color to Green for Machine Learning:**
```bash
ml-color
```

Ask the model to explain ML prediction:

```bash
ollama run llama3.2:3b "You are explaining how a machine learning model would approach customer prediction.

Training data: Customer A-E purchase history

Explain in simple terms:
1. What patterns would an ML model learn from this data? (e.g., purchase frequency → category preferences)
2. Given: Customer C browsed 'kitchen gadgets' 3 times this week
   What would the ML model predict? (probability of purchase, recommended products)
3. How is this different from rule-based AI?

Keep explanation under 100 words, focus on 'learning from patterns' vs 'following rules'."
```

**Observe:** ML **automatically discovers patterns** in data without explicit programming. 

**Key Concept:** ML uses algorithms to learn from historical data and make predictions. It adapts as more data becomes available.

<br><br>

## 4: Deep Learning Approach - "Understand Complex Patterns"

**Deep Learning** is a subset of ML using neural networks with many layers. It excels at finding complex patterns in unstructured data. It answers: **"Can you understand this complex, unstructured information?"

**Set color to Cyan for Deep Learning:**
```bash
dl-color
```

Ask the model to explain deep learning capabilities:

```bash
ollama run llama3.2:3b "You are explaining deep learning in the context of e-commerce.

Traditional ML: Works well with structured data (tables, numbers)
Deep Learning: Excels with unstructured data (images, text, behavior sequences)

For our customer data scenario:
1. What could a deep learning model do that traditional ML cannot?
   (Hint: analyzing product images, understanding review sentiment, sequential behavior patterns)
2. Give 2 specific examples using neural networks
3. Explain why it needs more data and computing power

Keep under 100 words. Focus on 'complex patterns in unstructured data'."
```

**Observe:** Deep Learning uses **neural networks** to automatically learn hierarchical representations. 

**Key Concept:** Deep Learning is ML with neural networks that have multiple layers. It's what powers modern image recognition, natural language processing, and enables Generative AI.

<br><br>

## 5. Generative AI Approach - "Create Something New"

**Generative AI** creates new, original content based on patterns learned from training data. It answers: **"Can you generate something new and creative?"**

**Set color to Yellow for Generative AI:**
```bash
gen-color
```

Now use the actual Generative AI model to CREATE content:

```bash
ollama run llama3.2:3b "You are a generative AI creating personalized marketing content.

Customer Profile: Customer C
- 12 purchases in 6 months (highly engaged)
- Avg \$25 per order (budget-conscious)
- Categories: books, home goods
- Recent behavior: Browsed 'kitchen gadgets' 3 times

Generate:
1. A personalized email subject line (creative, engaging)
2. A 2-sentence email body recommending kitchen gadgets
3. A product bundle name that combines their interests

Be creative and personalized. This is NEW content, not analysis."
```

**Observe:** Generative AI **creates original content** - text, images, code, etc. 

**Key Concept:** Generative AI produces novel content rather than analyzing or predicting. It's built on deep learning architectures. It's what you're using right now in this lab!

<br><br>

## 6. Understanding the Hierarchy

Let's clarify how these concepts relate:

**Reset to white for neutral explanations:**
```bash
white
```

```bash
ollama run llama3.2:3b "Explain the relationship between AI, Machine Learning, Deep Learning, and Generative AI in a hierarchy.

Use this format:
- Artificial Intelligence (AI) is: [definition]
  - Machine Learning (ML) is: [subset definition]
    - Deep Learning (DL) is: [subset definition]
      - Generative AI is: [subset definition]

Where does Data Science fit? [separate or overlapping?]

Use a simple analogy (like Russian nesting dolls or tools in a toolbox). Keep under 100 words."
```

**Observe:**
- **AI** is the broadest concept (any system exhibiting intelligent behavior)
- **ML** is a subset of AI (systems that learn from data)
- **Deep Learning** is a subset of ML (using neural networks with multiple layers)
- **Generative AI** is a subset of Deep Learning (systems that create new content)
- **Data Science** is a separate discipline that overlaps with all of them

<br><br>

## 7. Compare and Decide - When to Use What?

Let's see practical guidance on choosing the right approach:

```bash
ollama run llama3.2:3b "For each scenario, identify whether Data Science, Traditional AI, Machine Learning, Deep Learning, or Generative AI is the BEST fit:

Scenario 1: Understanding why sales dropped last quarter
Scenario 2: Automatically flagging fraudulent transactions in real-time
Scenario 3: Writing personalized product descriptions for 10,000 items
Scenario 4: Sorting emails into folders based on fixed company policies
Scenario 5: Predicting which customers will cancel their subscription
Scenario 6: Analyzing customer selfies to recommend clothing sizes

Format:
Scenario X: [Answer] - [One sentence why]

Be specific about which approach and explain the reasoning."
```

**Observe:** Each discipline excels at different types of problems. The key is matching the tool to the task.

<br><br>


<br><br>

## Bonus: Test Your Understanding

Try these challenges:

1. **Your task**: Identify duplicate customer records in a database
   - **Which approach?** (Hint: Think about rules vs learning)

2. **Your task**: Create 100 unique social media posts about your product
   - **Which approach?** (Hint: What creates new content?)

3. **Your task**: Figure out which marketing campaign worked best last month
   - **Which approach?** (Hint: What analyzes historical data?)

4. **Your task**: Predict customer lifetime value for the next year
   - **Which approach?** (Hint: What predicts future outcomes?)

5. **Your task**: Analyze product images to automatically detect defects in manufacturing
   - **Which approach?** (Hint: What handles unstructured visual data?)

### Answers:
1. Traditional AI (rule-based matching) or ML (fuzzy matching) 
2. Generative AI (content creation)
3. Data Science (historical analysis) 
4. Machine Learning (predictive modeling) 
5. Deep Learning (image recognition with neural networks) 



<p align="center">
<b>[END OF LAB]</b>
</p>
</br></br>


# Lab #2 - Prompt Engineering: Advanced Techniques for Product Review Analysis**

**Purpose: Learn how different prompting strategies dramatically affect model accuracy and consistency by analyzing product reviews. This lab demonstrates key prompt engineering techniques including role-based prompting, chain-of-thought reasoning, structured templates, and meta-prompting for real-world sentiment analysis.

<br><br>

## Test Review (Used Throughout Lab)

We'll analyze this product review with mixed sentiment:
```
"I bought this wireless mouse three weeks ago and I'm really disappointed.
The battery life is terrible - it dies after just 2 days even though they
claim 6 months. However, I will say the ergonomic design is comfortable
and the price was reasonable at $25. The Bluetooth connectivity drops
frequently which makes it unusable for gaming. I'm considering returning it."
```

<br><br>

## 1: Basic Natural Language Prompting

Run a simple, unstructured prompt:

```bash
ollama run llama3.2:3b "Analyze the sentiment of this product review: 'I bought this wireless mouse three weeks ago and I'm really disappointed. The battery life is terrible - it dies after just 2 days even though they claim 6 months. However, I will say the ergonomic design is comfortable and the price was reasonable at \$25. The Bluetooth connectivity drops frequently which makes it unusable for gaming. I'm considering returning it.'"
```

**Observe:** Vague instructions lead to inconsistent, verbose responses without clear structure.

<br><br>

## 2: Role-Based Prompting with Domain Expertise

Add expert role and context:

```bash
ollama run llama3.2:3b "You are a senior e-commerce analyst with 10 years of experience analyzing customer feedback for product improvement and quality assurance. Your specialty is identifying actionable insights from reviews.

Analyze this product review and provide sentiment classification with key insights:

Review: 'I bought this wireless mouse three weeks ago and I'm really disappointed. The battery life is terrible - it dies after just 2 days even though they claim 6 months. However, I will say the ergonomic design is comfortable and the price was reasonable at \$25. The Bluetooth connectivity drops frequently which makes it unusable for gaming. I'm considering returning it.'

Provide: Overall sentiment, critical issues, and purchase recommendation."
```

**Observe:** Role assignment improves domain-appropriate analysis and professional tone.

<br><br>

## 3: Few-Shot with Chain-of-Thought Reasoning

Provide examples that show explicit reasoning steps:

```bash
ollama run llama3.2:3b "Analyze product reviews by thinking through each aspect step-by-step.

Example 1:
Review: 'Amazing headphones! Crystal clear sound and super comfortable for long sessions.'
Reasoning: Customer uses strong positive language ('amazing', 'crystal clear', 'super comfortable'). No negatives mentioned. Clear satisfaction.
Sentiment: POSITIVE

Example 2:
Review: 'The design is sleek but it broke after one week. Waste of money.'
Reasoning: One minor positive (design) but major negative (broke quickly, waste of money). Dissatisfaction dominates.
Sentiment: NEGATIVE

Example 3:
Review: 'Good value for the price. Not the best quality but does the job.'
Reasoning: Balanced view - acknowledges limitations but satisfied with price-performance ratio.
Sentiment: MIXED

Now analyze:
Review: 'I bought this wireless mouse three weeks ago and I'm really disappointed. The battery life is terrible - it dies after just 2 days even though they claim 6 months. However, I will say the ergonomic design is comfortable and the price was reasonable at \$25. The Bluetooth connectivity drops frequently which makes it unusable for gaming. I'm considering returning it.'

Reasoning:"
```

**Observe:** Explicit reasoning improves accuracy and helps identify nuanced sentiments.

<br><br>

## 4: Structured Template + Constraint-Based Prompting

Use a systematic framework with explicit constraints for consistent, compliant output:

```bash
ollama run llama3.2:3b "Use this structured template with constraints to analyze the product review:

[CONTEXT]
System: Product Review Sentiment Analyzer
Categories: POSITIVE | NEGATIVE | MIXED
Task: Systematic sentiment analysis with strict compliance

[CONSTRAINTS]
- MUST use exactly ONE sentiment: POSITIVE, NEGATIVE, or MIXED
- MUST identify at least 2 positive and 2 negative aspects (or state 'none found')
- Final recommendation must be: BUY, AVOID, or CONSIDER
- Keep analysis under 50 words total

[INPUT]
Product Review: 'I bought this wireless mouse three weeks ago and I'm really disappointed. The battery life is terrible - it dies after just 2 days even though they claim 6 months. However, I will say the ergonomic design is comfortable and the price was reasonable at \$25. The Bluetooth connectivity drops frequently which makes it unusable for gaming. I'm considering returning it.'

[ANALYSIS FRAMEWORK]
1. Positive Aspects: <list findings>
2. Negative Aspects: <list findings>
3. Overall Sentiment: <POSITIVE|NEGATIVE|MIXED>
4. Recommendation: <BUY|AVOID|CONSIDER>

[OUTPUT]
Provide structured analysis following the framework above."
```

**Observe:** Combining structured frameworks with explicit constraints ensures systematic analysis AND compliance with requirements.

<br><br>

## 5: Zero-Shot Chain-of-Thought (Magic Phrase)

Trigger reasoning without providing examples:

```bash
ollama run llama3.2:3b "Analyze the sentiment of this product review:

'I bought this wireless mouse three weeks ago and I'm really disappointed. The battery life is terrible - it dies after just 2 days even though they claim 6 months. However, I will say the ergonomic design is comfortable and the price was reasonable at \$25. The Bluetooth connectivity drops frequently which makes it unusable for gaming. I'm considering returning it.'

Let's think step by step to determine the correct sentiment classification."
```

**Observe:** The phrase "Let's think step by step" triggers analytical reasoning without examples.

<br><br>

## 6: Compare Results and Key Takeaways

Review outputs from all five techniques:

| Technique | Purpose | Best For | Key Benefit |
|-----------|---------|----------|-------------|
| **Basic Natural Language** | Baseline | Quick exploration | Simple, fast |
| **Role-Based** | Domain expertise | Professional contexts | Authority, expertise |
| **Few-Shot + CoT** | Pattern learning + reasoning | Complex decisions | Accuracy through examples |
| **Structured + Constraints** | Systematic + compliant | Production systems | Consistency + compliance |
| **Zero-Shot CoT** | Reasoning without examples | General problems | No examples needed |

<br><br>

## Key Learnings

1. **Role-based prompting** adds domain expertise and professional context
2. **Chain-of-thought** (explicit reasoning) dramatically improves accuracy - works both with examples (few-shot) and without (zero-shot)
3. **Structured templates + constraints** ensure systematic analysis AND enforce business requirements
4. **"Let's think step by step"** is a powerful zero-shot reasoning trigger that works without examples
5. **Choose techniques based on use case**: Speed vs. accuracy vs. compliance needs

<br><br>

## Bonus Challenges (Optional)

Apply these techniques to other tasks:

1. **Entity Extraction**: Use structured templates to extract named entities (people, places, organizations) from news articles
2. **Code Review**: Apply role-based + constraint-based prompting for systematic code quality analysis
3. **Email Triage**: Combine few-shot + self-consistency for classifying support tickets by urgency
4. **Content Moderation**: Use chain-of-thought reasoning to explain moderation decisions

---

## Quick Reference: When to Use Each Technique

- **Just starting?** → Basic Natural Language (Step 1)
- **Need professional output?** → Role-Based (Step 2)
- **Need consistency with examples?** → Few-Shot + Chain-of-Thought (Step 3)
- **Production systems with compliance?** → Structured Templates + Constraints (Step 4)
- **No examples available?** → Zero-Shot CoT with "Let's think step by step" (Step 5)

<br><br>

## Additional Resources

- [Ollama Documentation](https://github.com/ollama/ollama)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- [Chain-of-Thought Paper](https://arxiv.org/abs/2201.11903)
- Try different models: `ollama list` and `ollama pull <model-name>`



<p align="center">
<b>[END OF LAB]</b>
</p>
</br></br>


# Lab #3 - Interacting with AI Models**

**Purpose: Learn how to use different approaches to interact with AI models.**

<br><br>

## 1. The Ollama app is already installed as part of the codespace setup. To see the different options Ollama makes available for working with models, you can run the first second command below in the *TERMINAL*. 

```
ollama
```

<br><br>

## 2. We can also look at what models are available currently for us to use with the command below. 

```
ollama list
```

## 3. You'll see that it has the llama3.2:3b model available. Let's find out more about this model. Go to https://ollama.com and in the *Search models* box at the top, enter *llama*. In the list that pops up, choose the entry for "llama3.2".

![searching for llama](./images/31ai7.png?raw=true "searching for llama")

<br><br>

## 4. This will put you on the specific page about that model. Scroll down and scan the various information available about this model.
![reading about llama3.2](./images/31ai8.png?raw=true "reading about llama3.2")

<br><br>

## 5. Let's run the model in interactive mode with the command below. This will load it and make it available to query/prompt. 

```
ollama run llama3.2
```

<br><br>

## 6. Now you can query the model by inputting text at the *>>>Send a message (/? for help)* prompt.  Let's ask it about what the weather is in Paris. What you'll see is it telling you that it doesn't have access to current weather data and suggesting some ways to gather it yourself.

```
What's the current weather in Paris?
```

![answer to weather prompt and response](./images/31ai10.png?raw=true "answer to weather prompt and response")

<br><br>

## 7. Now, let's try a call with the API. You can stop the current run with a Ctrl-D or switch to another terminal. Then put in the command below (or whatever simple prompt you want). 

```
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2",
  "prompt": "What causes weather changes?",
  "stream": false
}' | jq -r '.response'
```

<br><br>

## 8. This will take a minute or so to run. You should see a long text response . You can try out some other prompts/queries if you want.

![query response](./images/aiapps37.png?raw=true "Query response")

<br><br>

## 9. Now let's try a simple Python script that uses Ollama programmatically. We have a basic example script called `simple_ollama.py`. Take a look at it either via [**simple_ollama.py**](./simple_ollama.py) or via the command below.

```
code simple_ollama.py
```

You should see a simple script that:
- Imports the ChatOllama class from langchain_ollama
- Initializes the Ollama client with the llama3.2 model
- Takes user input
- Sends it to Ollama
- Displays the response

<br><br>

## 10. Now you can run the script with the command below. 

```
python simple_ollama.py
```

<br><br>

## 11. When prompted, enter a question like "What is the capital of France?" and press Enter. You should see the model's response printed to the terminal. This demonstrates how easy it is to integrate Ollama into a Python application. Feel free to try other prompts. 


<p align="center">
<b>[END OF LAB]</b>
</p>
</br></br>


# Lab #4 - Creating Your Own Intelligence Br**

**Purpose: Use AI to help keep you up-to-date on AI.**

<br><br>

## Extended Prompt

```
"Please provide me with a structured AI intelligence briefing covering the last 7-14 days. Format your response as an executive briefing with the following sections:
1. CRITICAL DEVELOPMENTS (What I Must Know)

Major AI model releases or significant updates
Regulatory changes or policy announcements
Security/safety incidents or concerns
Industry game-changers or disruptions

2. TECHNICAL ADVANCES (What's New)

New capabilities or benchmarks achieved
Research breakthroughs (summarize in 1-2 sentences each)
Open source releases worth noting
Performance improvements in existing tools

3. BUSINESS IMPACT (What Matters for Work)

New enterprise AI tools or features
Case studies of successful AI implementations
ROI data or efficiency metrics published
Vendor announcements (Microsoft, Google, AWS, etc.)

4. PRACTICAL APPLICATIONS (What I Can Use Now)

New tools I can try immediately (with links)
Prompt engineering techniques or tips discovered
Workflow improvements or automations
Cost-effective AI solutions released

5. INDUSTRY MOVEMENTS (Market Dynamics)

Major funding rounds or acquisitions
Partnership announcements
Competitive landscape shifts
Job market or skill demand changes

6. LOOKING AHEAD (What's Coming)

Upcoming releases or announcements to watch
Scheduled conferences or events
Predicted trends gaining momentum
Emerging concerns or opportunities

For each item, please:

Include the DATE it was announced/published
Provide a direct LINK to the source
Keep descriptions to 2-3 sentences max
Bold the most important items I shouldn't miss
Flag anything requiring immediate action with 🚨

End with a 'THREE THINGS TO DO THIS WEEK' section with specific, actionable recommendations based on the updates.
Focus on practical, business-relevant developments over purely academic research. Prioritize information from the last 14 days, and clearly mark anything older as 'BACKGROUND' if it's essential context."

```

<br><br>

## 1: Try out the simple one-line prompt

Open claude.ai, chatgpt.com, gemini.google.com or your preferred AI assistant and copy and paste the simplified prompt below (substituting in your role and industry in the appropriate places):

```
What are the 5 most important AI developments from the last week that a [YOUR ROLE] in [YOUR INDUSTRY] needs to know, with links to learn more?
```
<br><br>

## 2: Review the output and note what is useful/not useful.

<br><br>

## 3: Try out the extended prompt.

Copy and paste the extended prompt from the start of the lab into the AI assistant. 

<br><br>

## 4: Review the output and notice differences and details vs the simpler prompt. 

<br><br>

## 5: Run the extended prompt again but add the appropriate section (or create your own) for your role.

For Technical Practitioners
```
Include code examples, GitHub repos, and technical implementation details. Focus on tools I can integrate into development workflows.
```

For Business Leaders
```
Emphasize ROI metrics, competitive advantages, and strategic implications. Include cost comparisons and vendor assessments.
```
For Specific Industries
```
Filter for developments specifically relevant to [YOUR INDUSTRY: healthcare/finance/retail/etc.]. Include industry-specific use cases and compliance considerations.
```

## 6: Review the output with the additional details.

<br><br>

## 7: What would you change/add/delete to make the results better for you? Try out any changes you want to make to fine-tune and make this most useful.


<p align="center">
<b>[END OF LAB]</b>
</p>
</br></br>


<p align="center">
<b>For educational use only by the attendees of our workshops.</b>
</p>

<p align="center">
<b>(c) 2025 Tech Skills Transformations and Brent C. Laster. All rights reserved.</b>
</p>
