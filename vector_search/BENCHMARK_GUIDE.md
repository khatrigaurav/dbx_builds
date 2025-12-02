# Search Benchmark Guide - ANN vs Hybrid vs Full-Text

## 🎯 Overview

This guide explains how to use the `search_benchmark.py` notebook to compare three search methods for your medical research papers:

1. **ANN (Approximate Nearest Neighbor)** - Pure vector similarity search
2. **Hybrid Search** - Combines vector + keyword matching
3. **Full-Text Search** - Traditional keyword/SQL-based search

---

## 🚀 Quick Start

### Prerequisites
✅ Run `medical_research.py` first to create embeddings table and vector index

### Steps
1. Import `search_benchmark.py` to Databricks
2. Update configuration (Cell 2) if needed
3. Click "Run All"
4. Review results and recommendations

**Time**: ~5-10 minutes to complete

---

## 📊 What Gets Benchmarked

### 1. **Latency (Speed)**
- Average query time (milliseconds)
- Minimum and maximum response times
- Consistency across queries

### 2. **Result Quality**
- Number of results returned
- Overlap between methods (agreement)
- Relevance to query (qualitative)

### 3. **Reliability**
- Success rate (% of queries that complete)
- Error handling
- Edge case behavior

---

## 🔬 Search Methods Explained

### ANN (Vector Search)

**How it works:**
```python
# Converts query to embedding vector
# Finds nearest neighbor documents in vector space
query = "What treatments for diabetes?"
→ embedding = [0.23, -0.45, 0.89, ...]
→ finds similar vectors in index
→ returns most similar documents
```

**Strengths:**
- ✅ Semantic understanding (concepts, not just keywords)
- ✅ Works with natural language questions
- ✅ Finds related content even without exact keywords
- ✅ Fast with proper indexing

**Weaknesses:**
- ❌ May miss exact keyword matches
- ❌ Requires embedding generation (compute cost)
- ❌ "Black box" - hard to explain why results match

**Best for:**
- Natural language questions
- Conceptual similarity
- Research discovery
- When terminology varies

---

### Full-Text Search

**How it works:**
```python
# Extracts keywords from query
# Searches for exact matches in text
# Scores by keyword frequency
query = "What treatments for diabetes?"
→ keywords = ["treatments", "diabetes"]
→ SQL: WHERE chunk_text LIKE '%treatments%' 
           AND chunk_text LIKE '%diabetes%'
→ returns documents with most matches
```

**Strengths:**
- ✅ Exact keyword matching
- ✅ Fast for simple queries
- ✅ Transparent (know why results match)
- ✅ No embedding required
- ✅ Works well with technical terms

**Weaknesses:**
- ❌ No semantic understanding
- ❌ Misses synonyms/related concepts
- ❌ Poor with misspellings
- ❌ Struggles with natural language questions

**Best for:**
- Keyword lookup (drug names, procedures)
- Technical terminology search
- Exact phrase matching
- Simple, fast queries

---

### Hybrid Search

**How it works:**
```python
# Combines vector + keyword approaches
# Weighted scoring
query = "What treatments for diabetes?"
→ vector_score = 0.85 (semantic similarity)
→ keyword_score = 0.60 (2/3 keywords match)
→ hybrid_score = 0.7 * 0.85 + 0.3 * 0.60 = 0.775
→ reranks results by hybrid score
```

**Strengths:**
- ✅ Best of both worlds
- ✅ Semantic + keyword precision
- ✅ Tunable (adjust weights)
- ✅ Often best overall results

**Weaknesses:**
- ❌ Slower (runs both searches)
- ❌ More complex to tune
- ❌ Requires both systems

**Best for:**
- Production systems
- Mixed query types
- When accuracy is critical (medical!)
- RAG applications

---

## 📈 Understanding the Results

### Latency Charts

**Box Plot:**
```
Shows distribution of query times
• Box: 25th-75th percentile
• Line: Median
• Whiskers: Min/max
• Look for: Lower is better, tighter is more consistent
```

**Bar Chart by Category:**
```
Compares methods across query types
• Treatment queries
• Diagnosis queries
• Safety queries
• Look for: Which method is fastest for your use case
```

### Overlap Analysis

**What it means:**
```
ANN ∩ Full-Text = 5
→ 5 results appear in both methods
→ High overlap = methods agree
→ Low overlap = different perspectives
```

**Interpretation:**
- **High overlap (7-10)**: Methods agree on relevance
- **Medium overlap (4-6)**: Complementary results
- **Low overlap (0-3)**: Very different results (review quality)

### Sample Output

```
Method          Success Rate    Avg Latency    Min/Max Latency    Avg Results
--------------------------------------------------------------------------------
ANN             100.0%          156.45ms       98.23 / 234.56ms   10.0
Full-Text       100.0%          423.12ms       312.45 / 567.89ms  8.5
Hybrid          100.0%          198.34ms       145.67 / 289.12ms  10.0

🏆 WINNER BY CATEGORY
⚡ Fastest: ANN (156.45ms avg)
📊 Most Consistent: ANN (136.33ms variance)
✅ Most Reliable: All methods (100.0% success)
```

**What this tells you:**
- ANN is fastest (~156ms)
- Full-Text is slower (~423ms) - needs optimization
- Hybrid adds ~40ms overhead but acceptable
- All methods reliable (100% success)
- **Recommendation**: Use Hybrid for best quality, ANN if speed critical

---

## 🎯 Decision Matrix

| Scenario | Recommended Method | Why |
|----------|-------------------|-----|
| **Medical Q&A Chatbot** | Hybrid (0.7/0.3) | Accuracy critical, semantic + terminology |
| **Drug Name Lookup** | Full-Text | Exact matches for drug names |
| **Research Discovery** | ANN | Find conceptually similar papers |
| **Symptom Checker** | Hybrid (0.6/0.4) | Mix of keywords and concepts |
| **Clinical Guidelines** | Full-Text or Hybrid | Specific terminology important |
| **Patient Education** | ANN | Natural language questions |
| **Literature Review** | ANN | Explore related research |
| **Adverse Event Search** | Full-Text | Exact term matching critical |

---

## ⚙️ Tuning Hybrid Search

### Weight Configuration

```python
# In search_benchmark.py, modify search_hybrid():

# More semantic, less keyword
search_hybrid(query, vector_weight=0.8, keyword_weight=0.2)

# Balanced
search_hybrid(query, vector_weight=0.6, keyword_weight=0.4)

# More keyword, less semantic
search_hybrid(query, vector_weight=0.4, keyword_weight=0.6)
```

### When to Adjust

**Increase Vector Weight (0.8+) when:**
- Queries are natural language
- Terminology varies
- Conceptual similarity matters
- Users ask "how/why/what" questions

**Increase Keyword Weight (0.4+) when:**
- Looking for specific terms
- Medical terminology is standard
- Exact matching critical
- Users search by drug/procedure names

**Balanced (0.5-0.6) when:**
- Mixed query types
- General purpose search
- Unsure of user behavior
- Starting point for tuning

### Example Tuning Process

```python
# Test different weights
weights_to_test = [
    (0.9, 0.1),  # Heavy semantic
    (0.7, 0.3),  # Default
    (0.5, 0.5),  # Balanced
    (0.3, 0.7),  # Heavy keyword
]

for v_weight, k_weight in weights_to_test:
    results = search_hybrid(
        "diabetes treatment",
        vector_weight=v_weight,
        keyword_weight=k_weight
    )
    print(f"Weights {v_weight}/{k_weight}: {results['num_results']} results")
    # Review quality manually
```

---

## 🔍 Interpreting Query Comparisons

### Sample Comparison

```
QUERY: What are the most effective treatments for long COVID symptoms?

ANN SEARCH RESULTS
Latency: 145.23ms
Results: 10

1. Score: 0.8956
   Text: "Long COVID treatment strategies include rehabilitation,
          symptom management, and gradual return to activity..."
   
2. Score: 0.8734
   Text: "Effective interventions for persistent COVID-19 symptoms
          include physical therapy, cognitive rehabilitation..."

FULL-TEXT SEARCH RESULTS
Latency: 387.45ms
Results: 8

1. Score: 3 matches
   Text: "Long COVID symptoms treatment overview: pharmacological
          and non-pharmacological interventions..."
          
2. Score: 3 matches
   Text: "Treatment of long COVID: evidence-based approaches for
          managing symptoms including fatigue, breathlessness..."

HYBRID SEARCH RESULTS
Latency: 178.56ms
Results: 10

1. Score: 0.8967 (Vector: 0.8956, Keyword: 0.9000, Matches: 3)
   Text: "Long COVID treatment strategies include rehabilitation..."
   
2. Score: 0.8820 (Vector: 0.8734, Keyword: 0.9200, Matches: 3)
   Text: "Effective interventions for persistent COVID-19 symptoms..."
```

**Analysis:**
- ANN finds semantically relevant papers
- Full-Text finds exact keyword matches
- Hybrid combines both, reranking by combined score
- Hybrid top result = same as ANN #1 but boosted by keywords
- Full-Text slower (no index optimization)

---

## 📊 Typical Performance Ranges

### Expected Latencies (10 results)

| Method | Typical Range | Good | Needs Optimization |
|--------|---------------|------|-------------------|
| ANN | 50-200ms | <150ms | >300ms |
| Full-Text | 100-500ms | <300ms | >800ms |
| Hybrid | 100-300ms | <250ms | >500ms |

**Factors affecting latency:**
- Dataset size (more papers = slower full-text)
- Cluster size (bigger = faster)
- Number of results requested
- Query complexity
- Index optimization
- Network latency

---

## 🚀 Optimization Tips

### For ANN Search

```python
# 1. Ensure index is warmed up
for _ in range(5):
    index.similarity_search("warm up", num_results=1)

# 2. Request fewer results initially
results = search_ann(query, num_results=5)  # Instead of 20

# 3. Use filters to narrow search space
results = index.similarity_search(
    query_text=query,
    filters={"pub_year": {"$gte": "2020"}},  # If metadata available
    num_results=10
)
```

### For Full-Text Search

```python
# 1. Create text search index
spark.sql(f"""
    CREATE INDEX idx_chunk_text 
    ON {SILVER_EMBEDDINGS_TABLE} (chunk_text)
""")

# 2. Use more selective keywords
keywords = [w for w in query.split() if len(w) > 5]  # Longer words only

# 3. Add filters before text search
df = spark.table(SILVER_EMBEDDINGS_TABLE)
df = df.filter(col("chunk_length") > 200)  # Pre-filter
# Then do text search
```

### For Hybrid Search

```python
# 1. Get fewer ANN candidates
ann_results = search_ann(query, num_results=num_results * 2)  # Not *5

# 2. Cache ANN index in memory
index.describe()  # Warms up index

# 3. Parallel execution (advanced)
import concurrent.futures

with concurrent.futures.ThreadPoolExecutor() as executor:
    future_ann = executor.submit(search_ann, query)
    future_ft = executor.submit(search_fulltext, query)
    ann_results = future_ann.result()
    ft_results = future_ft.result()
# Combine results
```

---

## 🎓 Real-World Examples

### Example 1: Medical Chatbot

**Query**: "What are the side effects of metformin?"

**Analysis:**
- ANN: Finds papers about metformin safety, adverse events
- Full-Text: Finds exact "side effects" + "metformin" matches
- Hybrid: Best - combines semantic understanding with exact terms

**Recommendation**: Hybrid (0.6/0.4)
- Vector weight 0.6: Understand "side effects" = "adverse events"
- Keyword weight 0.4: Must include "metformin"

---

### Example 2: Research Discovery

**Query**: "Novel approaches to cancer immunotherapy"

**Analysis:**
- ANN: Finds related concepts, new techniques, even if terms differ
- Full-Text: Only finds exact "immunotherapy" mentions
- Hybrid: Good middle ground

**Recommendation**: ANN (pure vector)
- Goal is discovery, not precision
- Want to find related concepts
- Keywords might miss novel approaches with different terminology

---

### Example 3: Drug Interaction Check

**Query**: "warfarin aspirin interaction"

**Analysis:**
- ANN: Might find general drug interaction papers
- Full-Text: Finds exact drug names mentioned together
- Hybrid: Balanced

**Recommendation**: Full-Text or Hybrid (0.3/0.7)
- Critical to find EXACT drug names
- Safety-critical application
- High keyword weight ensures precision

---

## 📋 Benchmark Checklist

Before running benchmark:
- [ ] `medical_research.py` completed successfully
- [ ] Embeddings table has data (check count)
- [ ] Vector index is online (check status)
- [ ] Test queries are relevant to your use case
- [ ] Configuration matches your setup

After running benchmark:
- [ ] All methods completed successfully
- [ ] Latency results are reasonable (<1s)
- [ ] Results overlap makes sense
- [ ] Reviewed sample query comparisons
- [ ] Identified best method for your use case
- [ ] Documented decision and reasoning

---

## 🐛 Troubleshooting

### "Vector index not found"
**Solution**: Run `medical_research.py` first, wait for index to be online

### "Full-text search very slow (>2s)"
**Solution**: 
- Dataset too large for unindexed search
- Consider creating text index on chunk_text column
- Or filter data before searching

### "Hybrid search returning same results as ANN"
**Solution**:
- Keywords not matching any documents
- Increase keyword_weight
- Check keyword extraction logic

### "No results from Full-Text"
**Solution**:
- Query terms too specific
- Try broader keywords
- Check for spelling/terminology

### "Inconsistent latencies"
**Solution**:
- Cold start effect (first query slow)
- Run warm-up queries first
- Increase BENCHMARK_ITERATIONS

---

## 📈 Monitoring in Production

### Metrics to Track

```python
# Log these for every search query
{
    "timestamp": "2024-11-25T10:30:00",
    "query": "diabetes treatment",
    "method": "hybrid",
    "latency_ms": 156.4,
    "num_results": 10,
    "user_clicked": True,  # Did user click any result?
    "click_position": 1,  # Which result?
    "user_satisfied": True  # Did they find answer?
}
```

### Dashboards

**Key metrics:**
- P50, P95, P99 latency by method
- Success rate
- Results per query
- User engagement (click-through rate)
- Query types (categorize automatically)

### Alerts

Set up alerts for:
- Latency > 1000ms (performance degradation)
- Success rate < 95% (system issues)
- No results > 20% (index or data problems)

---

## 🎯 Next Steps

After running benchmark:

1. **Document your findings**
   - Which method performed best?
   - What latencies did you see?
   - Any surprising results?

2. **Choose your method**
   - Based on use case and results
   - Consider backup method if primary fails

3. **Implement in production**
   - Start with chosen method
   - Add monitoring
   - Collect user feedback

4. **Iterate and improve**
   - Tune hybrid weights based on usage
   - Add query preprocessing
   - Optimize slow queries

5. **A/B test**
   - Test different methods with users
   - Measure satisfaction
   - Adjust based on data

---

## 📚 Additional Resources

- **Main Pipeline**: `medical_research.py`
- **Setup Guide**: `MEDICAL_RESEARCH_GUIDE.md`
- **Architecture**: `ARCHITECTURE.md`
- **Troubleshooting**: `TROUBLESHOOTING.md`

---

**Questions?** Review the benchmark results carefully and choose the method that best fits your specific use case and performance requirements!

