# Medical Research Papers - Setup Guide

## 🎉 What Changed

Your notebook has been **updated from MLB prospects to medical research papers** using **Europe PMC API**.

---

## 🔄 Key Changes Made

### 1. **Data Source** ✅
- **Before**: FanGraphs MLB draft prospects (web scraping)
- **After**: Europe PMC medical research papers (JSON API)
- **API**: `https://www.ebi.ac.uk/europepmc/webservices/rest/search`

### 2. **Configuration** ✅
- **Schema**: `mlb_prospects` → `medical_research`
- **Tables**: `bronze_prospects` → `bronze_papers`
- **Volume**: Downloads medical papers instead of HTML pages
- **Search Query**: Configurable via `SEARCH_QUERY` parameter

### 3. **Download Function** ✅
- **Before**: `download_fangraphs_data()` - web scraping with BeautifulSoup
- **After**: `download_medical_papers()` - clean JSON API calls
- Downloads papers with metadata (title, authors, journal, abstract)
- Saves both text and JSON formats

### 4. **AI Parse Schema** ✅
Updated to extract medical research fields:
- `paper_title` (instead of player_name)
- `research_topic` (instead of position)
- `methodology` (study design)
- `key_findings` (results)
- `population` (sample size)
- `intervention` (treatment studied)
- `outcome` (results)

### 5. **Vector Search** ✅
- **Endpoint**: `mlb_prospects_endpoint` → `medical_research_endpoint`
- **Index**: `mlb_prospects_index` → `medical_papers_index`
- **Test Query**: Changed to medical research question

### 6. **Helper Functions** ✅
- `search_prospects()` → `search_medical_papers()`
- Updated example queries for medical use cases

---

## 🚀 Quick Start (5 Steps)

### Step 1: Open the Notebook
```
File: mlb_prospects_vector_search.py
Import to Databricks workspace
```

### Step 2: Configure Your Search (Cell 3)
```python
# Line 33: Update if needed
CATALOG = "main"  # Your catalog
SCHEMA = "medical_research"  # Keep or change

# Line 44-45: Choose your research topic
SEARCH_QUERY = "long covid AND treatment"  # ← CUSTOMIZE THIS!
MAX_PAPERS = 100  # Number of papers to download

# Optional: Add more topics (Line 48-52)
ADDITIONAL_QUERIES = [
    "mRNA vaccine efficacy",
    "diabetes management",
    # Add your topics here
]
```

### Step 3: Run All Cells
```
Click "Run All" → Wait 5-10 minutes
```

### Step 4: Verify Success
Look for these messages:
```
✓ Downloaded X papers from Europe PMC
✓ Bronze table created: main.medical_research.bronze_papers
✓ Silver cleaned table created: ...
✓ Silver embeddings table created: ...
✓ Vector Search Index created: ...
```

### Step 5: Query Your Papers
```python
# Test it out!
from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()
index = vsc.get_index(
    endpoint_name="medical_research_endpoint",
    index_name="main.medical_research.medical_papers_index"
)

results = index.similarity_search(
    query_text="What are effective treatments for long COVID?",
    columns=["chunk_text", "file_name"],
    num_results=10
)
```

---

## 📊 What You'll Get

### Downloaded Files (in Volume)
```
/Volumes/main/medical_research/raw_papers/
├── paper_PMC9876543_20241125_123456.txt     # Formatted paper
├── paper_PMC9876543_20241125_123456.json    # Full metadata
├── paper_PMC9876544_20241125_123456.txt
├── paper_PMC9876544_20241125_123456.json
└── download_summary_20241125_123456.json    # Summary
```

### Delta Tables Created
1. **Bronze** (`bronze_papers`): Raw papers with metadata
2. **Silver 1** (`silver_papers_cleaned`): Cleaned and chunked
3. **Silver 2** (`silver_papers_embeddings`): With embeddings

### Vector Search Index
- Searchable by semantic similarity
- Direct Access (no Delta Sync)
- Ready for RAG applications

---

## 💡 Example Queries

### Medical Literature Search
```python
results = search_medical_papers(
    "What are the symptoms of long COVID?",
    num_results=10
)
display(results)
```

### Treatment Research
```python
results = index.similarity_search(
    query_text="clinical trials for cancer immunotherapy",
    columns=["chunk_text", "file_name"],
    num_results=5
)
```

### Drug Information
```python
results = search_medical_papers(
    "side effects of mRNA vaccines",
    num_results=15
)
```

### Disease Information
```python
results = search_medical_papers(
    "early detection methods for Alzheimer's disease",
    num_results=10
)
```

---

## 🔧 Customization Tips

### Change Research Topics
**Edit Cell 3 (Configuration)**:
```python
# Single topic
SEARCH_QUERY = "diabetes treatment AND insulin resistance"

# Multiple topics
ADDITIONAL_QUERIES = [
    "heart disease prevention",
    "mental health interventions",
    "pediatric cancer treatment"
]
```

### Adjust Paper Count
```python
MAX_PAPERS = 200  # Download more papers (max ~1000 per query)
```

### Filter by Date
```python
# In download function, add date filter
SEARCH_QUERY = "long covid AND SRC_DATE:2023-01-01:2024-12-31"
```

### Change Chunk Size
```python
CHUNK_SIZE = 800  # Longer chunks for more context
CHUNK_OVERLAP = 100  # More overlap
```

---

## 📈 Expected Performance

| Metric | Value |
|--------|-------|
| **Download Time** | 1-2 minutes (100 papers) |
| **Processing Time** | 5-10 minutes total |
| **Papers per Query** | Up to 1,000 |
| **Query Latency** | <500ms |
| **Storage** | ~50 MB per 100 papers |

---

## 🎯 Use Cases You Can Build

### 1. **Medical Literature Assistant**
```python
# RAG pattern
context = retrieve_relevant_papers(patient_question)
answer = llm.generate(prompt_with_context)
```

### 2. **Clinical Decision Support**
```python
# Help doctors find relevant research
papers = search_medical_papers(
    "contraindications for prescribing metformin",
    num_results=10
)
```

### 3. **Research Discovery Tool**
```python
# Find latest research on a topic
recent_papers = search_medical_papers(
    "novel biomarkers for early cancer detection",
    num_results=20
)
```

### 4. **Patient Education Chatbot**
```python
# Answer patient questions with evidence
def answer_patient_question(question):
    papers = search_medical_papers(question, num_results=5)
    context = extract_relevant_sections(papers)
    return generate_patient_friendly_answer(context)
```

### 5. **Drug Safety Monitor**
```python
# Track adverse events in literature
safety_papers = search_medical_papers(
    "adverse events AND [drug name]",
    num_results=50
)
```

---

## 🔍 Europe PMC API Features

### Available Fields
- Title, Abstract, Full Text
- Authors and Affiliations
- Journal Information
- Publication Date
- DOI, PMID, PMCID
- Keywords and MeSH Terms
- Citation Count
- Grant Information

### Search Operators
```python
# AND operator
"covid AND vaccine"

# OR operator
"cancer OR tumor"

# NOT operator
"diabetes NOT type1"

# Phrase search
'"long covid"'

# Field-specific
'AUTH:"Smith J" AND JOURNAL:"Nature"'

# Date range
"SRC_DATE:2023-01-01:2024-12-31"
```

### Advanced Queries
```python
# Complex query example
SEARCH_QUERY = '(cancer OR tumor) AND immunotherapy AND SRC_DATE:2023-01-01:2024-12-31'

# High impact journals only
SEARCH_QUERY = 'long covid AND (JOURNAL:"Nature" OR JOURNAL:"Science" OR JOURNAL:"Lancet")'

# Recent papers with many citations
SEARCH_QUERY = 'mRNA vaccine AND SRC_DATE:2023-01-01:2024-12-31'
```

---

## ⚠️ Important Notes

### Rate Limits
- No strict rate limit on Europe PMC
- Be respectful: don't hammer the API
- Add `time.sleep(1)` between requests for multiple queries

### Data Quality
- Open Access only (free to use)
- All papers are peer-reviewed
- Includes preprints from some sources
- Quality varies by journal

### Legal/Ethical
- ✅ Free for research and commercial use
- ✅ No API key required (but recommended)
- ✅ Include citations when using findings
- ⚠️ Not a substitute for professional medical advice
- ⚠️ Verify critical information with primary sources

---

## 🐛 Troubleshooting

### "No papers found"
**Solution**: Try a broader search query
```python
# Too specific
SEARCH_QUERY = "super rare disease AND specific mutation"

# Better
SEARCH_QUERY = "rare disease treatment"
```

### "API timeout"
**Solution**: Reduce `MAX_PAPERS` or check network
```python
MAX_PAPERS = 50  # Start smaller
```

### "Index returns no results"
**Solution**: Wait for index to sync (5-10 minutes)
```python
# Check index status
index.describe()
```

### "Out of memory"
**Solution**: Process in batches or increase cluster size
```python
# Process fewer papers at a time
MAX_PAPERS = 50  # Instead of 500
```

---

## 📚 Additional Resources

### Europe PMC Documentation
- API Guide: https://europepmc.org/RestfulWebService
- Search Syntax: https://europepmc.org/help
- Field Descriptions: https://europepmc.org/docs/EBI_Europe_PMC_Web_Service_Reference.pdf

### Medical Terminology
- MeSH Browser: https://meshb.nlm.nih.gov/
- Medical Subject Headings for better searches

### Similar Sources
- **PubMed Central**: https://www.ncbi.nlm.nih.gov/pmc/
- **arXiv (q-bio)**: https://arxiv.org/archive/q-bio
- **bioRxiv**: https://www.biorxiv.org/ (preprints)

---

## 🎓 Example Notebook Flow

```python
# 1. Configure (Edit these values)
SEARCH_QUERY = "Type 2 diabetes treatment"
MAX_PAPERS = 100

# 2. Run All Cells (5-10 minutes)

# 3. Query your index
results = search_medical_papers(
    "What are effective treatments for Type 2 diabetes?",
    num_results=10
)

# 4. Display results
display(results)

# 5. Build RAG application
def medical_qa(question):
    papers = search_medical_papers(question, 5)
    context = papers.select("chunk_text").collect()
    # Send to LLM with context
    return answer
```

---

## ✅ Success Checklist

Before moving to production:

- [ ] Papers downloaded successfully (check volume)
- [ ] Bronze table created with papers
- [ ] Silver tables created with chunks and embeddings
- [ ] Vector index created and online
- [ ] Test queries return relevant results
- [ ] Query latency is acceptable (<1s)
- [ ] Understand how to update with new papers
- [ ] Documented your search queries
- [ ] Set up monitoring/alerts
- [ ] Scheduled regular updates (optional)

---

## 🚀 Next Steps

1. **Test Different Queries**: Experiment with various medical topics
2. **Build RAG App**: Create a medical Q&A chatbot
3. **Add More Sources**: Combine with PubMed, arXiv, etc.
4. **Improve Chunks**: Optimize chunk size for your use case
5. **Add Filters**: Filter by publication date, journal, etc.
6. **Monitor Quality**: Track query relevance and accuracy
7. **Schedule Updates**: Run daily to get new papers

---

## 💬 Sample RAG Application

```python
def medical_assistant(patient_question):
    """
    Simple medical literature assistant using RAG
    """
    # 1. Retrieve relevant papers
    papers = search_medical_papers(patient_question, num_results=5)
    
    # 2. Extract text
    context = "\n\n".join([
        row.chunk_text for row in papers.collect()
    ])
    
    # 3. Create prompt
    prompt = f"""Based on the following medical research papers, answer the question.
    
Context:
{context}

Question: {patient_question}

Answer: Based on the research papers, """
    
    # 4. Send to LLM (pseudo-code)
    # answer = your_llm_api(prompt)
    
    return answer

# Example usage
answer = medical_assistant("What are the symptoms of long COVID?")
print(answer)
```

---

**You're all set!** 🎉

Run the notebook and start exploring medical research literature with semantic search!

Questions? Check `TROUBLESHOOTING.md` or reach out to your Databricks support team.

