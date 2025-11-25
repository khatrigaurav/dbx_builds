# Quick Start Guide - MLB Prospects Vector Search

Get up and running in 5 minutes! ⚡

## 🎯 Prerequisites Checklist

Before starting, ensure you have:

- [ ] Databricks workspace with Runtime 14.3 LTS or higher
- [ ] Unity Catalog enabled
- [ ] Vector Search feature enabled
- [ ] Permissions to create catalogs, schemas, and volumes
- [ ] Access to Databricks Foundation Models

## 🚀 Step-by-Step Setup

### Step 1: Import the Notebook (1 minute)

1. Download `mlb_prospects_vector_search.py` from this repository
2. In Databricks workspace, navigate to **Workspace** → **Users** → **[your-user]**
3. Click **Import** → Select the `.py` file
4. The notebook will open automatically

### Step 2: Configure Settings (1 minute)

Open the notebook and find the **Configuration Parameters** cell (Cell 2):

```python
# Update these three lines:
CATALOG = "main"  # ← Change to your catalog name
SCHEMA = "mlb_prospects"  # ← Keep or customize
VOLUME = "raw_data"  # ← Keep or customize
```

**Optional**: Import the `config.py` notebook for advanced configuration:
```python
%run ./config
```

### Step 3: Run the Pipeline (2-3 minutes)

Two options:

**Option A - Run All**: Click **Run All** at the top of the notebook

**Option B - Step by Step**: Execute cells sequentially using Shift+Enter

Watch for ✓ checkmarks indicating successful completion of each step.

### Step 4: Verify Success (30 seconds)

Look for these confirmation messages in the final cells:

```
✓ Bronze table created: main.mlb_prospects.bronze_prospects
✓ Silver cleaned table created: main.mlb_prospects.silver_prospects_cleaned
✓ Silver embeddings table created: main.mlb_prospects.silver_prospects_embeddings
✓ Vector Search Index created: main.mlb_prospects.mlb_prospects_index
```

### Step 5: Test Your Index (30 seconds)

Run a test query using the helper function:

```python
# In a new cell:
results = search_prospects("Who are the top pitching prospects?", num_results=5)
display(results)
```

## 🎉 You're Done!

Your vector search pipeline is now live and queryable.

---

## 📊 What Was Created?

After running the notebook, you'll have:

### Unity Catalog Objects

```
your_catalog/
└── mlb_prospects/
    ├── raw_data/                          # Volume
    │   ├── mlb_prospects_TIMESTAMP.html
    │   ├── mlb_prospects_TIMESTAMP.txt
    │   └── mlb_prospects_table_*.html
    ├── bronze_prospects                    # Table
    ├── silver_prospects_cleaned            # Table
    └── silver_prospects_embeddings         # Table
```

### Vector Search Objects

- **Endpoint**: `mlb_prospects_endpoint`
- **Index**: `your_catalog.mlb_prospects.mlb_prospects_index`
- **Type**: Direct Access (no Delta Sync)

---

## 🔍 Common Use Cases

### Use Case 1: Search for Prospects

```python
from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()
index = vsc.get_index(
    endpoint_name="mlb_prospects_endpoint",
    index_name=f"{CATALOG}.{SCHEMA}.mlb_prospects_index"
)

results = index.similarity_search(
    query_text="power hitting prospects with good defense",
    columns=["chunk_text", "file_name"],
    num_results=10
)
```

### Use Case 2: Build a Chatbot

```python
def ask_about_prospects(question):
    """Simple RAG chatbot"""
    # Get relevant context
    results = index.similarity_search(
        query_text=question,
        columns=["chunk_text"],
        num_results=3
    )
    
    context = "\n\n".join([r[0] for r in results['result']['data_array']])
    
    # Send to LLM (pseudo-code)
    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    # answer = call_your_llm(prompt)
    
    return answer

# Usage
answer = ask_about_prospects("What are Chase Burns' strengths?")
```

### Use Case 3: Compare Prospects

```python
def compare_prospects(prospect1, prospect2):
    """Compare two prospects"""
    results1 = search_prospects(f"information about {prospect1}", 5)
    results2 = search_prospects(f"information about {prospect2}", 5)
    
    # Analyze and compare
    return comparison_report
```

---

## 🔄 Updating the Index

When you scrape new data:

```python
# 1. Run cells 1-6 to download and process new data

# 2. Upsert to the vector index
from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()
index = vsc.get_index(
    endpoint_name="mlb_prospects_endpoint",
    index_name=f"{CATALOG}.{SCHEMA}.mlb_prospects_index"
)

# Read updated embeddings
embeddings_df = spark.table(f"{CATALOG}.{SCHEMA}.silver_prospects_embeddings")

# Upsert
index.upsert(embeddings_df)
```

---

## ⚙️ Customization Options

### Change Chunk Size

```python
# In cell 2, modify:
CHUNK_SIZE = 300  # Smaller chunks for shorter responses
CHUNK_OVERLAP = 30
```

### Use Different Embedding Model

```python
# In cell 2:
EMBEDDING_MODEL = "databricks-gte-large-en"  # Alternative model
```

### Add More Data Sources

```python
# In the download cell, add more URLs:
urls = [
    "https://www.fangraphs.com/prospects/the-board/2025-mlb-draft/summary",
    "https://www.fangraphs.com/prospects/the-board/2025-mlb-draft/college",
    "https://www.fangraphs.com/prospects/the-board/2025-mlb-draft/high-school"
]

for url in urls:
    download_fangraphs_data(url, VOLUME_PATH)
```

---

## 🐛 Quick Troubleshooting

| Issue | Solution |
|-------|----------|
| `ai_parse not found` | Normal - pipeline uses fallback. Continue running. |
| `Endpoint not found` | Wait 2-3 minutes for endpoint to provision. |
| `No search results` | Wait 5 minutes for index to sync after creation. |
| `Permission denied` | Verify you have CREATE privileges on the catalog. |
| `Embeddings are null` | Check Foundation Model access in your workspace. |

For detailed troubleshooting, see [README.md](README.md#troubleshooting).

---

## 📞 Need Help?

1. **Check the logs**: Each cell has detailed output messages
2. **Read the README**: [Full documentation](README.md)
3. **Databricks Docs**: [Vector Search Guide](https://docs.databricks.com/en/generative-ai/vector-search.html)
4. **Contact Support**: Your Databricks account team

---

## 🎓 Next Steps

After completing the quick start:

1. **Explore the data**: Query the bronze/silver tables
2. **Tune parameters**: Adjust chunk size and overlap
3. **Build applications**: Create a RAG chatbot or Q&A system
4. **Schedule updates**: Set up a Databricks job for daily refreshes
5. **Monitor performance**: Track query latency and relevance

---

## 💡 Pro Tips

✨ **Tip 1**: Run the notebook on a small single-node cluster to save costs
```
Node Type: i3.xlarge or smaller
Workers: 0 (single node)
Runtime: 14.3 LTS ML
```

✨ **Tip 2**: Cache the endpoint and index objects to avoid repeated lookups
```python
# At the top of your queries
index = vsc.get_index(endpoint_name=ENDPOINT, index_name=INDEX)
# Reuse 'index' for all queries
```

✨ **Tip 3**: Use filters to narrow search scope
```python
results = index.similarity_search(
    query_text="pitchers",
    filters={"position": "RHP"},  # Filter by metadata
    num_results=5
)
```

✨ **Tip 4**: Monitor index status
```python
# Check if index is ready
index_info = index.describe()
print(index_info['status'])  # Should be "ONLINE"
```

---

**Happy Searching! 🚀**

Questions? Check the [full README](README.md) or [configuration guide](config.py).

