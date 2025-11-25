# MLB Draft Prospects Vector Search Pipeline

A complete end-to-end vector search pipeline built for Databricks that downloads, processes, and indexes MLB draft prospect data from FanGraphs.

## 📋 Overview

This notebook implements a production-ready vector search pipeline following Databricks medallion architecture:

```
Source URL → Volume → Bronze → Silver (Clean) → Silver (Embeddings) → Vector Index
```

### Pipeline Stages

1. **Download**: Fetches HTML/text data from FanGraphs and stores in Unity Catalog Volume
2. **Bronze Layer**: Uses `ai_parse()` to extract structured information from raw documents
3. **Silver Layer 1**: Cleans and chunks documents into searchable segments
4. **Silver Layer 2**: Generates vector embeddings using Databricks Foundation Models
5. **Vector Index**: Creates a Direct Access vector search index (no Delta Sync)

## 🎯 Features

- ✅ **No Delta Sync**: Uses Direct Access index as requested
- ✅ **Medallion Architecture**: Clean separation of Bronze/Silver layers
- ✅ **Intelligent Chunking**: Splits documents at sentence boundaries with overlap
- ✅ **Foundation Model Embeddings**: Uses `databricks-bge-large-en` for high-quality embeddings
- ✅ **Full Pipeline Automation**: From download to searchable index
- ✅ **Error Handling**: Graceful fallbacks for API limitations
- ✅ **Helper Functions**: Easy-to-use search utilities included

## 🚀 Getting Started

### Prerequisites

1. Databricks workspace with Unity Catalog enabled
2. Access to Vector Search endpoints
3. Access to Databricks Foundation Models
4. Permissions to create catalogs, schemas, and volumes

### Configuration

Before running the notebook, update these parameters in **Step 1**:

```python
CATALOG = "main"  # Your catalog name
SCHEMA = "mlb_prospects"  # Your schema name
VOLUME = "raw_data"  # Volume name for raw data

EMBEDDING_MODEL = "databricks-bge-large-en"
CHUNK_SIZE = 500  # Characters per chunk
CHUNK_OVERLAP = 50  # Overlap between chunks
```

### Running the Pipeline

1. **Import the Notebook**
   - Upload `mlb_prospects_vector_search.py` to your Databricks workspace
   - Open it as a notebook

2. **Run All Cells**
   - Execute cells sequentially or run all at once
   - The pipeline will automatically create all required resources

3. **Monitor Progress**
   - Check the output of each cell for status messages
   - Look for ✓ symbols indicating successful completion

## 📊 Data Flow

### Bronze Table
```sql
bronze_prospects
├── file_name
├── file_path
├── raw_content
├── parsed_data (from ai_parse)
├── ingestion_timestamp
└── source_url
```

### Silver Table (Cleaned)
```sql
silver_prospects_cleaned
├── file_name
├── source_url
├── ingestion_timestamp
├── cleaned_content
├── chunk_text
├── chunk_id (SHA-256 hash)
├── chunk_length
└── chunk_sequence
```

### Silver Table (Embeddings)
```sql
silver_prospects_embeddings
├── chunk_id (PRIMARY KEY)
├── file_name
├── source_url
├── chunk_text
├── chunk_sequence
├── chunk_length
├── embedding (ARRAY<DOUBLE>)
└── embedding_timestamp
```

## 🔍 Using the Vector Index

### Basic Search

```python
from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()
index = vsc.get_index(
    endpoint_name="mlb_prospects_endpoint",
    index_name="main.mlb_prospects.mlb_prospects_index"
)

results = index.similarity_search(
    query_text="Who are the top pitching prospects for 2025?",
    columns=["chunk_text", "file_name", "source_url"],
    num_results=5
)
```

### Using Helper Function

The notebook includes a built-in helper function:

```python
# Search for prospects
results_df = search_prospects("best power hitters in the draft", num_results=10)
display(results_df)
```

### Updating the Index

Since this uses Direct Access (no Delta Sync), manually update when data changes:

```python
# After updating the embeddings table
embeddings_df = spark.table("main.mlb_prospects.silver_prospects_embeddings")
index.upsert(embeddings_df)
```

## 🏗️ Architecture Decisions

### Why Direct Access Index?

As requested, this pipeline uses **Direct Access** instead of Delta Sync:

- ✅ Manual control over when index updates occur
- ✅ No automatic background syncing
- ✅ Explicit upsert operations for data updates
- ✅ Better for batch-style updates

### Chunking Strategy

Documents are split into ~500 character chunks with 50 character overlap:

- **Benefits**: Better semantic coherence, manageable embedding sizes
- **Overlap**: Ensures context isn't lost at chunk boundaries
- **Smart Splitting**: Breaks at sentence boundaries when possible

### Embedding Model

Uses `databricks-bge-large-en` (BGE = BAAI General Embedding):

- **Dimension**: 1024
- **Quality**: State-of-the-art retrieval performance
- **Speed**: Optimized for Databricks infrastructure
- **Fallback**: Includes sentence-transformers alternative if Foundation Model unavailable

## 📈 Performance Considerations

### Optimization Tips

1. **Batch Processing**: The notebook uses Pandas UDFs for efficient embedding generation
2. **Chunk Size**: Adjust `CHUNK_SIZE` based on your content:
   - Shorter (250-400): Better for FAQ-style content
   - Longer (500-1000): Better for narrative documents
3. **Index Updates**: For large updates, use `upsert()` in batches

### Monitoring

Check these metrics in the Databricks UI:

- Vector Search endpoint status
- Index sync status (should be "Ready")
- Query latency
- Number of indexed vectors

## 🛠️ Troubleshooting

### Common Issues

**Issue**: `ai_parse` function not found
```python
# The notebook includes a fallback that stores raw content
# You can still complete the pipeline without ai_parse
```

**Issue**: Foundation Model API not available
```python
# The notebook falls back to sentence-transformers
# Install: %pip install sentence-transformers
```

**Issue**: Index creation fails
```python
# Check that your endpoint exists and is running
# Verify you have permissions to create indexes
# Ensure the embeddings table has data
```

**Issue**: Search returns no results
```python
# Wait a few minutes for index to sync
# Verify data was upserted: index.describe()
# Check that embeddings are not null/zero vectors
```

## 🔄 Scheduling and Automation

### Daily Updates

Create a Databricks Job to run this notebook daily:

```bash
# In Databricks Jobs UI:
1. Create new job
2. Add notebook task → select mlb_prospects_vector_search
3. Schedule: Daily at 2 AM
4. Cluster: Use a small cluster (single node is fine)
```

### Incremental Updates

To avoid reprocessing all data:

```python
# Modify the download step to check for new data
# Use watermark timestamps
# Only process new files since last run
```

## 📝 Customization Guide

### Different Data Source

Replace the download logic in **Step 2**:

```python
def download_your_data(url, output_path):
    # Your custom download logic
    pass
```

### Custom Chunking Logic

Modify the `chunk_text()` function in **Step 4**:

```python
def chunk_text(text, chunk_size=500, overlap=50):
    # Your custom chunking strategy
    pass
```

### Different Embedding Model

Change the embedding model in **Step 5**:

```python
EMBEDDING_MODEL = "your-preferred-model"
# Adjust embedding_dimension in index creation accordingly
```

## 📚 Additional Resources

- [Databricks Vector Search Documentation](https://docs.databricks.com/en/generative-ai/vector-search.html)
- [Foundation Models on Databricks](https://docs.databricks.com/en/machine-learning/foundation-models/index.html)
- [Unity Catalog Volumes](https://docs.databricks.com/en/connect/unity-catalog/volumes.html)
- [Medallion Architecture](https://www.databricks.com/glossary/medallion-architecture)

## 🤝 Contributing

To extend this pipeline:

1. **Add new data sources**: Modify the download function
2. **Improve parsing**: Enhance the ai_parse schema
3. **Better chunking**: Implement semantic chunking
4. **Hybrid search**: Add BM25 alongside vector search
5. **Re-ranking**: Add a cross-encoder for result re-ranking

## 📄 License

This pipeline is provided as-is for educational and production use.

## 🐛 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review Databricks documentation
3. Contact your Databricks support team

---

**Last Updated**: November 2025  
**Databricks Runtime**: 14.3 LTS or higher recommended  
**Python Version**: 3.10+

