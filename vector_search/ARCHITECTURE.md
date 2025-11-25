# Architecture Documentation - MLB Prospects Vector Search Pipeline

## 🏗️ System Architecture Overview

This document provides a detailed technical overview of the vector search pipeline architecture.

## 📐 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA INGESTION LAYER                         │
│                                                                      │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐       │
│  │  FanGraphs   │────▶│   Download   │────▶│Unity Catalog │       │
│  │  Web Source  │     │   Service    │     │   Volume     │       │
│  └──────────────┘     └──────────────┘     └──────────────┘       │
└─────────────────────────────────────────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      BRONZE LAYER (Raw Data)                        │
│                                                                      │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐       │
│  │  Raw HTML    │────▶│   AI Parse   │────▶│    Bronze    │       │
│  │    Files     │     │   Service    │     │    Table     │       │
│  └──────────────┘     └──────────────┘     └──────────────┘       │
└─────────────────────────────────────────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  SILVER LAYER 1 (Cleaned Data)                      │
│                                                                      │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐       │
│  │Text Cleaning │────▶│   Chunking   │────▶│   Silver     │       │
│  │   Service    │     │   Service    │     │   Table 1    │       │
│  └──────────────┘     └──────────────┘     └──────────────┘       │
└─────────────────────────────────────────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                SILVER LAYER 2 (Embeddings Data)                     │
│                                                                      │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐       │
│  │  Foundation  │────▶│  Embedding   │────▶│   Silver     │       │
│  │    Model     │     │  Generation  │     │   Table 2    │       │
│  └──────────────┘     └──────────────┘     └──────────────┘       │
└─────────────────────────────────────────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     VECTOR SEARCH LAYER                             │
│                                                                      │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐       │
│  │   Direct     │────▶│Vector Search │────▶│    Query     │       │
│  │   Access     │     │    Index     │     │   Results    │       │
│  │   Index      │     │   (ANN)      │     │              │       │
│  └──────────────┘     └──────────────┘     └──────────────┘       │
└─────────────────────────────────────────────────────────────────────┘
```

## 🔄 Data Flow Pipeline

### Phase 1: Data Ingestion

```python
Source URL → HTTP Request → BeautifulSoup Parser → Volume Storage
```

**Components:**
- **Source**: FanGraphs web pages (HTML)
- **Parser**: BeautifulSoup4 with html5lib
- **Storage**: Unity Catalog Volume (managed storage)

**Data Format:**
- Raw HTML files: `mlb_prospects_TIMESTAMP.html`
- Cleaned text files: `mlb_prospects_TIMESTAMP.txt`
- Table extracts: `mlb_prospects_table_N_TIMESTAMP.html`

**Error Handling:**
- Retry logic with exponential backoff
- User-Agent headers to prevent blocking
- Timeout configuration (30s default)

### Phase 2: Bronze Layer (Raw Data)

```python
Volume Files → Spark DataFrame → AI Parse → Bronze Delta Table
```

**Schema:**
```sql
CREATE TABLE bronze_prospects (
    file_name STRING,
    file_path STRING,
    raw_content STRING,
    parsed_data STRING,  -- JSON from ai_parse
    ingestion_timestamp TIMESTAMP,
    source_url STRING
)
```

**AI Parse Process:**
1. Load raw content into memory
2. Apply LLM-based extraction (ai_parse function)
3. Extract structured fields:
   - player_name
   - position
   - school
   - ranking
   - description

**Fallback Strategy:**
If `ai_parse` is unavailable, pipeline continues with raw content only.

### Phase 3: Silver Layer 1 (Cleaned & Chunked)

```python
Bronze Table → Clean Text → Chunk Text → Silver Delta Table
```

**Cleaning Operations:**
1. **Whitespace Normalization**: Multiple spaces → single space
2. **Newline Normalization**: Multiple newlines → single newline
3. **Character Filtering**: Remove non-ASCII (optional)
4. **HTML Stripping**: Remove residual HTML tags
5. **Length Filtering**: Remove content < 100 characters

**Chunking Algorithm:**

```python
def chunk_text(text, chunk_size=500, overlap=50):
    """
    Sliding window chunking with smart boundary detection
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence boundary
        if end < len(text):
            for i in range(end, start + chunk_size - 100, -1):
                if text[i] in '.!?\n':
                    end = i + 1
                    break
        
        chunk = text[start:end].strip()
        chunks.append(chunk)
        
        start = end - overlap  # Overlap for context preservation
    
    return chunks
```

**Why Overlap?**
- Prevents loss of context at boundaries
- Improves retrieval quality
- Default: 50 characters (10% of chunk size)

**Schema:**
```sql
CREATE TABLE silver_prospects_cleaned (
    file_name STRING,
    source_url STRING,
    ingestion_timestamp TIMESTAMP,
    cleaned_content STRING,
    chunk_text STRING,
    chunk_id STRING,  -- SHA-256 hash
    chunk_sequence INT,
    chunk_length INT
)
```

### Phase 4: Silver Layer 2 (Embeddings)

```python
Silver Table 1 → Batch Chunks → Foundation Model → Embeddings Table
```

**Embedding Generation Process:**

1. **Batch Creation**: Group chunks into batches of 100
2. **API Call**: Send to Databricks Foundation Model endpoint
3. **Vector Generation**: Receive 1024-dimensional vectors
4. **Storage**: Write to Delta table with embeddings

**Foundation Model Details:**

| Model | Dimension | Use Case | Performance |
|-------|-----------|----------|-------------|
| databricks-bge-large-en | 1024 | General retrieval | High accuracy |
| databricks-gte-large-en | 1024 | Long documents | Good speed |
| all-MiniLM-L6-v2 (fallback) | 384 | Fast retrieval | Lower accuracy |

**Pandas UDF Implementation:**

```python
@pandas_udf("array<double>")
def generate_embedding_udf(texts: pd.Series) -> pd.Series:
    """
    Vectorized embedding generation using Pandas UDF
    """
    batch_size = 100
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size].tolist()
        
        # Call Foundation Model API
        response = deploy_client.predict(
            endpoint=EMBEDDING_MODEL,
            inputs={"input": batch}
        )
        
        embeddings = [item['embedding'] for item in response['data']]
        all_embeddings.extend(embeddings)
    
    return pd.Series(all_embeddings)
```

**Schema:**
```sql
CREATE TABLE silver_prospects_embeddings (
    chunk_id STRING PRIMARY KEY,
    file_name STRING,
    source_url STRING,
    chunk_text STRING,
    chunk_sequence INT,
    chunk_length INT,
    embedding ARRAY<DOUBLE>,  -- 1024 dimensions
    embedding_timestamp TIMESTAMP
)
TBLPROPERTIES (delta.enableChangeDataFeed = true)
```

### Phase 5: Vector Search Index

```python
Embeddings Table → Direct Access Index → ANN Algorithm → Search Results
```

**Index Configuration:**

```python
VectorIndex(
    name=f"{CATALOG}.{SCHEMA}.mlb_prospects_index",
    endpoint_name="mlb_prospects_endpoint",
    primary_key="chunk_id",
    index_type=VectorIndexType.DIRECT_ACCESS,
    embedding_dimension=1024,
    similarity_metric="cosine"
)
```

**Index Type Comparison:**

| Feature | Direct Access | Delta Sync |
|---------|--------------|------------|
| Sync Mode | Manual upsert | Automatic |
| Update Control | Explicit | CDC-based |
| Use Case | Batch updates | Real-time |
| Performance | High (no polling) | Medium |
| Complexity | Low | Medium |

**ANN Algorithm:**
- Uses **HNSW** (Hierarchical Navigable Small World)
- Approximate nearest neighbor search
- Trade-off: Speed vs. Accuracy (configurable)

## 🔍 Query Processing Architecture

### Search Flow

```
Query Text → Embedding Generation → Vector Search → Ranking → Results
```

**Step-by-Step:**

1. **Query Embedding**
   ```python
   query_vector = generate_embedding(query_text)  # 1024-dim vector
   ```

2. **Vector Search**
   ```python
   # ANN search in high-dimensional space
   candidates = index.search(query_vector, k=100)
   ```

3. **Re-ranking** (optional)
   ```python
   # Cross-encoder for precise ranking
   final_results = rerank(candidates, query_text, top_k=10)
   ```

4. **Result Assembly**
   ```python
   return {
       "chunk_text": text,
       "source_url": url,
       "similarity_score": score
   }
   ```

### Similarity Metrics

**Cosine Similarity** (Default)
```
similarity = (A · B) / (||A|| * ||B||)
Range: [-1, 1], Higher is better
```

**Dot Product**
```
similarity = A · B
Range: [-∞, ∞], Higher is better
```

**Euclidean Distance**
```
distance = sqrt(Σ(A[i] - B[i])²)
Range: [0, ∞], Lower is better
```

## 🏛️ Unity Catalog Integration

### Governance Model

```
Catalog (main)
├── Schema (mlb_prospects)
│   ├── Volume (raw_data)
│   │   └── Files (HTML, TXT)
│   ├── Tables
│   │   ├── bronze_prospects
│   │   ├── silver_prospects_cleaned
│   │   └── silver_prospects_embeddings
│   └── Vector Index
│       └── mlb_prospects_index
```

**Benefits:**
- ✅ **Lineage Tracking**: See data flow from source to index
- ✅ **Access Control**: Fine-grained permissions
- ✅ **Audit Logging**: Track all data access
- ✅ **Data Discovery**: Searchable metadata

## ⚡ Performance Optimization

### Spark Optimization

**Partitioning Strategy:**
```python
# Optimal partitions = 2-4x number of cores
spark.conf.set("spark.sql.shuffle.partitions", "200")

# Repartition before expensive operations
df = df.repartition(200, "chunk_id")
```

**Caching Strategy:**
```python
# Cache frequently accessed data
df_bronze.cache()
df_bronze.count()  # Materialize cache
```

**Broadcast Joins:**
```python
# For small lookup tables
broadcast(small_df).join(large_df, "key")
```

### Embedding Generation Optimization

**Batching:**
- Batch size: 100 texts per API call
- Reduces API overhead by 100x
- Uses Pandas UDF for vectorized processing

**Parallelism:**
```python
# Process embeddings in parallel across partitions
df.repartition(200).mapInPandas(generate_embeddings)
```

### Vector Search Optimization

**Index Tuning:**
- **ef_construction**: 128 (higher = better quality, slower build)
- **M**: 16 (HNSW parameter, higher = better recall)

**Query Optimization:**
```python
# Pre-filter before vector search
index.similarity_search(
    query_vector=vector,
    filters={"chunk_length": {"$gt": 200}},
    num_results=10
)
```

## 🔐 Security Architecture

### Access Control

**Catalog Level:**
```sql
GRANT USE CATALOG ON CATALOG main TO `user@company.com`;
GRANT USE SCHEMA ON SCHEMA mlb_prospects TO `user@company.com`;
```

**Table Level:**
```sql
GRANT SELECT ON TABLE bronze_prospects TO `analyst_group`;
GRANT MODIFY ON TABLE silver_prospects_embeddings TO `data_engineer_group`;
```

**Vector Search Level:**
```sql
GRANT USAGE ON VECTOR SEARCH ENDPOINT mlb_prospects_endpoint TO `app_service_principal`;
```

### Data Privacy

- **PII Handling**: No PII in this dataset
- **Data Encryption**: At-rest and in-transit (managed by Databricks)
- **Audit Logs**: All access logged to Unity Catalog

## 📊 Monitoring & Observability

### Key Metrics

**Pipeline Metrics:**
- Documents processed: COUNT(*)
- Average chunk length: AVG(chunk_length)
- Embedding generation time: Timer
- Index sync time: Timer

**Query Metrics:**
- Query latency (p50, p95, p99)
- Results per query
- Average relevance score
- Cache hit rate

**System Metrics:**
- Endpoint health status
- Index size (MB)
- API token usage
- Spark job duration

### Logging Strategy

```python
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Log pipeline events
logger.info(f"Processing {count} documents")
logger.info(f"Generated {count} embeddings in {duration}s")
logger.info(f"Index upsert completed: {count} vectors")
```

## 🔄 Scalability Considerations

### Horizontal Scaling

**Data Volume:**
- Current: ~10K chunks
- Target: 10M+ chunks
- Strategy: Partition by date/source

**Query Load:**
- Current: <10 QPS
- Target: 1000+ QPS
- Strategy: Multiple endpoint instances

### Vertical Scaling

**Embedding Generation:**
- Use GPU-accelerated clusters for local models
- Increase Foundation Model endpoint throughput

**Index Performance:**
- Increase index memory allocation
- Use larger endpoint instance types

## 🧪 Testing Strategy

### Unit Tests
```python
def test_chunk_text():
    text = "A" * 1000
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) == 3  # Expected chunks
    assert len(chunks[0]) <= 500
```

### Integration Tests
```python
def test_end_to_end():
    # Download → Parse → Chunk → Embed → Index
    result = run_pipeline(test_url)
    assert result.success == True
    assert result.chunks_created > 0
```

### Performance Tests
```python
def test_query_latency():
    query = "test query"
    start = time.time()
    results = index.similarity_search(query, num_results=10)
    latency = time.time() - start
    assert latency < 1.0  # <1s latency requirement
```

## 🚀 Deployment Architecture

### Development Environment
- Single-node cluster (i3.xlarge)
- Small dataset (<1000 chunks)
- Foundation Model API (shared)

### Production Environment
- Multi-node cluster (3x i3.2xlarge)
- Full dataset (100K+ chunks)
- Dedicated Foundation Model endpoint
- High-availability vector search endpoint

### CI/CD Pipeline
```yaml
stages:
  - lint: Run pylint on notebooks
  - test: Execute unit tests
  - deploy-dev: Deploy to dev workspace
  - integration-test: Run E2E tests
  - deploy-prod: Deploy to production
```

## 📚 Additional Resources

- [HNSW Algorithm Paper](https://arxiv.org/abs/1603.09320)
- [Databricks Vector Search Internals](https://www.databricks.com/blog/introducing-databricks-vector-search)
- [Medallion Architecture Whitepaper](https://www.databricks.com/glossary/medallion-architecture)

---

**Architecture Version**: 1.0  
**Last Updated**: November 2025  
**Maintained By**: Data Engineering Team

