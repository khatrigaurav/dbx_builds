# Databricks Vector Search Projects

This repository contains production-ready Databricks projects for building vector search and RAG (Retrieval Augmented Generation) applications.

## 📁 Projects

### 🔍 Vector Search - MLB Prospects Pipeline

**Location**: `vector_search/`

A complete end-to-end vector search pipeline that demonstrates:
- Web scraping and data ingestion into Unity Catalog Volumes
- Medallion architecture (Bronze → Silver → Gold)
- AI-powered document parsing
- Intelligent text chunking
- Embedding generation using Databricks Foundation Models
- Direct Access vector search index (no Delta Sync)

**Data Source**: FanGraphs MLB Draft Prospects 2025

[➡️ View Full Documentation](vector_search/README.md)

#### Quick Start

1. Import `vector_search/mlb_prospects_vector_search.py` into Databricks
2. Update configuration in the notebook or use `vector_search/config.py`
3. Run all cells to create the complete pipeline
4. Query the vector index for semantic search

```python
# Example query
results = search_prospects("Who are the top pitching prospects?", num_results=5)
```

## 🛠️ Prerequisites

- Databricks Runtime 14.3 LTS or higher
- Unity Catalog enabled
- Vector Search enabled
- Access to Databricks Foundation Models

## 📚 Architecture Patterns

All projects follow Databricks best practices:

- ✅ **Medallion Architecture**: Bronze → Silver → Gold layers
- ✅ **Unity Catalog**: Full governance and lineage
- ✅ **Delta Lake**: ACID transactions and time travel
- ✅ **Vector Search**: Native integration with Databricks
- ✅ **Foundation Models**: State-of-the-art embeddings

## 🚀 Getting Started

1. Clone this repository
2. Choose a project folder
3. Import the notebooks into your Databricks workspace
4. Update configuration parameters
5. Run the pipeline

## 📖 Additional Resources

- [Databricks Vector Search Documentation](https://docs.databricks.com/en/generative-ai/vector-search.html)
- [Unity Catalog Guide](https://docs.databricks.com/en/data-governance/unity-catalog/index.html)
- [Foundation Models on Databricks](https://docs.databricks.com/en/machine-learning/foundation-models/index.html)
- [Medallion Architecture](https://www.databricks.com/glossary/medallion-architecture)

## 🤝 Contributing

Feel free to extend these projects with:
- New data sources
- Alternative embedding models
- Enhanced chunking strategies
- Hybrid search implementations
- Production monitoring dashboards

## 📄 License

MIT License - Use freely in your projects

---

**Last Updated**: November 2025