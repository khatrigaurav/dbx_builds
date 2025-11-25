# Databricks notebook source
# MAGIC %md
# MAGIC # Configuration File for MLB Prospects Vector Search Pipeline
# MAGIC 
# MAGIC Modify these parameters to customize the pipeline for your environment

# COMMAND ----------

# DBTITLE 1,Unity Catalog Configuration
"""
Unity Catalog Configuration
Update these to match your Databricks environment
"""

CATALOG = "main"  # Replace with your catalog name
SCHEMA = "mlb_prospects"  # Schema for all tables
VOLUME = "raw_data"  # Volume name for storing downloaded files

# COMMAND ----------

# DBTITLE 1,Data Source Configuration
"""
Data Source Configuration
"""

SOURCE_URL = "https://www.fangraphs.com/prospects/the-board/2025-mlb-draft/summary"
SOURCE_NAME = "fangraphs_mlb_draft"

# Additional URLs to scrape (optional)
ADDITIONAL_URLS = [
    # Add more URLs here if you want to scrape multiple pages
    # "https://www.fangraphs.com/prospects/the-board/2025-mlb-draft/..."
]

# COMMAND ----------

# DBTITLE 1,Table Names
"""
Table Name Configuration
"""

BRONZE_TABLE = f"{CATALOG}.{SCHEMA}.bronze_prospects"
SILVER_CLEANED_TABLE = f"{CATALOG}.{SCHEMA}.silver_prospects_cleaned"
SILVER_EMBEDDINGS_TABLE = f"{CATALOG}.{SCHEMA}.silver_prospects_embeddings"

# COMMAND ----------

# DBTITLE 1,Embedding Configuration
"""
Embedding Model and Chunking Parameters
"""

# Embedding model to use
EMBEDDING_MODEL = "databricks-bge-large-en"  # Options: databricks-bge-large-en, databricks-gte-large-en
EMBEDDING_DIMENSION = 1024  # Dimension for databricks-bge-large-en

# Alternative models (if foundation model not available):
# - all-MiniLM-L6-v2 (384 dimensions)
# - all-mpnet-base-v2 (768 dimensions)
FALLBACK_MODEL = "all-MiniLM-L6-v2"
FALLBACK_DIMENSION = 384

# Chunking parameters
CHUNK_SIZE = 500  # Characters per chunk
CHUNK_OVERLAP = 50  # Overlap between consecutive chunks
MIN_CHUNK_LENGTH = 100  # Minimum characters to keep a chunk

# COMMAND ----------

# DBTITLE 1,Vector Search Configuration
"""
Vector Search Index Configuration
"""

VECTOR_SEARCH_ENDPOINT = "mlb_prospects_endpoint"
INDEX_NAME = f"{CATALOG}.{SCHEMA}.mlb_prospects_index"
INDEX_TYPE = "DIRECT_ACCESS"  # Options: DIRECT_ACCESS, DELTA_SYNC

# Vector search parameters
SIMILARITY_METRIC = "cosine"  # Options: cosine, dot_product, euclidean
NUM_SEARCH_RESULTS = 10  # Default number of results to return

# COMMAND ----------

# DBTITLE 1,AI Parse Configuration
"""
AI Parse Schema Configuration
Define what structured data to extract from documents
"""

AI_PARSE_SCHEMA = """{
    "type": "object",
    "properties": {
        "player_name": {
            "type": "string",
            "description": "Full name of the MLB draft prospect"
        },
        "position": {
            "type": "string",
            "description": "Primary playing position (e.g., SS, RHP, OF)"
        },
        "school": {
            "type": "string",
            "description": "High school, college, or organization the player is from"
        },
        "ranking": {
            "type": "string",
            "description": "Draft ranking or prospect ranking number"
        },
        "grade": {
            "type": "string",
            "description": "Prospect grade or rating"
        },
        "description": {
            "type": "string",
            "description": "Scouting report, player description, or summary"
        },
        "strengths": {
            "type": "string",
            "description": "Key strengths and skills"
        },
        "weaknesses": {
            "type": "string",
            "description": "Areas for improvement"
        },
        "eta": {
            "type": "string",
            "description": "Estimated time of arrival to MLB (ETA)"
        }
    },
    "required": ["player_name"]
}"""

# COMMAND ----------

# DBTITLE 1,Data Quality Configuration
"""
Data Quality and Cleaning Parameters
"""

# Text cleaning options
REMOVE_NON_ASCII = False  # Set to True to remove non-ASCII characters
NORMALIZE_WHITESPACE = True  # Normalize multiple spaces/newlines
REMOVE_HTML_TAGS = True  # Strip any remaining HTML tags

# Filtering options
MIN_CONTENT_LENGTH = 100  # Minimum content length to keep
MAX_CONTENT_LENGTH = 1000000  # Maximum content length (1MB)

# COMMAND ----------

# DBTITLE 1,Performance Configuration
"""
Performance and Processing Parameters
"""

# Batch sizes
EMBEDDING_BATCH_SIZE = 100  # Number of texts to process per batch
UPSERT_BATCH_SIZE = 1000  # Number of records to upsert at once

# Parallelism
SPARK_SHUFFLE_PARTITIONS = 200  # Adjust based on cluster size

# Timeouts
DOWNLOAD_TIMEOUT = 30  # Seconds to wait for HTTP requests
API_TIMEOUT = 60  # Seconds to wait for API calls

# COMMAND ----------

# DBTITLE 1,Logging and Monitoring
"""
Logging Configuration
"""

LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR
ENABLE_DETAILED_LOGGING = True

# Monitoring
TRACK_METRICS = True  # Track performance metrics
METRICS_TABLE = f"{CATALOG}.{SCHEMA}.pipeline_metrics"

# COMMAND ----------

# DBTITLE 1,Advanced Options
"""
Advanced Configuration Options
"""

# Feature flags
USE_AI_PARSE = True  # Set to False to skip ai_parse step
USE_FOUNDATION_MODEL = True  # Set to False to use fallback embeddings
ENABLE_INCREMENTAL_LOAD = False  # Set to True for incremental processing

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 5  # Seconds between retries

# Cache configuration
CACHE_EMBEDDINGS = True  # Cache embeddings to avoid recomputation

# COMMAND ----------

# DBTITLE 1,Validation
"""
Validate Configuration
"""

def validate_config():
    """
    Validate configuration parameters
    """
    errors = []
    
    # Check required fields
    if not CATALOG:
        errors.append("CATALOG must be specified")
    if not SCHEMA:
        errors.append("SCHEMA must be specified")
    if not SOURCE_URL:
        errors.append("SOURCE_URL must be specified")
    
    # Check numeric ranges
    if CHUNK_SIZE < 50:
        errors.append("CHUNK_SIZE must be at least 50")
    if CHUNK_OVERLAP >= CHUNK_SIZE:
        errors.append("CHUNK_OVERLAP must be less than CHUNK_SIZE")
    if EMBEDDING_BATCH_SIZE < 1:
        errors.append("EMBEDDING_BATCH_SIZE must be positive")
    
    # Check embedding dimensions
    valid_dimensions = [384, 768, 1024, 1536]
    if EMBEDDING_DIMENSION not in valid_dimensions:
        print(f"Warning: Unusual embedding dimension {EMBEDDING_DIMENSION}. Common values are {valid_dimensions}")
    
    if errors:
        print("❌ Configuration Errors:")
        for error in errors:
            print(f"  - {error}")
        raise ValueError("Invalid configuration")
    else:
        print("✓ Configuration validated successfully")

# Run validation
validate_config()

# COMMAND ----------

# DBTITLE 1,Display Configuration Summary
"""
Display Configuration Summary
"""

def print_config_summary():
    """
    Print a summary of the current configuration
    """
    print("="*80)
    print("CONFIGURATION SUMMARY")
    print("="*80)
    print("\n📁 Unity Catalog:")
    print(f"   Catalog: {CATALOG}")
    print(f"   Schema: {SCHEMA}")
    print(f"   Volume: {VOLUME}")
    print(f"\n📊 Tables:")
    print(f"   Bronze: {BRONZE_TABLE}")
    print(f"   Silver (Cleaned): {SILVER_CLEANED_TABLE}")
    print(f"   Silver (Embeddings): {SILVER_EMBEDDINGS_TABLE}")
    print(f"\n🔍 Vector Search:")
    print(f"   Endpoint: {VECTOR_SEARCH_ENDPOINT}")
    print(f"   Index: {INDEX_NAME}")
    print(f"   Type: {INDEX_TYPE}")
    print(f"\n🤖 Embeddings:")
    print(f"   Model: {EMBEDDING_MODEL}")
    print(f"   Dimension: {EMBEDDING_DIMENSION}")
    print(f"   Chunk Size: {CHUNK_SIZE}")
    print(f"   Chunk Overlap: {CHUNK_OVERLAP}")
    print(f"\n🌐 Data Source:")
    print(f"   URL: {SOURCE_URL}")
    print(f"   Additional URLs: {len(ADDITIONAL_URLS)}")
    print("="*80)

print_config_summary()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Usage
# MAGIC 
# MAGIC To use this configuration in the main pipeline notebook:
# MAGIC 
# MAGIC ```python
# MAGIC # Run this config notebook first
# MAGIC %run ./config
# MAGIC 
# MAGIC # Then use the variables in your pipeline
# MAGIC print(f"Using catalog: {CATALOG}")
# MAGIC ```

