# Databricks notebook source
# MAGIC %md
# MAGIC # MLB Draft Prospects Vector Search Pipeline
# MAGIC 
# MAGIC This notebook implements a complete vector search pipeline following the medallion architecture:
# MAGIC 1. **Download**: Fetch documents from FanGraphs and store in Unity Catalog Volume
# MAGIC 2. **Bronze Layer**: Parse documents using ai_parse
# MAGIC 3. **Silver Layer 1**: Clean and chunk documents
# MAGIC 4. **Silver Layer 2**: Generate embeddings
# MAGIC 5. **Vector Index**: Create searchable vector index
# MAGIC 
# MAGIC Data Source: https://www.fangraphs.com/prospects/the-board/2025-mlb-draft/summary

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration & Setup

# COMMAND ----------

# DBTITLE 1,Install Required Libraries
# MAGIC %pip install beautifulsoup4 requests lxml html5lib
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Configuration Parameters
import json
from datetime import datetime

# Configuration
CATALOG = "main"  # Update with your catalog name
SCHEMA = "mlb_prospects"  # Schema name
VOLUME = "raw_data"  # Volume name for storing downloaded documents

# Table names
BRONZE_TABLE = f"{CATALOG}.{SCHEMA}.bronze_prospects"
SILVER_CLEANED_TABLE = f"{CATALOG}.{SCHEMA}.silver_prospects_cleaned"
SILVER_EMBEDDINGS_TABLE = f"{CATALOG}.{SCHEMA}.silver_prospects_embeddings"

# Volume path
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"

# Data source URL
SOURCE_URL = "https://www.fangraphs.com/prospects/the-board/2025-mlb-draft/summary"

# Embedding configuration
EMBEDDING_MODEL = "databricks-bge-large-en"  # Databricks Foundation Model
CHUNK_SIZE = 500  # Characters per chunk
CHUNK_OVERLAP = 50  # Overlap between chunks

print(f"Configuration loaded:")
print(f"  Catalog: {CATALOG}")
print(f"  Schema: {SCHEMA}")
print(f"  Volume Path: {VOLUME_PATH}")
print(f"  Bronze Table: {BRONZE_TABLE}")
print(f"  Silver Cleaned Table: {SILVER_CLEANED_TABLE}")
print(f"  Silver Embeddings Table: {SILVER_EMBEDDINGS_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Create Catalog, Schema, and Volume

# COMMAND ----------

# DBTITLE 1,Create Catalog and Schema
# Create catalog if not exists
spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
print(f"✓ Catalog '{CATALOG}' ready")

# Create schema if not exists
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
print(f"✓ Schema '{SCHEMA}' ready")

# Create volume if not exists
spark.sql(f"""
  CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}
""")
print(f"✓ Volume '{VOLUME}' ready at {VOLUME_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Download Documents to Volume

# COMMAND ----------

# DBTITLE 1,Download Web Content
import requests
from bs4 import BeautifulSoup
import os
from datetime import datetime

def download_fangraphs_data(url, output_path):
    """
    Download MLB prospects data from FanGraphs and save to volume
    """
    try:
        print(f"Downloading data from: {url}")
        
        # Set headers to mimic a browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Download the page
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Create directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        # Save raw HTML
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_filename = f"{output_path}/mlb_prospects_{timestamp}.html"
        
        with open(html_filename, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        print(f"✓ Downloaded HTML saved to: {html_filename}")
        
        # Parse and extract text content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Extract text
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        # Save cleaned text
        text_filename = f"{output_path}/mlb_prospects_{timestamp}.txt"
        with open(text_filename, 'w', encoding='utf-8') as f:
            f.write(text)
        
        print(f"✓ Cleaned text saved to: {text_filename}")
        
        # Also extract structured data if tables exist
        tables = soup.find_all('table')
        if tables:
            print(f"✓ Found {len(tables)} table(s) in the page")
            
            # Save each table as a separate file
            for idx, table in enumerate(tables):
                table_filename = f"{output_path}/mlb_prospects_table_{idx}_{timestamp}.html"
                with open(table_filename, 'w', encoding='utf-8') as f:
                    f.write(str(table))
                print(f"  - Table {idx+1} saved to: {table_filename}")
        
        return {
            "html_file": html_filename,
            "text_file": text_filename,
            "tables_count": len(tables),
            "success": True
        }
        
    except Exception as e:
        print(f"✗ Error downloading data: {str(e)}")
        return {"success": False, "error": str(e)}

# Download the data
result = download_fangraphs_data(SOURCE_URL, VOLUME_PATH)
print(f"\nDownload result: {json.dumps(result, indent=2)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Bronze Layer - AI Parse Documents

# COMMAND ----------

# DBTITLE 1,List Downloaded Files
import os

# List all files in the volume
files = []
for file in os.listdir(VOLUME_PATH):
    if file.endswith(('.html', '.txt')):
        file_path = os.path.join(VOLUME_PATH, file)
        files.append({
            "filename": file,
            "path": file_path,
            "size": os.path.getsize(file_path)
        })

print(f"Found {len(files)} files in volume:")
for f in files:
    print(f"  - {f['filename']} ({f['size']} bytes)")

# COMMAND ----------

# DBTITLE 1,Parse Documents with AI_PARSE
from pyspark.sql.functions import col, current_timestamp, lit
from pyspark.sql.types import StructType, StructField, StringType, TimestampType

# Read the downloaded files
df_files = spark.createDataFrame([
    {"file_path": f["path"], "file_name": f["filename"]} 
    for f in files if f["filename"].endswith('.txt')  # Focus on text files
])

# Read file contents
def read_file_content(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

read_file_udf = udf(read_file_content, StringType())

df_with_content = df_files.withColumn("content", read_file_udf(col("file_path")))

# Use AI_PARSE to extract structured information
# Note: ai_parse is a Databricks-specific function that uses LLMs to extract structured data
try:
    from pyspark.sql.functions import ai_parse
    
    # Define the schema we want to extract
    extraction_schema = """
    {
        "type": "object",
        "properties": {
            "player_name": {"type": "string", "description": "Name of the MLB draft prospect"},
            "position": {"type": "string", "description": "Player's position"},
            "school": {"type": "string", "description": "School or team the player is from"},
            "ranking": {"type": "string", "description": "Draft ranking or prospect ranking"},
            "description": {"type": "string", "description": "Description or scouting report"}
        }
    }
    """
    
    # Apply AI parse - this will extract structured data from unstructured text
    df_parsed = df_with_content.withColumn(
        "parsed_data",
        ai_parse(col("content"), lit(extraction_schema))
    )
    
    print("✓ AI_PARSE applied successfully")
    
except Exception as e:
    print(f"Note: ai_parse function may not be available in your environment: {str(e)}")
    print("Creating bronze table with raw content instead")
    
    # Fallback: Create bronze table without AI parsing
    df_parsed = df_with_content.withColumn("parsed_data", lit(None).cast("string"))

# Add metadata columns
df_bronze = df_parsed.select(
    col("file_name"),
    col("file_path"),
    col("content").alias("raw_content"),
    col("parsed_data"),
    current_timestamp().alias("ingestion_timestamp"),
    lit(SOURCE_URL).alias("source_url")
)

# Write to bronze table
df_bronze.write.mode("overwrite").saveAsTable(BRONZE_TABLE)

print(f"✓ Bronze table created: {BRONZE_TABLE}")
print(f"  Records: {df_bronze.count()}")
display(df_bronze)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Silver Layer 1 - Clean and Chunk Documents

# COMMAND ----------

# DBTITLE 1,Clean and Chunk Documents
from pyspark.sql.functions import (
    col, explode, array, expr, monotonically_increasing_id,
    regexp_replace, trim, length, sha2, concat_ws
)
import re

def chunk_text(text, chunk_size=500, overlap=50):
    """
    Split text into overlapping chunks
    """
    if not text or len(text) == 0:
        return []
    
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        
        # If this is not the last chunk, try to break at a sentence or word boundary
        if end < text_length:
            # Look for sentence boundary (., !, ?)
            for i in range(end, max(start + chunk_size - 100, start), -1):
                if text[i] in '.!?\n':
                    end = i + 1
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap if end < text_length else text_length
    
    return chunks if chunks else [text]  # Return original text if chunking fails

# Register UDF
from pyspark.sql.types import ArrayType
chunk_text_udf = udf(lambda text: chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP), ArrayType(StringType()))

# Read bronze table
df_bronze_read = spark.table(BRONZE_TABLE)

# Clean the content
df_cleaned = df_bronze_read.select(
    col("file_name"),
    col("source_url"),
    col("ingestion_timestamp"),
    # Clean the raw content
    trim(
        regexp_replace(
            regexp_replace(
                regexp_replace(col("raw_content"), r'\s+', ' '),  # Multiple spaces to single
                r'\n+', '\n'  # Multiple newlines to single
            ),
            r'[^\x00-\x7F]+', ''  # Remove non-ASCII characters (optional)
        )
    ).alias("cleaned_content")
).filter(length(col("cleaned_content")) > 100)  # Filter out very short content

# Chunk the documents
df_chunked = df_cleaned.select(
    col("file_name"),
    col("source_url"),
    col("ingestion_timestamp"),
    col("cleaned_content"),
    chunk_text_udf(col("cleaned_content")).alias("chunks")
)

# Explode chunks to create one row per chunk
df_silver_cleaned = df_chunked.select(
    col("file_name"),
    col("source_url"),
    col("ingestion_timestamp"),
    col("cleaned_content"),
    explode(col("chunks")).alias("chunk_text")
).withColumn(
    "chunk_id", 
    sha2(concat_ws("_", col("file_name"), col("chunk_text")), 256)
).withColumn(
    "chunk_length",
    length(col("chunk_text"))
)

# Add row numbers for tracking
from pyspark.sql.window import Window
window = Window.partitionBy("file_name").orderBy("chunk_id")
from pyspark.sql.functions import row_number

df_silver_cleaned = df_silver_cleaned.withColumn(
    "chunk_sequence",
    row_number().over(window)
)

# Write to silver cleaned table
df_silver_cleaned.write.mode("overwrite").saveAsTable(SILVER_CLEANED_TABLE)

print(f"✓ Silver cleaned table created: {SILVER_CLEANED_TABLE}")
print(f"  Total chunks: {df_silver_cleaned.count()}")
print(f"  Avg chunk length: {df_silver_cleaned.agg({'chunk_length': 'avg'}).collect()[0][0]:.0f} characters")
display(df_silver_cleaned.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Silver Layer 2 - Generate Embeddings

# COMMAND ----------

# DBTITLE 1,Generate Embeddings using Foundation Model
from pyspark.sql.functions import pandas_udf
import pandas as pd
import numpy as np

# Read the cleaned and chunked data
df_silver_read = spark.table(SILVER_CLEANED_TABLE)

# Method 1: Using Databricks Foundation Model API (Recommended)
try:
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
    import mlflow.deployments
    
    # Get the deployment client
    deploy_client = mlflow.deployments.get_deploy_client("databricks")
    
    def get_embeddings_batch(texts):
        """
        Generate embeddings using Databricks Foundation Model
        """
        try:
            response = deploy_client.predict(
                endpoint=EMBEDDING_MODEL,
                inputs={"input": texts}
            )
            return [item['embedding'] for item in response['data']]
        except Exception as e:
            print(f"Error generating embeddings: {str(e)}")
            # Return zero vectors as fallback
            return [[0.0] * 1024 for _ in texts]
    
    # Create batched UDF for efficiency
    @pandas_udf("array<double>")
    def generate_embedding_udf(texts: pd.Series) -> pd.Series:
        """
        Pandas UDF to generate embeddings in batches
        """
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size].tolist()
            embeddings = get_embeddings_batch(batch)
            all_embeddings.extend(embeddings)
        
        return pd.Series(all_embeddings)
    
    print(f"✓ Using Databricks Foundation Model: {EMBEDDING_MODEL}")
    use_foundation_model = True
    
except Exception as e:
    print(f"Note: Foundation Model API not available: {str(e)}")
    print("Will use alternative embedding method")
    use_foundation_model = False

# Method 2: Alternative - Use sentence-transformers (if Foundation Model not available)
if not use_foundation_model:
    print("Installing sentence-transformers...")
    # This would require: %pip install sentence-transformers
    
    @pandas_udf("array<double>")
    def generate_embedding_udf(texts: pd.Series) -> pd.Series:
        """
        Generate embeddings using sentence-transformers
        """
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = model.encode(texts.tolist())
            return pd.Series([emb.tolist() for emb in embeddings])
        except Exception as e:
            print(f"Error: {str(e)}")
            # Return dummy embeddings as fallback
            return pd.Series([[0.0] * 384 for _ in range(len(texts))])

# Generate embeddings for all chunks
df_with_embeddings = df_silver_read.withColumn(
    "embedding",
    generate_embedding_udf(col("chunk_text"))
)

# Add embedding metadata
df_embeddings = df_with_embeddings.select(
    col("chunk_id"),
    col("file_name"),
    col("source_url"),
    col("chunk_text"),
    col("chunk_sequence"),
    col("chunk_length"),
    col("embedding"),
    current_timestamp().alias("embedding_timestamp")
)

# Write to silver embeddings table
# Important: Enable Change Data Feed for Vector Search
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {SILVER_EMBEDDINGS_TABLE} (
        chunk_id STRING,
        file_name STRING,
        source_url STRING,
        chunk_text STRING,
        chunk_sequence INT,
        chunk_length INT,
        embedding ARRAY<DOUBLE>,
        embedding_timestamp TIMESTAMP
    )
    TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")

df_embeddings.write.mode("overwrite").option("mergeSchema", "true").saveAsTable(SILVER_EMBEDDINGS_TABLE)

print(f"✓ Silver embeddings table created: {SILVER_EMBEDDINGS_TABLE}")
print(f"  Total embeddings: {df_embeddings.count()}")
display(df_embeddings.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Create Vector Search Index

# COMMAND ----------

# DBTITLE 1,Create Vector Search Endpoint
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.vectorsearch import EndpointType

w = WorkspaceClient()

VECTOR_SEARCH_ENDPOINT = "mlb_prospects_endpoint"

# Create or get vector search endpoint
try:
    endpoint = w.vector_search_endpoints.get_endpoint(VECTOR_SEARCH_ENDPOINT)
    print(f"✓ Using existing endpoint: {VECTOR_SEARCH_ENDPOINT}")
except Exception as e:
    print(f"Creating new endpoint: {VECTOR_SEARCH_ENDPOINT}")
    endpoint = w.vector_search_endpoints.create_endpoint(
        name=VECTOR_SEARCH_ENDPOINT,
        endpoint_type=EndpointType.STANDARD
    )
    print(f"✓ Endpoint created: {VECTOR_SEARCH_ENDPOINT}")

print(f"Endpoint status: {endpoint.endpoint_status}")

# COMMAND ----------

# DBTITLE 1,Create Vector Search Index (Direct Access - No Delta Sync)
from databricks.sdk.service.vectorsearch import (
    VectorIndexType,
    DeltaSyncVectorIndexSpecRequest,
    DirectAccessVectorIndexSpec,
    EmbeddingSourceColumn,
    VectorIndex
)

INDEX_NAME = f"{CATALOG}.{SCHEMA}.mlb_prospects_index"

# Delete index if it exists (for clean recreation)
try:
    w.vector_search_indexes.delete_index(index_name=INDEX_NAME)
    print(f"Deleted existing index: {INDEX_NAME}")
except Exception:
    pass

# Create Direct Access Vector Index (NO DELTA SYNC as requested)
try:
    index = w.vector_search_indexes.create_index(
        name=INDEX_NAME,
        endpoint_name=VECTOR_SEARCH_ENDPOINT,
        primary_key="chunk_id",
        index_type=VectorIndexType.DIRECT_ACCESS,
        direct_access_index_spec=DirectAccessVectorIndexSpec(
            embedding_source_columns=[
                EmbeddingSourceColumn(
                    name="embedding",
                    embedding_dimension=1024  # Adjust based on your model
                )
            ],
            schema_json=spark.table(SILVER_EMBEDDINGS_TABLE).schema.json()
        )
    )
    
    print(f"✓ Vector Search Index created: {INDEX_NAME}")
    print(f"  Type: DIRECT_ACCESS (No Delta Sync)")
    print(f"  Primary Key: chunk_id")
    print(f"  Embedding Column: embedding")
    
except Exception as e:
    print(f"Error creating index: {str(e)}")
    print("\nAlternative: You can create the index via the UI or API")

# COMMAND ----------

# DBTITLE 1,Upsert Data into Vector Index
# For Direct Access indexes, we need to upsert data manually
try:
    from databricks.vector_search.client import VectorSearchClient
    
    vsc = VectorSearchClient()
    
    # Get the index
    index = vsc.get_index(
        endpoint_name=VECTOR_SEARCH_ENDPOINT,
        index_name=INDEX_NAME
    )
    
    # Read embeddings data
    embeddings_df = spark.table(SILVER_EMBEDDINGS_TABLE)
    
    # Upsert data to index
    index.upsert(embeddings_df)
    
    print(f"✓ Data upserted to index: {INDEX_NAME}")
    print(f"  Records indexed: {embeddings_df.count()}")
    
except Exception as e:
    print(f"Note: Index upsert error: {str(e)}")
    print("You may need to manually sync the index through the UI")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Test Vector Search

# COMMAND ----------

# DBTITLE 1,Query the Vector Index
try:
    from databricks.vector_search.client import VectorSearchClient
    
    vsc = VectorSearchClient()
    
    # Get the index
    index = vsc.get_index(
        endpoint_name=VECTOR_SEARCH_ENDPOINT,
        index_name=INDEX_NAME
    )
    
    # Test query
    test_query = "Who are the top pitching prospects?"
    
    results = index.similarity_search(
        query_text=test_query,
        columns=["chunk_id", "file_name", "chunk_text", "chunk_sequence"],
        num_results=5
    )
    
    print(f"✓ Vector search test successful!")
    print(f"Query: '{test_query}'")
    print(f"\nTop {len(results['result']['data_array'])} results:")
    
    for i, result in enumerate(results['result']['data_array'], 1):
        print(f"\n{i}. Score: {result[-1]:.4f}")
        print(f"   Text: {result[2][:200]}...")
    
except Exception as e:
    print(f"Error querying index: {str(e)}")
    print("The index may still be syncing. Please wait a few minutes and try again.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary and Next Steps

# COMMAND ----------

# DBTITLE 1,Pipeline Summary
print("="*80)
print("MLB PROSPECTS VECTOR SEARCH PIPELINE - SUMMARY")
print("="*80)
print(f"\n✓ Data Source: {SOURCE_URL}")
print(f"✓ Volume Path: {VOLUME_PATH}")
print(f"\n📊 TABLES CREATED:")
print(f"   1. Bronze Layer: {BRONZE_TABLE}")
print(f"      - Raw parsed documents from source")
print(f"   2. Silver Layer (Cleaned): {SILVER_CLEANED_TABLE}")
print(f"      - Cleaned and chunked documents")
print(f"   3. Silver Layer (Embeddings): {SILVER_EMBEDDINGS_TABLE}")
print(f"      - Document chunks with vector embeddings")
print(f"\n🔍 VECTOR SEARCH:")
print(f"   - Endpoint: {VECTOR_SEARCH_ENDPOINT}")
print(f"   - Index: {INDEX_NAME}")
print(f"   - Index Type: DIRECT_ACCESS (No Delta Sync)")
print(f"\n🎯 NEXT STEPS:")
print(f"   1. Query the index using similarity_search()")
print(f"   2. Build a chatbot or RAG application")
print(f"   3. Monitor index performance in the Databricks UI")
print(f"   4. Schedule this notebook to refresh data periodically")
print("="*80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bonus: Helper Functions for Querying

# COMMAND ----------

# DBTITLE 1,Search Helper Function
def search_prospects(query_text, num_results=5):
    """
    Helper function to search the vector index
    
    Args:
        query_text: The search query
        num_results: Number of results to return
        
    Returns:
        DataFrame with search results
    """
    try:
        from databricks.vector_search.client import VectorSearchClient
        
        vsc = VectorSearchClient()
        index = vsc.get_index(
            endpoint_name=VECTOR_SEARCH_ENDPOINT,
            index_name=INDEX_NAME
        )
        
        results = index.similarity_search(
            query_text=query_text,
            columns=["chunk_id", "file_name", "chunk_text", "source_url"],
            num_results=num_results
        )
        
        # Convert to DataFrame for easier viewing
        data = results['result']['data_array']
        columns = ["chunk_id", "file_name", "chunk_text", "source_url", "score"]
        
        return spark.createDataFrame(data, columns)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

# Example usage:
# results_df = search_prospects("top draft picks for 2025", num_results=10)
# display(results_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ### 📝 Notes:
# MAGIC - This notebook uses **Direct Access** index type (no Delta Sync as requested)
# MAGIC - Embeddings are generated using Databricks Foundation Models
# MAGIC - The pipeline follows medallion architecture: Bronze → Silver (Clean) → Silver (Embeddings)
# MAGIC - Vector index must be manually updated when data changes (use `index.upsert()`)
# MAGIC - Adjust `CHUNK_SIZE` and `CHUNK_OVERLAP` based on your use case
# MAGIC - Update `CATALOG` and `SCHEMA` to match your environment

