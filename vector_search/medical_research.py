# Databricks notebook source
# MAGIC %md
# MAGIC # Medical Research Papers Vector Search Pipeline
# MAGIC
# MAGIC This notebook implements a complete vector search pipeline following the medallion architecture:
# MAGIC 1. **Download**: Fetch medical research papers from Europe PMC and store in Unity Catalog Volume
# MAGIC 2. **Bronze Layer**: Parse documents using ai_parse
# MAGIC 3. **Silver Layer 1**: Clean and chunk documents
# MAGIC 4. **Silver Layer 2**: Generate embeddings
# MAGIC 5. **Vector Index**: Create searchable vector index
# MAGIC
# MAGIC Data Source: Europe PMC Open Access - https://www.ebi.ac.uk/europepmc/

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration & Setup

# COMMAND ----------

# DBTITLE 1,Configuration Parameters
import json
from datetime import datetime

# Configuration
CATALOG = "gaurav_catalog"  # Update with your catalog name
SCHEMA = "medical_research"  # Schema name
VOLUME = "raw_papers"  # Volume name for storing downloaded papers

# Table names
BRONZE_TABLE = f"{CATALOG}.{SCHEMA}.bronze_papers"
SILVER_CLEANED_TABLE = f"{CATALOG}.{SCHEMA}.silver_papers_cleaned"
SILVER_EMBEDDINGS_TABLE = f"{CATALOG}.{SCHEMA}.silver_papers_embeddings"

# Volume path
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"

# Europe PMC API Configuration
EUROPEPMC_API_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
SEARCH_QUERY = "long covid AND treatment"  # Modify this to search for different topics
MAX_PAPERS = 100  # Number of papers to download per query

# Additional search queries (optional - can search multiple topics)
ADDITIONAL_QUERIES = [
    # "mRNA vaccine efficacy",
    # "diabetes management",
    # "cancer immunotherapy"
]

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
print(f"  Search Query: '{SEARCH_QUERY}'")
print(f"  Max Papers: {MAX_PAPERS}")

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

# DBTITLE 1,Download Medical Papers from Europe PMC
import requests
import os
from datetime import datetime
import time

def download_medical_papers(query, output_path, max_papers=100):
    """
    Download medical research papers from Europe PMC API and save to volume
    
    Args:
        query: Search query (e.g., "long covid AND treatment")
        output_path: Path to save downloaded papers
        max_papers: Maximum number of papers to download
    
    Returns:
        Dictionary with download results
    """
    try:
        print(f"Downloading papers from Europe PMC...")
        print(f"Search Query: '{query}'")
        print(f"Max Papers: {max_papers}")
        
        # Create directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        # Europe PMC API parameters
        base_url = EUROPEPMC_API_URL
        params = {
            'query': query,
            'resultType': 'core',  # Get full details
            'format': 'json',
            'pageSize': min(max_papers, 1000),  # Max 1000 per request
            'cursorMark': '*'  # For pagination
        }
        
        # Make API request
        print(f"Calling Europe PMC API...")
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract results
        results = data.get('resultList', {}).get('result', [])
        total_found = data.get('hitCount', 0)
        
        print(f"✓ API returned {len(results)} papers (total available: {total_found})")
        
        if len(results) == 0:
            print("⚠️  No papers found for this query. Try a different search term.")
            return {
                "papers_downloaded": 0,
                "query": query,
                "success": False,
                "error": "No results found"
            }
        
        # Process and save each paper
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        papers_saved = []
        
        for idx, paper in enumerate(results[:max_papers], 1):
            # Extract paper details
            paper_id = paper.get('pmcid', paper.get('pmid', f'paper_{idx}'))
            title = paper.get('title', 'No title')
            abstract = paper.get('abstractText', '')
            authors = paper.get('authorString', 'Unknown')
            journal = paper.get('journalTitle', 'Unknown')
            pub_year = paper.get('pubYear', 'Unknown')
            doi = paper.get('doi', '')
            
            # Create formatted text content
            content = f"""PAPER ID: {paper_id}
                            TITLE: {title}
                            AUTHORS: {authors}
                            JOURNAL: {journal}
                            YEAR: {pub_year}
                            DOI: {doi}

                            ABSTRACT:
                            {abstract}

                            ---
                            Full metadata available in JSON file.
"""
            
            # Save as text file
            safe_id = paper_id.replace('/', '_').replace(' ', '_')
            text_filename = f"{output_path}/paper_{safe_id}_{timestamp}.txt"
            
            with open(text_filename, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Save raw JSON metadata
            json_filename = f"{output_path}/paper_{safe_id}_{timestamp}.json"
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(paper, f, indent=2)
            
            papers_saved.append({
                "paper_id": paper_id,
                "title": title[:100] + "..." if len(title) > 100 else title,
                "text_file": text_filename,
                "json_file": json_filename
            })
            
            # Progress indicator
            if idx % 10 == 0:
                print(f"  Processed {idx}/{min(len(results), max_papers)} papers...")
        
        print(f"✓ Successfully saved {len(papers_saved)} papers to volume")
        
        # Create a summary file
        summary_filename = f"{output_path}/download_summary_{timestamp}.json"
        summary = {
            "timestamp": timestamp,
            "query": query,
            "total_available": total_found,
            "papers_downloaded": len(papers_saved),
            "papers": papers_saved
        }
        
        with open(summary_filename, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ Summary saved to: {summary_filename}")
        
        return {
            "query": query,
            "papers_downloaded": len(papers_saved),
            "total_available": total_found,
            "summary_file": summary_filename,
            "success": True
        }
        
    except requests.exceptions.RequestException as e:
        print(f"✗ API Error: {str(e)}")
        return {"success": False, "error": f"API Error: {str(e)}"}
    except Exception as e:
        print(f"✗ Error downloading papers: {str(e)}")
        return {"success": False, "error": str(e)}

# Download papers for primary query
print("="*80)
print("DOWNLOADING MEDICAL RESEARCH PAPERS")
print("="*80)

result = download_medical_papers(SEARCH_QUERY, VOLUME_PATH, max_papers=MAX_PAPERS)
print(f"\n✓ Download complete!")
print(f"  Query: {result.get('query', 'N/A')}")
print(f"  Papers downloaded: {result.get('papers_downloaded', 0)}")
print(f"  Total available: {result.get('total_available', 0)}")

# Optional: Download papers for additional queries
if ADDITIONAL_QUERIES:
    print(f"\nDownloading {len(ADDITIONAL_QUERIES)} additional query topics...")
    for additional_query in ADDITIONAL_QUERIES:
        print(f"\n--- Query: '{additional_query}' ---")
        result_additional = download_medical_papers(additional_query, VOLUME_PATH, max_papers=MAX_PAPERS)
        print(f"✓ Downloaded {result_additional.get('papers_downloaded', 0)} papers")
        time.sleep(1)  # Be nice to the API

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
    
    # Define the schema we want to extract from medical papers
    extraction_schema = """
    {
        "type": "object",
        "properties": {
            "paper_title": {"type": "string", "description": "Full title of the research paper"},
            "research_topic": {"type": "string", "description": "Main research topic or disease studied"},
            "methodology": {"type": "string", "description": "Research methodology or study design"},
            "key_findings": {"type": "string", "description": "Main findings or conclusions"},
            "population": {"type": "string", "description": "Study population or sample size"},
            "intervention": {"type": "string", "description": "Treatment, drug, or intervention studied"},
            "outcome": {"type": "string", "description": "Primary outcome or results"}
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
    lit(EUROPEPMC_API_URL).alias("source_url")
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
from pyspark.sql.functions import pandas_udf, col, current_timestamp
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

VECTOR_SEARCH_ENDPOINT = "medical_research_endpoint"

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

print(f"Endpoint status: {endpoint.endpoint_status.state}")

# COMMAND ----------

# DBTITLE 1,Create Vector Search Index (Direct Access - No Delta Sync)
from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient(disable_notice=True)
INDEX_NAME = f"{CATALOG}.{SCHEMA}.medical_papers_index"

# Get the schema from the embeddings table
embeddings_schema = spark.table(SILVER_EMBEDDINGS_TABLE).schema

# Get embedding dimension from sample
embedding_dim = len(spark.table(SILVER_EMBEDDINGS_TABLE).select("embedding").first()[0])
print(f"Detected embedding dimension: {embedding_dim}")

# Build schema dictionary for the index
schema_dict = {}
for field in embeddings_schema.fields:
    if field.name == "embedding":
        schema_dict[field.name] = "array<double>"
    elif str(field.dataType) == "StringType()":
        schema_dict[field.name] = "string"
    elif str(field.dataType) == "IntegerType()":
        schema_dict[field.name] = "int"
    elif str(field.dataType) == "LongType()":
        schema_dict[field.name] = "long"
    elif str(field.dataType) == "TimestampType()":
        schema_dict[field.name] = "timestamp"
    else:
        schema_dict[field.name] = "string"

print(f"\nSchema for index: {schema_dict}")

# Create Direct Access Vector Index
try:
    index = vsc.create_direct_access_index(
        endpoint_name=VECTOR_SEARCH_ENDPOINT,
        index_name=INDEX_NAME,
        primary_key="chunk_id",
        embedding_dimension=embedding_dim,
        embedding_vector_column="embedding",
        schema=schema_dict
    )
    
    print(f"\n✓ Vector Search Index created: {INDEX_NAME}")
    print(f"  Type: DIRECT_ACCESS")
    print(f"  Primary Key: chunk_id")
    print(f"  Embedding Column: embedding")
    print(f"  Embedding Dimension: {embedding_dim}")
    
except Exception as e:
    if "already exists" in str(e).lower():
        print(f"\nIndex already exists: {INDEX_NAME}")
        print("You can proceed to the next step.")
    else:
        print(f"\nError creating index: {str(e)}")
        import traceback
        traceback.print_exc()

# COMMAND ----------

# DBTITLE 1,Upsert Data into Vector Index
from databricks.vector_search.client import VectorSearchClient
from datetime import datetime

vsc = VectorSearchClient(disable_notice=True)

# Get the index
index = vsc.get_index(
    endpoint_name=VECTOR_SEARCH_ENDPOINT,
    index_name=INDEX_NAME
)

# Read embeddings data
embeddings_df = spark.table(SILVER_EMBEDDINGS_TABLE)

print(f"Total records in table: {embeddings_df.count()}")

# Convert to list of dictionaries and fix timestamp serialization
print("Converting DataFrame to list of dictionaries...")
embeddings_list = []
for row in embeddings_df.collect():
    record = row.asDict()
    # Convert datetime to ISO format string for JSON serialization
    if 'embedding_timestamp' in record and isinstance(record['embedding_timestamp'], datetime):
        record['embedding_timestamp'] = record['embedding_timestamp'].isoformat()
    embeddings_list.append(record)

print(f"Total records to upsert: {len(embeddings_list)}")

# Check sample
if embeddings_list:
    sample = embeddings_list[0]
    print(f"\nSample data:")
    print(f"  Embedding dimension: {len(sample['embedding'])}")
    print(f"  Timestamp type: {type(sample.get('embedding_timestamp'))}")

try:
    # Upsert data to index
    print(f"\nUpserting {len(embeddings_list)} records to index...")
    result = index.upsert(embeddings_list)
    
    print(f"\nUpsert Result:")
    print(f"  Status: {result.get('status')}")
    print(f"  Success count: {result.get('result', {}).get('success_row_count', 0)}")
    
    if result.get('status') == 'FAILURE':
        failed_keys = result.get('result', {}).get('failed_primary_keys', [])
        print(f"  Failed count: {len(failed_keys)}")
        if failed_keys:
            print(f"  First few failed keys: {failed_keys[:3]}")
    else:
        print(f"\n✓ Successfully upserted {result.get('result', {}).get('success_row_count', 0)} records!")
        
except Exception as e:
    print(f"\nError during upsert: {str(e)}")
    import traceback
    traceback.print_exc()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Test Vector Search

# COMMAND ----------

# DBTITLE 1,Query the Vector Index
try:
    from databricks.vector_search.client import VectorSearchClient
    import mlflow.deployments
    
    vsc = VectorSearchClient(disable_notice=True)
    
    # Get the index
    index = vsc.get_index(
        endpoint_name=VECTOR_SEARCH_ENDPOINT,
        index_name=INDEX_NAME
    )
    
    # Test query - relevant to medical research
    test_query = "What are the most effective treatments for long COVID symptoms?"
    
    # For Direct Access indexes, we need to generate embeddings for the query
    print(f"Generating embeddings for query: '{test_query}'")
    deploy_client = mlflow.deployments.get_deploy_client("databricks")
    
    response = deploy_client.predict(
        endpoint=EMBEDDING_MODEL,
        inputs={"input": [test_query]}
    )
    query_vector = response['data'][0]['embedding']
    
    # Search using query_vector instead of query_text
    results = index.similarity_search(
        query_vector=query_vector,
        columns=["chunk_id", "file_name", "chunk_text", "chunk_sequence"],
        num_results=5
    )
    
    print(f"\n✓ Vector search test successful!")
    print(f"Query: '{test_query}'")
    
    # Parse results
    if 'result' in results and 'data_array' in results['result']:
        data_array = results['result']['data_array']
        print(f"\nTop {len(data_array)} results:")
        
        for i, result in enumerate(data_array, 1):
            print(f"\n{i}. Score: {result[-1]:.4f}")
            print(f"   Paper: {result[1]}")
            print(f"   Text: {result[2][:200]}...")
    else:
        print(f"\nNo results found. Result structure: {results}")
    
except Exception as e:
    print(f"Error querying index: {str(e)}")
    import traceback
    traceback.print_exc()
    print("\nNote: For Direct Access indexes, you need to provide query_vector.")
    print("Make sure the embedding model endpoint is available.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary and Next Steps

# COMMAND ----------

# DBTITLE 1,Pipeline Summary
print("="*80)
print("MEDICAL RESEARCH PAPERS VECTOR SEARCH PIPELINE - SUMMARY")
print("="*80)
print(f"\n✓ Data Source: Europe PMC Open Access")
print(f"✓ Search Query: '{SEARCH_QUERY}'")
print(f"✓ Volume Path: {VOLUME_PATH}")
print(f"\n📊 TABLES CREATED:")
print(f"   1. Bronze Layer: {BRONZE_TABLE}")
print(f"      - Raw medical papers from Europe PMC API")
print(f"   2. Silver Layer (Cleaned): {SILVER_CLEANED_TABLE}")
print(f"      - Cleaned and chunked papers")
print(f"   3. Silver Layer (Embeddings): {SILVER_EMBEDDINGS_TABLE}")
print(f"      - Paper chunks with vector embeddings")
print(f"\n🔍 VECTOR SEARCH:")
print(f"   - Endpoint: {VECTOR_SEARCH_ENDPOINT}")
print(f"   - Index: {INDEX_NAME}")
print(f"   - Index Type: DIRECT_ACCESS (No Delta Sync)")
print(f"\n🎯 NEXT STEPS:")
print(f"   1. Query the index: 'What are effective treatments for X?'")
print(f"   2. Build a medical literature assistant or RAG application")
print(f"   3. Monitor index performance in the Databricks UI")
print(f"   4. Schedule this notebook to download new papers daily")
print(f"   5. Add more search queries in ADDITIONAL_QUERIES")
print("="*80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bonus: Helper Functions for Querying

# COMMAND ----------

# DBTITLE 1,Search Helper Function
def search_medical_papers(query_text, num_results=5):
    """
    Helper function to search medical research papers
    
    Args:
        query_text: The search query (e.g., "treatments for long COVID")
        num_results: Number of results to return
        
    Returns:
        DataFrame with search results
    """
    try:
        from databricks.vector_search.client import VectorSearchClient
        import mlflow.deployments
        
        vsc = VectorSearchClient(disable_notice=True)
        index = vsc.get_index(
            endpoint_name=VECTOR_SEARCH_ENDPOINT,
            index_name=INDEX_NAME
        )
        
        # For Direct Access indexes, generate embeddings for the query
        deploy_client = mlflow.deployments.get_deploy_client("databricks")
        response = deploy_client.predict(
            endpoint=EMBEDDING_MODEL,
            inputs={"input": [query_text]}
        )
        query_vector = response['data'][0]['embedding']
        
        # Search using query_vector
        results = index.similarity_search(
            query_vector=query_vector,
            columns=["chunk_id", "file_name", "chunk_text", "source_url"],
            num_results=num_results
        )
        
        # Convert to DataFrame for easier viewing
        if 'result' in results and 'data_array' in results['result']:
            data = results['result']['data_array']
        else:
            print("No results found")
            return None
            
        columns = ["chunk_id", "file_name", "chunk_text", "source_url", "score"]
        
        return spark.createDataFrame(data, columns)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# Example usage:
print("Searching for: 'What are effective treatments for diabetes?'")
results_df = search_medical_papers("What are effective treatments for diabetes?", num_results=10)
if results_df:
    display(results_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ### 📝 Notes:
# MAGIC - This notebook uses **Europe PMC API** to download medical research papers
# MAGIC - Data source: Open Access medical literature (3M+ full-text papers available)
# MAGIC - This notebook uses **Direct Access** index type (no Delta Sync as requested)
# MAGIC - Embeddings are generated using Databricks Foundation Models
# MAGIC - The pipeline follows medallion architecture: Bronze → Silver (Clean) → Silver (Embeddings)
# MAGIC - Vector index must be manually updated when data changes (use `index.upsert()`)
# MAGIC - Adjust `SEARCH_QUERY` to download papers on different medical topics
# MAGIC - Add multiple queries in `ADDITIONAL_QUERIES` to broaden your knowledge base
# MAGIC - Adjust `CHUNK_SIZE` and `CHUNK_OVERLAP` based on your use case
# MAGIC - Update `CATALOG` and `SCHEMA` to match your environment
# MAGIC
# MAGIC ### 🔬 Example Queries for Medical Research:
# MAGIC - "What are the latest treatments for Type 2 diabetes?"
# MAGIC - "Clinical trials for cancer immunotherapy"
# MAGIC - "Side effects of mRNA vaccines"
# MAGIC - "Biomarkers for early Alzheimer's detection"
# MAGIC - "Rehabilitation protocols for stroke patients"
# MAGIC