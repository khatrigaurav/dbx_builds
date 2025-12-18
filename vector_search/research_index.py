# Databricks notebook source
# MAGIC %md
# MAGIC # Medical Research Papers Vector Search Pipeline
# MAGIC
# MAGIC This notebook implements a complete vector search pipeline following the medallion architecture:
# MAGIC 1. **Download**: Fetch pdf records from arxiv
# MAGIC 2. **Bronze Layer**: Parse documents using ai_parse
# MAGIC 3. **Silver Layer 1**: Clean and chunk documents
# MAGIC 4. **Silver Layer 2**: Generate embeddings
# MAGIC 5. **Vector Index**: Create searchable vector index
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration & Setup

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 

# COMMAND ----------

# MAGIC %pip install databricks-vectorsearch
# MAGIC dbutils.library.restartPython()
# MAGIC

# COMMAND ----------

# DBTITLE 1,Configuration Parameters
import json
from datetime import datetime


with open('config.json', 'r') as f:
    config = json.load(f)


CATALOG = config['databricks']['CATALOG']
SCHEMA = config['databricks']['SCHEMA']
VOLUME = config['databricks']['VOLUME']

#table names
BRONZE_TABLE =  config['databricks']["BRONZE_TABLE"]
SILVER_CLEANED_TABLE =  config['databricks']["SILVER_CLEANED_TABLE"]
SILVER_EMBEDDINGS_TABLE  =  config['databricks']["SILVER_EMBEDDINGS_TABLE"]

# Volume path
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"

#Search Parameters
search_query = "covid treatment"
paper_to_download = 10

# Embedding configuration
EMBEDDING_MODEL = config['embedding']['EMBEDDING_MODEL']  # Databricks Foundation Model
CHUNK_SIZE = config['embedding']['CHUNK_SIZE']  # Characters per chunk
CHUNK_OVERLAP = config['embedding']['CHUNK_OVERLAP']  # Overlap between chunks

print(f"Configuration loaded:")
print(f"  Catalog: {CATALOG}")
print(f"  Schema: {SCHEMA}")
print(f"  Volume Path: {VOLUME_PATH}")
print(f"  Bronze Table: {BRONZE_TABLE}")
print(f"  Silver Cleaned Table: {SILVER_CLEANED_TABLE}")
print(f"  Silver Embeddings Table: {SILVER_EMBEDDINGS_TABLE}")
print(f"  Include Sample PDFs: {INCLUDE_SAMPLE_PDFS}")
if INCLUDE_SAMPLE_PDFS:
    print(f"  Number of Samples: {NUM_SAMPLE_PDFS}")

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
import time
import xml.etree.ElementTree as ET
from datetime import datetime


def download_arxiv_papers(search_query, num_papers=10, output_folder="./arxiv_papers"):
    """
    Download research papers from arXiv based on a search query
    
    Args:
        search_query (str): What to search for (e.g., "covid treatment", "machine learning")
        num_papers (int): How many papers to download (default: 10)
        output_folder (str): Where to save the PDFs (default: ./arxiv_papers)
    
    Returns:
        dict: Results with 'success', 'downloaded', 'failed', 'papers' list
    
    Example:
        >>> download_arxiv_papers("covid treatment", num_papers=5)
        >>> download_arxiv_papers("machine learning medical", num_papers=3, output_folder="./papers")
    """
    
    print(f"Searching arXiv for: '{search_query}'")
    print(f"Number of papers: {num_papers}")
    print()
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Search arXiv API
    api_url = "http://export.arxiv.org/api/query"
    params = {
        'search_query': f"all:{search_query}",
        'start': 0,
        'max_results': num_papers,
        'sortBy': 'relevance',
        'sortOrder': 'descending'
    }
    
    try:
        # Get search results
        response = requests.get(api_url, params=params, timeout=30)
        response.raise_for_status()
        
        # Parse XML response
        root = ET.fromstring(response.content)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        entries = root.findall('atom:entry', ns)
        
        if not entries:
            print("No papers found for this query.")
            return {'success': True, 'downloaded': 0, 'failed': 0, 'papers': []}
        
        print(f"Found {len(entries)} paper(s). Downloading...\n")
        
        # Download each paper
        downloaded = []
        failed = []
        
        for i, entry in enumerate(entries, 1):
            try:
                # Get paper info
                title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
                paper_id = entry.find('atom:id', ns).text.split('/abs/')[-1]
                
                # Find PDF link
                pdf_url = None
                for link in entry.findall('atom:link', ns):
                    if link.get('title') == 'pdf':
                        pdf_url = link.get('href')
                        break
                
                if not pdf_url:
                    pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
                
                print(f"[{i}/{len(entries)}] {title[:60]}...")
                
                # Download PDF
                pdf_response = requests.get(pdf_url, timeout=60)
                pdf_response.raise_for_status()
                
                if pdf_response.headers.get('content-type', '').startswith('application/pdf'):
                    # Create filename
                    safe_title = "".join(c for c in title if c.isalnum() or c in ' -_').strip()
                    safe_title = safe_title.replace(' ', '_')[:50]
                    filename = f"arxiv_{paper_id.replace('/', '_')}_{safe_title}.pdf"
                    filepath = os.path.join(output_folder, filename)
                    
                    # Save PDF
                    with open(filepath, 'wb') as f:
                        f.write(pdf_response.content)
                    
                    size_mb = len(pdf_response.content) / (1024 * 1024)
                    print(f"    ✓ Downloaded ({size_mb:.2f} MB)\n")
                    
                    downloaded.append({
                        'title': title,
                        'id': paper_id,
                        'filepath': filepath
                    })
                else:
                    print(f"    ✗ Not a PDF\n")
                    failed.append(paper_id)
                
                # Rate limit: 1 second between downloads
                if i < len(entries):
                    time.sleep(1)
                    
            except Exception as e:
                print(f"    ✗ Error: {e}\n")
                failed.append(paper_id if 'paper_id' in locals() else 'unknown')
                continue
        
        # Summary
        print("="*60)
        print(f"COMPLETE: Downloaded {len(downloaded)} of {len(entries)} papers")
        print(f"Saved to: {output_folder}")
        print("="*60)
        
        return {
            'success': True,
            'downloaded': len(downloaded),
            'failed': len(failed),
            'papers': downloaded,
            'output_folder': output_folder
        }
        
    except Exception as e:
        print(f"Error: {e}")
        return {
            'success': False,
            'downloaded': 0,
            'failed': 0,
            'error': str(e)
        }

result = download_arxiv_papers(search_query, num_papers=paper_to_download, output_folder="/Volumes/gaurav_catalog/medical_research/raw_papers/")

if result.get('success'):
    print(f"\n✓ Setup complete!")
    print(f"  Total PDFs ready for processing: {result.get('downloaded', 0)}")
    if result.get('failed', 0) > 0:
        print(f"  Samples that failed: {result.get('failed', 0)}")
else:
    print(f"\n✗ Setup failed: {result.get('error', 'Unknown error')}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Bronze Layer - AI Parse Documents

# COMMAND ----------

Catalog: raw_papers
  Schema: gaurav_catalog
  Volume Path: /Volumes/raw_papers/gaurav_catalog/medical_research
  Bronze Table: bronze_papers

# COMMAND ----------

os.listdir('/Volumes/gaurav_catalog/medical_research/raw_papers')


# COMMAND ----------

# DBTITLE 1,List Downloaded Files
import os

# List all files in the volume
files = []
for file in os.listdir(VOLUME_PATH):
    if file.endswith(('.pdf', '.json')):
        file_path = os.path.join(VOLUME_PATH, file)
        files.append({
            "filename": file,
            "path": file_path,
            "size": os.path.getsize(file_path)
        })

print(f"Found {len(files)} files in volume:")
pdf_files = [f for f in files if f['filename'].endswith('.pdf')]
print(f"  PDF files: {len(pdf_files)}")
for f in pdf_files:  # Show first 5 PDFs
    print(f"  - {f['filename']} ({round(f['size']/(1024*1024), 2)} MB)")

# COMMAND ----------

# DBTITLE 1,Parse Documents with AI_PARSE

df_bronze = spark.sql(
    """
    WITH parsed AS (
        SELECT
            path,
            ai_parse_document(
                content,
                map('version', '2.0')
                ):document:elements AS elements

        FROM READ_FILES(
            '/Volumes/gaurav_catalog/medical_research/raw_papers/*.pdf',
            format => 'binaryFile'
            )
        )
        SELECT
            path as file_path,
            ai_query(
                'databricks-gpt-5-1',
                concat('parse abstract from the text ', elements)
                ) AS abstract,
            current_timestamp() as ingestion_ts,
            elements as raw_content
        FROM parsed                
                      """)

# # Write to bronze table
df_bronze.write.mode("overwrite").saveAsTable(BRONZE_TABLE)

print(f"✓ Bronze table created: {BRONZE_TABLE}")


# COMMAND ----------

# MAGIC %sql
# MAGIC select * from gaurav_catalog.medical_research.bronze_papers limit 5

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


# Register UDF
from pyspark.sql.types import ArrayType
from pyspark.sql.types import StringType


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

chunk_text_udf = udf(lambda text: chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP), ArrayType(StringType()))


# Read bronze table
df_bronze_read = spark.table(BRONZE_TABLE)

# Clean the content
df_cleaned = df_bronze_read.select(
    col("file_path").alias("file_name"),
    col("ingestion_ts").alias("ingestion_timestamp"),
    # Clean the raw content
    trim(
        regexp_replace(
            regexp_replace(
                regexp_replace(col("abstract"), r'\s+', ' '),  # Multiple spaces to single
                r'\n+', '\n'  # Multiple newlines to single
            ),
            r'[^\x00-\x7F]+', ''  # Remove non-ASCII characters (optional)
        )
    ).alias("cleaned_content")
).filter(length(col("cleaned_content")) > 100)  # Filter out very short content

# Chunk the documents
df_chunked = df_cleaned.select(
    col("file_name"),
    col("ingestion_timestamp"),
    col("cleaned_content"),
    chunk_text_udf(col("cleaned_content")).alias("chunks")
)

# Explode chunks to create one row per chunk
df_silver_cleaned = df_chunked.select(
    col("file_name"),
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

# MAGIC %md
# MAGIC   | Wait till you get the message : Endpoint status: EndpointStatusState.ONLINE
# MAGIC

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

test_query = "What are the promising solutions for COVID Diagnosis?"
results_count = 5

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
        num_results=results_count
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
# MAGIC ## Summary 

# COMMAND ----------

# DBTITLE 1,Pipeline Summary
print("="*80)
print("MEDICAL RESEARCH PAPERS VECTOR SEARCH PIPELINE - SUMMARY")
print("="*80)
print(f"\n✓ Data Source: Europe PMC Open Access")
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


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ### 📝 Next :
# MAGIC - This notebook uses **Direct Access** index type (no Delta Sync as requested)
# MAGIC - Embeddings are generated using Databricks Foundation Models
# MAGIC - The pipeline follows medallion architecture: Bronze → Silver (Clean) → Silver (Embeddings)
# MAGIC - Vector index must be manually updated when data changes (use `index.upsert()`)
# MAGIC - Adjust `CHUNK_SIZE` and `CHUNK_OVERLAP` based on your use case
# MAGIC - Update `CATALOG` and `SCHEMA` to match your environment
# MAGIC
# MAGIC ### 🔬 NEXT STEPS:
# MAGIC
# MAGIC - Query the index: 'What are effective treatments for X?'"
# MAGIC - Build a medical literature assistant or RAG application"
# MAGIC - Monitor index performance in the Databricks UI"
# MAGIC - Schedule this notebook to download new papers daily"
# MAGIC - Add more search queries in ADDITIONAL_QUERIES"
# MAGIC