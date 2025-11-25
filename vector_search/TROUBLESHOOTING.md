# Troubleshooting Guide - MLB Prospects Vector Search Pipeline

This guide helps you diagnose and resolve common issues with the vector search pipeline.

## 🔍 Quick Diagnosis

Run this diagnostic cell to check your environment:

```python
# DIAGNOSTIC CELL - Run this first
import sys
print(f"Python Version: {sys.version}")
print(f"Databricks Runtime: {spark.conf.get('spark.databricks.clusterUsageTags.sparkVersion')}")

# Check Unity Catalog
try:
    catalogs = spark.sql("SHOW CATALOGS").collect()
    print(f"✓ Unity Catalog enabled: {len(catalogs)} catalogs found")
except Exception as e:
    print(f"✗ Unity Catalog issue: {e}")

# Check Vector Search
try:
    from databricks.vector_search.client import VectorSearchClient
    vsc = VectorSearchClient()
    print("✓ Vector Search client initialized")
except Exception as e:
    print(f"✗ Vector Search issue: {e}")

# Check Foundation Models
try:
    import mlflow.deployments
    client = mlflow.deployments.get_deploy_client("databricks")
    print("✓ Foundation Models client initialized")
except Exception as e:
    print(f"✗ Foundation Models issue: {e}")
```

---

## 🚨 Common Issues & Solutions

### Issue 1: `ai_parse` Function Not Found

**Error Message:**
```
AnalysisException: [UNRESOLVED_ROUTINE] Cannot resolve function `ai_parse`
```

**Root Cause:**
- `ai_parse` is a Databricks-specific function
- May not be available in all workspace tiers
- Requires specific Databricks Runtime version

**Solution:**

✅ **Option 1**: Continue without ai_parse (Recommended)
```python
# The notebook automatically falls back to raw content
# No action needed - pipeline will continue normally
```

✅ **Option 2**: Update to supported runtime
```python
# Use Databricks Runtime 14.3 LTS ML or higher
# Check: Cluster → Configuration → Databricks Runtime Version
```

✅ **Option 3**: Manual parsing
```python
# Parse content yourself before bronze layer
from bs4 import BeautifulSoup

def manual_parse(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    # Extract what you need
    return parsed_data
```

**Verification:**
```python
# Check if ai_parse is available
try:
    spark.sql("SELECT ai_parse('test', '{}')").collect()
    print("✓ ai_parse is available")
except:
    print("✗ ai_parse not available (using fallback)")
```

---

### Issue 2: Vector Search Endpoint Not Found

**Error Message:**
```
ResourceNotFound: Vector Search endpoint 'mlb_prospects_endpoint' not found
```

**Root Cause:**
- Endpoint doesn't exist
- Endpoint is in different region
- Insufficient permissions

**Solution:**

✅ **Step 1**: Check if endpoint exists
```python
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# List all endpoints
endpoints = w.vector_search_endpoints.list_endpoints()
for ep in endpoints:
    print(f"Found endpoint: {ep.name} - Status: {ep.endpoint_status}")
```

✅ **Step 2**: Create the endpoint
```python
from databricks.sdk.service.vectorsearch import EndpointType

endpoint = w.vector_search_endpoints.create_endpoint(
    name="mlb_prospects_endpoint",
    endpoint_type=EndpointType.STANDARD
)
print(f"Created endpoint: {endpoint.name}")
```

✅ **Step 3**: Wait for endpoint to be ready
```python
import time

while True:
    status = w.vector_search_endpoints.get_endpoint("mlb_prospects_endpoint")
    print(f"Status: {status.endpoint_status}")
    
    if status.endpoint_status == "ONLINE":
        print("✓ Endpoint is ready!")
        break
    
    print("Waiting 30 seconds...")
    time.sleep(30)
```

**Verification:**
```python
# Verify endpoint is accessible
endpoint = w.vector_search_endpoints.get_endpoint("mlb_prospects_endpoint")
assert endpoint.endpoint_status == "ONLINE"
print("✓ Endpoint verified")
```

---

### Issue 3: Foundation Model API Errors

**Error Message:**
```
HTTPError: 403 Client Error: Forbidden for url: ...
```

**Root Cause:**
- Insufficient permissions to use Foundation Models
- Model endpoint not enabled
- Token/authentication issue

**Solution:**

✅ **Step 1**: Check Foundation Model access
```python
import mlflow.deployments

client = mlflow.deployments.get_deploy_client("databricks")

# List available models
try:
    models = client.list_endpoints()
    print(f"Available models: {[m['name'] for m in models]}")
except Exception as e:
    print(f"Error listing models: {e}")
```

✅ **Step 2**: Use alternative model
```python
# Try a different Foundation Model
EMBEDDING_MODEL = "databricks-gte-large-en"  # Alternative
```

✅ **Step 3**: Use sentence-transformers fallback
```python
# Install fallback library
%pip install sentence-transformers

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(["test text"])
print(f"✓ Fallback model working: {len(embeddings[0])} dimensions")
```

**Verification:**
```python
# Test embedding generation
response = client.predict(
    endpoint="databricks-bge-large-en",
    inputs={"input": ["test"]}
)
print(f"✓ Embedding generated: {len(response['data'][0]['embedding'])} dims")
```

---

### Issue 4: Empty Search Results

**Error Message:**
```
# No error, but search returns 0 results
```

**Root Cause:**
- Index not synced yet
- No data upserted to index
- Query embedding dimension mismatch

**Solution:**

✅ **Step 1**: Check index status
```python
from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()
index = vsc.get_index(
    endpoint_name="mlb_prospects_endpoint",
    index_name=f"{CATALOG}.{SCHEMA}.mlb_prospects_index"
)

# Check index details
details = index.describe()
print(f"Index status: {details['status']}")
print(f"Index type: {details.get('index_type', 'N/A')}")
print(f"Row count: {details.get('num_rows', 0)}")
```

✅ **Step 2**: Verify embeddings table has data
```python
# Check if embeddings exist
embeddings_df = spark.table(f"{CATALOG}.{SCHEMA}.silver_prospects_embeddings")
count = embeddings_df.count()
print(f"Embeddings in table: {count}")

# Check for null embeddings
null_count = embeddings_df.filter("embedding IS NULL").count()
print(f"Null embeddings: {null_count}")

# Sample an embedding
sample = embeddings_df.select("embedding").first()
if sample:
    print(f"Sample embedding length: {len(sample.embedding)}")
```

✅ **Step 3**: Manually upsert data
```python
# Force index update
embeddings_df = spark.table(f"{CATALOG}.{SCHEMA}.silver_prospects_embeddings")
index.upsert(embeddings_df)
print("✓ Data upserted to index")

# Wait a bit for processing
import time
time.sleep(30)
```

✅ **Step 4**: Test with a simple query
```python
# Try a very general query
results = index.similarity_search(
    query_text="baseball",
    columns=["chunk_id", "chunk_text"],
    num_results=1
)
print(f"Results found: {len(results['result']['data_array'])}")
```

**Verification:**
```python
# Comprehensive check
def verify_index():
    details = index.describe()
    
    checks = {
        "Status is ONLINE": details['status'] == 'ONLINE',
        "Has rows": details.get('num_rows', 0) > 0,
        "Primary key set": 'primary_key' in details,
        "Embeddings dimension": details.get('embedding_dimension', 0) > 0
    }
    
    for check, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"{status} {check}")
    
    return all(checks.values())

if verify_index():
    print("\n✓ Index is healthy!")
else:
    print("\n✗ Index has issues - review above")
```

---

### Issue 5: Permission Denied Errors

**Error Message:**
```
PermissionDenied: User does not have CREATE privilege on catalog 'main'
```

**Root Cause:**
- Insufficient Unity Catalog permissions
- Wrong catalog/schema ownership
- Service principal lacks permissions

**Solution:**

✅ **Step 1**: Check current permissions
```python
# Check what you can access
catalogs = spark.sql("SHOW CATALOGS").collect()
print("Accessible catalogs:")
for cat in catalogs:
    print(f"  - {cat.catalog}")

# Check schemas in a catalog
try:
    schemas = spark.sql(f"SHOW SCHEMAS IN {CATALOG}").collect()
    print(f"\nSchemas in {CATALOG}:")
    for schema in schemas:
        print(f"  - {schema.databaseName}")
except Exception as e:
    print(f"Cannot access catalog {CATALOG}: {e}")
```

✅ **Step 2**: Request permissions
```python
# Ask admin to run:
# GRANT CREATE CATALOG ON ACCOUNT TO `your-user@company.com`;
# GRANT USE CATALOG ON CATALOG main TO `your-user@company.com`;
# GRANT CREATE SCHEMA ON CATALOG main TO `your-user@company.com`;
```

✅ **Step 3**: Use existing catalog/schema
```python
# Modify configuration to use a catalog/schema you own
CATALOG = "your_username_catalog"  # Change this
SCHEMA = "mlb_prospects"
```

**Verification:**
```python
# Test create permissions
try:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.test_schema")
    spark.sql(f"DROP SCHEMA {CATALOG}.test_schema")
    print("✓ You have CREATE permissions")
except Exception as e:
    print(f"✗ Missing permissions: {e}")
```

---

### Issue 6: Out of Memory Errors

**Error Message:**
```
OutOfMemoryError: Java heap space
```

**Root Cause:**
- Processing too much data at once
- Insufficient cluster resources
- Memory-intensive operations not optimized

**Solution:**

✅ **Step 1**: Increase cluster size
```python
# Recommended cluster config:
# Driver: 16 GB RAM (e.g., m5.2xlarge)
# Workers: 2-4 nodes with 16+ GB RAM each
```

✅ **Step 2**: Process in smaller batches
```python
# Instead of processing all at once
batch_size = 1000

df = spark.table(SILVER_CLEANED_TABLE)
total_rows = df.count()

for i in range(0, total_rows, batch_size):
    batch_df = df.limit(batch_size).offset(i)
    
    # Process batch
    batch_with_embeddings = batch_df.withColumn(
        "embedding",
        generate_embedding_udf(col("chunk_text"))
    )
    
    # Write incrementally
    batch_with_embeddings.write.mode("append").saveAsTable(SILVER_EMBEDDINGS_TABLE)
    
    print(f"Processed batch {i//batch_size + 1}")
```

✅ **Step 3**: Optimize Spark configuration
```python
# Reduce memory per partition
spark.conf.set("spark.sql.shuffle.partitions", "400")  # More, smaller partitions
spark.conf.set("spark.sql.adaptive.enabled", "true")    # Adaptive query execution
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
```

✅ **Step 4**: Clear caches
```python
# Clear DataFrame caches
spark.catalog.clearCache()

# Unpersist DataFrames
df.unpersist()
```

**Verification:**
```python
# Monitor memory usage
def check_memory():
    # Get Spark UI memory stats
    sc = spark.sparkContext
    status = sc.statusTracker()
    
    executor_info = status.getExecutorInfos()
    for executor in executor_info:
        print(f"Executor {executor.executorId()}")
        print(f"  Memory: {executor.totalMemory() / 1e9:.2f} GB")
```

---

### Issue 7: Slow Query Performance

**Symptom:**
```
# Queries taking >5 seconds
```

**Root Cause:**
- Index not optimized
- Too many results requested
- Cold start (first query after index creation)

**Solution:**

✅ **Step 1**: Warm up the index
```python
# Run a few dummy queries
for i in range(5):
    index.similarity_search(
        query_text="test query",
        num_results=1
    )
print("✓ Index warmed up")
```

✅ **Step 2**: Optimize query parameters
```python
# Request fewer results
results = index.similarity_search(
    query_text="your query",
    num_results=10,  # Don't request 100+ results
    columns=["chunk_id", "chunk_text"]  # Only needed columns
)
```

✅ **Step 3**: Use filters to narrow search
```python
# Pre-filter before vector search
results = index.similarity_search(
    query_text="pitching prospects",
    filters={"chunk_length": {"$gt": 200}},  # Only longer chunks
    num_results=10
)
```

✅ **Step 4**: Cache frequent queries
```python
import functools

@functools.lru_cache(maxsize=100)
def cached_search(query_text, num_results=10):
    return index.similarity_search(
        query_text=query_text,
        num_results=num_results
    )
```

**Verification:**
```python
import time

def benchmark_query(query, iterations=10):
    times = []
    for _ in range(iterations):
        start = time.time()
        index.similarity_search(query_text=query, num_results=10)
        times.append(time.time() - start)
    
    avg_time = sum(times) / len(times)
    print(f"Average query time: {avg_time:.3f}s")
    print(f"Min: {min(times):.3f}s, Max: {max(times):.3f}s")

benchmark_query("top prospects")
```

---

### Issue 8: Data Not Appearing in Volume

**Error Message:**
```
FileNotFoundError: [Errno 2] No such file or directory: '/Volumes/...'
```

**Root Cause:**
- Volume path incorrect
- Volume not mounted
- Permission issue

**Solution:**

✅ **Step 1**: Verify volume exists
```python
# Check volume
try:
    volumes = spark.sql(f"SHOW VOLUMES IN {CATALOG}.{SCHEMA}").collect()
    print("Volumes found:")
    for vol in volumes:
        print(f"  - {vol.volume_name}")
except Exception as e:
    print(f"Error: {e}")
```

✅ **Step 2**: Create volume if missing
```python
spark.sql(f"""
    CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}
""")
print(f"✓ Volume created: {CATALOG}.{SCHEMA}.{VOLUME}")
```

✅ **Step 3**: Test write access
```python
import os

test_file = f"{VOLUME_PATH}/test.txt"
try:
    with open(test_file, 'w') as f:
        f.write("test")
    os.remove(test_file)
    print("✓ Volume is writable")
except Exception as e:
    print(f"✗ Cannot write to volume: {e}")
```

**Verification:**
```python
# List files in volume
import os

if os.path.exists(VOLUME_PATH):
    files = os.listdir(VOLUME_PATH)
    print(f"Files in volume: {len(files)}")
    for f in files[:5]:  # Show first 5
        print(f"  - {f}")
else:
    print(f"✗ Volume path does not exist: {VOLUME_PATH}")
```

---

## 🔧 Advanced Debugging

### Enable Debug Logging

```python
import logging

# Set log level
logging.basicConfig(level=logging.DEBUG)

# Enable Spark logging
spark.sparkContext.setLogLevel("DEBUG")

# Log all HTTP requests
import http.client
http.client.HTTPConnection.debuglevel = 1
```

### Inspect Failed Jobs

```python
# Get recent Spark jobs
sc = spark.sparkContext
status = sc.statusTracker()

job_ids = status.getJobIdsForGroup()
for job_id in job_ids:
    job_info = status.getJobInfo(job_id)
    print(f"Job {job_id}: {job_info.status()}")
```

### Check Delta Table Health

```python
# Check for corrupted files
from delta.tables import DeltaTable

delta_table = DeltaTable.forName(spark, SILVER_EMBEDDINGS_TABLE)

# Show table details
delta_table.detail().show()

# Check for issues
delta_table.vacuum()  # Clean up old files
```

---

## 📞 Getting Help

### Before Asking for Help

Collect this information:

```python
# Diagnostic information
print("="*60)
print("DIAGNOSTIC INFO")
print("="*60)
print(f"Databricks Runtime: {spark.conf.get('spark.databricks.clusterUsageTags.sparkVersion')}")
print(f"Catalog: {CATALOG}")
print(f"Schema: {SCHEMA}")

# Table counts
for table in [BRONZE_TABLE, SILVER_CLEANED_TABLE, SILVER_EMBEDDINGS_TABLE]:
    try:
        count = spark.table(table).count()
        print(f"{table}: {count} rows")
    except:
        print(f"{table}: NOT FOUND")

# Index status
try:
    details = index.describe()
    print(f"Index status: {details['status']}")
    print(f"Index rows: {details.get('num_rows', 0)}")
except:
    print("Index: NOT ACCESSIBLE")

print("="*60)
```

### Support Channels

1. **Check Logs**: Review Spark UI and cluster logs
2. **Databricks Docs**: [docs.databricks.com](https://docs.databricks.com)
3. **Community Forums**: [community.databricks.com](https://community.databricks.com)
4. **Support Ticket**: For enterprise customers

---

## ✅ Health Check Script

Run this comprehensive health check:

```python
def health_check():
    """
    Comprehensive health check for the pipeline
    """
    print("🏥 RUNNING HEALTH CHECK")
    print("="*60)
    
    checks = []
    
    # Check 1: Unity Catalog
    try:
        spark.sql(f"USE CATALOG {CATALOG}")
        checks.append(("Unity Catalog access", True, ""))
    except Exception as e:
        checks.append(("Unity Catalog access", False, str(e)))
    
    # Check 2: Schema exists
    try:
        spark.sql(f"USE SCHEMA {SCHEMA}")
        checks.append(("Schema exists", True, ""))
    except Exception as e:
        checks.append(("Schema exists", False, str(e)))
    
    # Check 3: Volume exists
    try:
        files = os.listdir(VOLUME_PATH)
        checks.append(("Volume accessible", True, f"{len(files)} files"))
    except Exception as e:
        checks.append(("Volume accessible", False, str(e)))
    
    # Check 4: Tables exist
    for table_name, table_full_path in [
        ("Bronze", BRONZE_TABLE),
        ("Silver Cleaned", SILVER_CLEANED_TABLE),
        ("Silver Embeddings", SILVER_EMBEDDINGS_TABLE)
    ]:
        try:
            count = spark.table(table_full_path).count()
            checks.append((f"{table_name} table", True, f"{count} rows"))
        except Exception as e:
            checks.append((f"{table_name} table", False, str(e)))
    
    # Check 5: Vector Search endpoint
    try:
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
        endpoint = w.vector_search_endpoints.get_endpoint(VECTOR_SEARCH_ENDPOINT)
        status = endpoint.endpoint_status
        checks.append(("Vector Search endpoint", status == "ONLINE", status))
    except Exception as e:
        checks.append(("Vector Search endpoint", False, str(e)))
    
    # Check 6: Vector index
    try:
        from databricks.vector_search.client import VectorSearchClient
        vsc = VectorSearchClient()
        index = vsc.get_index(
            endpoint_name=VECTOR_SEARCH_ENDPOINT,
            index_name=INDEX_NAME
        )
        details = index.describe()
        checks.append(("Vector index", details['status'] == 'ONLINE', f"{details.get('num_rows', 0)} vectors"))
    except Exception as e:
        checks.append(("Vector index", False, str(e)))
    
    # Print results
    print("\nRESULTS:")
    print("-"*60)
    passed = 0
    for name, success, detail in checks:
        status = "✓" if success else "✗"
        print(f"{status} {name:30s} {detail}")
        if success:
            passed += 1
    
    print("-"*60)
    print(f"\nPASSED: {passed}/{len(checks)} checks")
    
    if passed == len(checks):
        print("\n🎉 All checks passed! Pipeline is healthy.")
    else:
        print("\n⚠️  Some checks failed. Review errors above.")
    
    return passed == len(checks)

# Run health check
health_check()
```

---

**Last Updated**: November 2025  
**Version**: 1.0

For additional help, consult the [main README](README.md) or [architecture documentation](ARCHITECTURE.md).

