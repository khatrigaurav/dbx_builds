# Dependencies and Requirements

## System Requirements

### Databricks Environment

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **Databricks Runtime** | 13.3 LTS | 14.3 LTS ML |
| **Python Version** | 3.9+ | 3.10+ |
| **Spark Version** | 3.4+ | 3.5+ |
| **Unity Catalog** | Enabled | Enabled |
| **Vector Search** | Enabled | Enabled |

### Cluster Configuration

**Development/Testing:**
```
Node Type: i3.xlarge (or equivalent)
Workers: 0 (Single Node)
Driver Memory: 16 GB
Driver Cores: 4
Databricks Runtime: 14.3 LTS ML
```

**Production:**
```
Node Type: i3.2xlarge (or equivalent)
Workers: 2-4
Worker Memory: 32 GB per node
Worker Cores: 8 per node
Driver Memory: 32 GB
Driver Cores: 8
Databricks Runtime: 14.3 LTS ML
Autoscaling: Enabled
```

## Python Package Dependencies

### Core Dependencies (Auto-installed in Databricks Runtime)

These packages are included in Databricks Runtime and don't need manual installation:

```
pyspark>=3.4.0
pandas>=1.5.0
numpy>=1.23.0
mlflow>=2.8.0
```

### Additional Dependencies (Install Required)

Install these in the notebook using `%pip install`:

```bash
%pip install beautifulsoup4>=4.11.0
%pip install requests>=2.28.0
%pip install lxml>=4.9.0
%pip install html5lib>=1.1
```

### Optional Dependencies

For enhanced functionality:

```bash
# For local embedding models (fallback)
%pip install sentence-transformers>=2.2.0

# For advanced HTML parsing
%pip install selectolax>=0.3.0

# For better JSON handling
%pip install ujson>=5.7.0

# For progress bars
%pip install tqdm>=4.65.0
```

## Databricks SDK Dependencies

### Python SDK

Already included in Databricks Runtime 13.3+:

```python
from databricks.sdk import WorkspaceClient
from databricks.vector_search.client import VectorSearchClient
import mlflow.deployments
```

Version requirements:
```
databricks-sdk>=0.12.0
databricks-vectorsearch>=0.22
```

## Feature Requirements

### Unity Catalog

**Required Setup:**
1. Unity Catalog enabled in workspace
2. At least one catalog accessible
3. CREATE CATALOG or USE CATALOG privileges
4. CREATE SCHEMA privileges
5. WRITE VOLUME privileges

**Verification:**
```python
# Check Unity Catalog access
spark.sql("SHOW CATALOGS").display()
```

### Vector Search

**Required Setup:**
1. Vector Search feature enabled (contact Databricks if not available)
2. Permissions to create Vector Search endpoints
3. Permissions to create indexes

**Verification:**
```python
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
endpoints = w.vector_search_endpoints.list_endpoints()
print(f"Accessible endpoints: {len(list(endpoints))}")
```

### Foundation Models

**Required Setup:**
1. Foundation Model API access enabled
2. Model serving endpoint permissions
3. Token generation permissions

**Verification:**
```python
import mlflow.deployments
client = mlflow.deployments.get_deploy_client("databricks")
endpoints = client.list_endpoints()
print(f"Available models: {len(endpoints)}")
```

## Network Requirements

### Outbound Access

The pipeline requires outbound internet access for:

1. **Data Source**:
   - `fangraphs.com` (HTTPS port 443)
   
2. **Databricks APIs**:
   - Foundation Model endpoints
   - Vector Search endpoints
   - MLflow endpoints

3. **Package Installation** (if not pre-installed):
   - `pypi.org` (HTTPS port 443)

### Firewall Rules

If behind a corporate firewall, whitelist:
```
*.fangraphs.com
*.databricks.com
pypi.org
files.pythonhosted.org
```

## Storage Requirements

### Volume Storage

**Minimum:** 100 MB
**Recommended:** 1 GB

Breakdown:
- Raw HTML files: ~500 KB per download
- Cleaned text files: ~200 KB per download
- Table extracts: ~100-500 KB each

### Delta Table Storage

| Table | Size (Estimate) | Description |
|-------|----------------|-------------|
| Bronze | ~10 MB | Raw content + metadata |
| Silver (Cleaned) | ~20 MB | Chunked documents |
| Silver (Embeddings) | ~100 MB | Chunks + vectors (1024-dim) |

**Total Estimated Storage:** ~150 MB for initial dataset

For larger datasets (100K+ chunks):
- Embeddings table: ~5-10 GB
- Vector index: ~1-2 GB

### Temporary Storage

Spark shuffle and cache operations may use:
- Development: 1-5 GB
- Production: 10-50 GB

## Compute Requirements

### Memory Requirements

**Per 10K chunks:**
- Spark driver: 8 GB minimum
- Spark executor: 8 GB minimum per executor
- Vector index: 500 MB - 1 GB

**Per 100K chunks:**
- Spark driver: 16 GB minimum
- Spark executor: 16 GB minimum per executor
- Vector index: 5-10 GB

### CPU Requirements

**Embedding Generation:**
- CPU-based: ~10-20 chunks/second per core
- GPU-based (if using local models): ~100-200 chunks/second

**Vector Search:**
- Queries: <100ms per query (after warm-up)
- Indexing: ~1000 vectors/second

## Permission Requirements

### Unity Catalog Permissions

```sql
-- Minimum required permissions
GRANT USE CATALOG ON CATALOG main TO `user@company.com`;
GRANT CREATE SCHEMA ON CATALOG main TO `user@company.com`;
GRANT USE SCHEMA ON SCHEMA mlb_prospects TO `user@company.com`;
GRANT CREATE TABLE ON SCHEMA mlb_prospects TO `user@company.com`;
GRANT SELECT ON SCHEMA mlb_prospects TO `user@company.com`;
GRANT MODIFY ON SCHEMA mlb_prospects TO `user@company.com`;
GRANT READ VOLUME ON SCHEMA mlb_prospects TO `user@company.com`;
GRANT WRITE VOLUME ON SCHEMA mlb_prospects TO `user@company.com`;
```

### Vector Search Permissions

```sql
-- Required for vector search
GRANT CREATE VECTOR SEARCH ENDPOINT TO `user@company.com`;
GRANT CREATE VECTOR SEARCH INDEX TO `user@company.com`;
GRANT QUERY VECTOR SEARCH INDEX TO `user@company.com`;
```

### Service Principal (Production)

For production deployments, use a service principal:

```sql
-- Create service principal permissions
GRANT USE CATALOG ON CATALOG main TO `service-principal-id`;
GRANT ALL PRIVILEGES ON SCHEMA mlb_prospects TO `service-principal-id`;
```

## Version Compatibility Matrix

| Component | Version | Compatible Runtime | Status |
|-----------|---------|-------------------|--------|
| Databricks Runtime | 13.3 LTS | ✓ | Supported |
| Databricks Runtime | 14.3 LTS | ✓ | Recommended |
| Databricks Runtime | 15.0+ | ✓ | Supported |
| Unity Catalog | Any | ✓ | Required |
| Vector Search | 0.22+ | ✓ | Required |
| Foundation Models API | v1 | ✓ | Required |
| Delta Lake | 2.4+ | ✓ | Auto-included |

## Installation Script

Run this in a Databricks notebook to install all dependencies:

```python
# Databricks notebook source
# MAGIC %md
# MAGIC ## Install Dependencies

# COMMAND ----------

# Core dependencies
%pip install beautifulsoup4 requests lxml html5lib

# COMMAND ----------

# Optional: Sentence transformers (fallback embedding model)
# %pip install sentence-transformers torch

# COMMAND ----------

# Restart Python to load packages
dbutils.library.restartPython()

# COMMAND ----------

# Verify installations
import sys
import importlib

packages = [
    "bs4",
    "requests", 
    "lxml",
    "html5lib",
    "pyspark",
    "pandas",
    "numpy",
    "mlflow"
]

print("Package Verification:")
print("="*60)
for package in packages:
    try:
        mod = importlib.import_module(package)
        version = getattr(mod, "__version__", "unknown")
        print(f"✓ {package:20s} {version}")
    except ImportError:
        print(f"✗ {package:20s} NOT FOUND")

# COMMAND ----------

# Verify Databricks-specific packages
from databricks.sdk import WorkspaceClient
from databricks.vector_search.client import VectorSearchClient
import mlflow.deployments

print("\n✓ All Databricks SDK packages available")

# COMMAND ----------

# Check runtime version
runtime = spark.conf.get("spark.databricks.clusterUsageTags.sparkVersion")
python_version = sys.version

print(f"\nRuntime Information:")
print(f"  Databricks Runtime: {runtime}")
print(f"  Python Version: {python_version}")
print(f"  Spark Version: {spark.version}")
```

## Troubleshooting Dependencies

### Issue: Package Import Errors

```python
# Solution 1: Restart Python kernel
dbutils.library.restartPython()

# Solution 2: Reinstall with --force
%pip install --force-reinstall beautifulsoup4

# Solution 3: Clear cache
%pip cache purge
```

### Issue: Version Conflicts

```python
# Check installed versions
%pip list | grep -E "beautifulsoup4|requests|lxml"

# Install specific versions
%pip install beautifulsoup4==4.11.0 requests==2.28.0
```

### Issue: Databricks SDK Not Found

```python
# Upgrade Databricks SDK
%pip install --upgrade databricks-sdk databricks-vectorsearch

# Restart Python
dbutils.library.restartPython()
```

## Production Checklist

Before deploying to production, verify:

- [ ] Databricks Runtime 14.3 LTS ML or higher
- [ ] Unity Catalog enabled and accessible
- [ ] Vector Search feature enabled
- [ ] Foundation Models API access confirmed
- [ ] Appropriate cluster size provisioned
- [ ] All required permissions granted
- [ ] Network access to external sources
- [ ] Sufficient storage allocated
- [ ] Monitoring and logging configured
- [ ] Backup and recovery plan in place

## Additional Resources

- [Databricks Runtime Release Notes](https://docs.databricks.com/release-notes/runtime/index.html)
- [Unity Catalog Setup Guide](https://docs.databricks.com/data-governance/unity-catalog/get-started.html)
- [Vector Search Prerequisites](https://docs.databricks.com/generative-ai/vector-search.html#requirements)
- [Foundation Models Access](https://docs.databricks.com/machine-learning/foundation-models/index.html)

---

**Last Updated**: November 2025  
**Maintained By**: Data Engineering Team

