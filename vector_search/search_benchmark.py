# Databricks notebook source
# MAGIC %md
# MAGIC # Search Methods Benchmark: ANN vs Hybrid vs Full-Text
# MAGIC
# MAGIC This notebook benchmarks three search approaches for medical research papers:
# MAGIC
# MAGIC 1. **ANN (Approximate Nearest Neighbor)** - Pure vector similarity search
# MAGIC 2. **Hybrid Search** - Combines vector search with keyword matching (BM25)
# MAGIC 3. **Full-Text Search** - Traditional keyword/SQL-based search
# MAGIC
# MAGIC **Prerequisites**: Run `medical_research.py` first to create the embeddings table.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup & Configuration

# COMMAND ----------

# MAGIC %pip install databricks-vectorsearch --quiet
# MAGIC dbutils.library.restartPython()
# MAGIC

# COMMAND ----------

# DBTITLE 1,Configuration
import time
import json
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

with open('config.json', 'r') as f:
    config = json.load(f)


# Configuration - Update to match your medical_research.py setup
CATALOG = config['databricks']['CATALOG']
SCHEMA = config['databricks']['SCHEMA']

#table names
BRONZE_TABLE =  config['databricks']["BRONZE_TABLE"]
SILVER_CLEANED_TABLE =  config['databricks']["SILVER_CLEANED_TABLE"]
SILVER_EMBEDDINGS_TABLE  =  config['databricks']["SILVER_EMBEDDINGS_TABLE"]

# Vector Search
VECTOR_SEARCH_ENDPOINT = config['vector_search']["VECTOR_SEARCH_ENDPOINT"]
INDEX_NAME = config['vector_search']["INDEX_NAME"]
INDEX_NAME = f"{CATALOG}.{SCHEMA}.{INDEX_NAME}"

# Embedding Model (Databricks Foundation Model)
EMBEDDING_MODEL = config['embedding']['EMBEDDING_MODEL']  # Databricks Foundation Model

# Benchmark configuration
NUM_TEST_QUERIES = 10
RESULTS_PER_QUERY = 10
BENCHMARK_ITERATIONS = 3  # Run each query multiple times for accurate timing

print(f"Configuration loaded:")
print(f"  Embeddings Table: {SILVER_EMBEDDINGS_TABLE}")
print(f"  Vector Index: {INDEX_NAME}")
print(f"  Embedding Model: {EMBEDDING_MODEL}")
print(f"  Test Queries: {NUM_TEST_QUERIES}")
print(f"  Results per Query: {RESULTS_PER_QUERY}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Data Availability

# COMMAND ----------

# DBTITLE 1,Check Tables and Index
from pyspark.sql.functions import count, col

# Check embeddings table
try:
    df_embeddings = spark.table(SILVER_EMBEDDINGS_TABLE)
    total_chunks = df_embeddings.count()
    print(f"✓ Embeddings table found: {total_chunks} chunks")
    
    # Sample a few records
    print(f"\nSample records:")
    display(df_embeddings.select("file_name", "chunk_text", "chunk_length").limit(3))
    
except Exception as e:
    print(f"✗ Error accessing embeddings table: {e}")
    print("Make sure you've run medical_research.py first!")

# Check vector index
try:
    from databricks.vector_search.client import VectorSearchClient
    
    vsc = VectorSearchClient()
    index = vsc.get_index(
        endpoint_name=VECTOR_SEARCH_ENDPOINT,
        index_name=INDEX_NAME
    )
    
    index_info = index.describe()
    print(f"\n✓ Vector index found")
    print(f"  Status: {index_info.get('status', 'Unknown')}")
    print(f"  Indexed rows: {index_info.get('num_rows', 0)}")
    
except Exception as e:
    print(f"\n✗ Error accessing vector index: {e}")
    print("Vector search will not be available for benchmarking")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Test Queries

# COMMAND ----------

# DBTITLE 1,Medical Research Test Queries
# Define test queries with expected relevance
TEST_QUERIES = [
    {
        "query": "What are the most effective treatments for acute COVID-19 infection?",
        "keywords": ["treatment", "acute covid", "therapy", "antiviral"],
        "category": "Treatment"
    },
    {
        "query": "Evidence for long COVID management strategies and symptom relief",
        "keywords": ["long covid", "post-acute", "management", "symptom"],
        "category": "Long COVID"
    },
    {
        "query": "Risk factors for severe COVID-19 and hospitalization",
        "keywords": ["risk factor", "severe", "hospitalization", "covid"],
        "category": "Risk Factors"
    },
    {
        "query": "Vaccine effectiveness against COVID-19 infection and severe outcomes",
        "keywords": ["vaccine", "effectiveness", "covid", "severe"],
        "category": "Vaccines"
    },
    {
        "query": "Adverse events and safety profile of COVID-19 vaccines",
        "keywords": ["covid vaccine", "adverse", "side effects", "safety"],
        "category": "Safety"
    },
    {
        "query": "Impact of prior infection and hybrid immunity on COVID-19 outcomes",
        "keywords": ["hybrid immunity", "prior infection", "covid", "protection"],
        "category": "Immunity"
    },
    {
        "query": "Transmission dynamics of SARS-CoV-2 in household and community settings",
        "keywords": ["transmission", "sars-cov-2", "household", "community"],
        "category": "Transmission"
    },
    {
        "query": "Clinical trial results for novel antiviral drugs targeting COVID-19",
        "keywords": ["clinical trial", "antiviral", "covid", "results"],
        "category": "Clinical Trials"
    },
    {
        "query": "Biomarkers predicting progression from mild to severe COVID-19",
        "keywords": ["biomarker", "severity", "progression", "covid"],
        "category": "Diagnosis"
    },
    {
        "query": "Effect of mask-wearing and ventilation on reducing COVID-19 spread",
        "keywords": ["mask", "ventilation", "prevention", "covid"],
        "category": "Prevention"
    }
]


print(f"Defined {len(TEST_QUERIES)} test queries:")
for i, q in enumerate(TEST_QUERIES, 1):
    print(f"{i}. [{q['category']}] {q['query'][:60]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Method 1: ANN Search (Vector Similarity)

# COMMAND ----------

# DBTITLE 1,Implement ANN Search
from databricks.vector_search.client import VectorSearchClient
import time

def search_ann(query_text, num_results=10):
    """
    Pure vector similarity search using Databricks Vector Search
    
    Args:
        query_text: Natural language query
        num_results: Number of results to return
        
    Returns:
        Dictionary with results and timing
    """
    try:
        import mlflow.deployments
        
        vsc = VectorSearchClient(disable_notice=True)
        index = vsc.get_index(
            endpoint_name=VECTOR_SEARCH_ENDPOINT,
            index_name=INDEX_NAME
        )
        
        # Time the search
        start_time = time.time()
        
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
            columns=["chunk_id", "file_name", "chunk_text"],
            num_results=num_results,
            disable_notice=True
        )
        
        latency = time.time() - start_time
        
        # Extract results
        if 'result' in results and 'data_array' in results['result']:
            data = results['result']['data_array']
        else:
            data = []
        
        return {
            "method": "ANN",
            "success": True,
            "latency_ms": latency * 1000,
            "num_results": len(data),
            "results": [
                {
                    "chunk_id": row[0],
                    "file_name": row[1],
                    "text": row[2][:200] + "..." if len(row[2]) > 200 else row[2],
                    "score": row[3]
                }
                for row in data
            ]
        }
        
    except Exception as e:
        import traceback
        return {
            "method": "ANN",
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "latency_ms": 0,
            "num_results": 0,
            "results": []
        }

# Test ANN search
print("Testing ANN Search...")
test_result = search_ann("What are treatments for covid?", num_results=5)

if test_result["success"]:
    print(f"✓ ANN Search working")
    print(f"  Latency: {test_result['latency_ms']:.2f}ms")
    print(f"  Results: {test_result['num_results']}")
    print(f"\nTop result:")
    if test_result["results"]:
        print(f"  Score: {test_result['results'][0]['score']:.4f}")
        print(f"  Text: {test_result['results'][0]['text']}")
else:
    print(f"✗ ANN Search failed: {test_result.get('error', 'Unknown error')}")
    if 'traceback' in test_result:
        print(f"\nTraceback:\n{test_result['traceback']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Method 2: Full-Text Search (SQL/BM25)

# COMMAND ----------

# DBTITLE 1,Implement Full-Text Search
from pyspark.sql.functions import col, lower, lit, length, when, sum as spark_sum, expr

def search_fulltext(query_text, num_results=10):
    """
    Keyword-based search using SQL LIKE and scoring
    
    Args:
        query_text: Query string
        num_results: Number of results to return
        
    Returns:
        Dictionary with results and timing
    """
    try:
        import mlflow.deployments
        
        start_time = time.time()
        ##################
        
        
        vsc = VectorSearchClient(disable_notice=True)
        index = vsc.get_index(
            endpoint_name=VECTOR_SEARCH_ENDPOINT,
            index_name=INDEX_NAME
        )
        
        # Time the search
        start_time = time.time()
        
        # For Direct Access indexes, generate embeddings for the query
        
        
        # Search using query_vector
        results = index.similarity_search(
            query_text=query_text,
            columns=["chunk_id", "file_name", "chunk_text"],
            num_results=num_results,
            disable_notice=True,
            query_type = 'FULL_TEXT'
        )

        
        # Extract results
        latency = time.time() - start_time
        
        if 'result' in results and 'data_array' in results['result']:
            data = results['result']['data_array']
        else:
            data = []
        
        data = results['result']['data_array'] 

        max_hybrid_score = max(x[-1] for x in data)
        min_hybrid_score = min(x[-1] for x in data)
        score_range = max_hybrid_score - min_hybrid_score if max_hybrid_score != min_hybrid_score else 1.0

        return {
            "method": "Full Text Search",
            "success": True,
            "latency_ms": latency * 1000,
            "num_results": len(data),
            "results": [
                {
                    "chunk_id": row[0],
                    "file_name": row[1],
                    "text": row[2][:200] + "..." if len(row[2]) > 200 else row[2],
                    "score": (row[-1]  - min_hybrid_score) / score_range

                }
                for row in data
            ]
        }
        
    except Exception as e:
        return {
            "method": "Full-Text",
            "success": False,
            "error": str(e),
            "latency_ms": 0,
            "num_results": 0,
            "results": []
        }

# Test full-text search
print("Testing Full-Text Search...")
test_result = search_fulltext("What are treatments for covid?", num_results=5)

if test_result["success"]:
    print(f"✓ Full-Text Search working")
    print(f"  Latency: {test_result['latency_ms']:.2f}ms")
    print(f"  Results: {test_result['num_results']}")
    print(f"\nTop result:")
    if test_result["results"]:
        print(f"  Score: {test_result['results'][0]['score']:.3f} ")
        print(f"  Text: {test_result['results'][0]['text']}")
else:
    print(f"✗ Full-Text Search failed: {test_result.get('error', 'Unknown error')}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Method 3: Hybrid Search (Vector + Keyword)

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
import time

def search_hybrid(query_text, num_results=10):
    """
    Pure vector similarity search using Databricks Vector Search
    
    Args:
        query_text: Natural language query
        num_results: Number of results to return
        
    Returns:
        Dictionary with results and timing
    """
    try:
        import mlflow.deployments
        
        vsc = VectorSearchClient(disable_notice=True)
        index = vsc.get_index(
            endpoint_name=VECTOR_SEARCH_ENDPOINT,
            index_name=INDEX_NAME
        )
        
        # Time the search
        start_time = time.time()
        
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
            query_text=query_text,
            columns=["chunk_id", "file_name", "chunk_text"],
            num_results=num_results,
            disable_notice=True,
            query_type="hybrid"
        )
        
        latency = time.time() - start_time
        
        # Extract results
        if 'result' in results and 'data_array' in results['result']:
            data = results['result']['data_array']
        else:
            data = []
        
        return {
            "method": "Hybrid",
            "success": True,
            "latency_ms": latency * 1000,
            "num_results": len(data),
            "results": [
                {
                    "chunk_id": row[0],
                    "file_name": row[1],
                    "text": row[2][:200] + "..." if len(row[2]) > 200 else row[2],
                    "score": row[3]
                }
                for row in data
            ]
        }
        
    except Exception as e:
        import traceback
        return {
            "method": "Hybrid",
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "latency_ms": 0,
            "num_results": 0,
            "results": []
        }

# Test Hybrid search
print("Testing Hybrid Search...")
test_result = search_hybrid("What are treatments for covid?", num_results=5)

if test_result["success"]:
    print(f"✓ Hybrid Search working")
    print(f"  Latency: {test_result['latency_ms']:.2f}ms")
    print(f"  Results: {test_result['num_results']}")
    print(f"\nTop result:")
    if test_result["results"]:
        print(f"  Score: {test_result['results'][0]['score']:.4f}")
        print(f"  Text: {test_result['results'][0]['text']}")
else:
    print(f"✗ Hybrid Search failed: {test_result.get('error', 'Unknown error')}")
    if 'traceback' in test_result:
        print(f"\nTraceback:\n{test_result['traceback']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Comprehensive Benchmark

# COMMAND ----------

# DBTITLE 1,Execute Benchmark for All Queries
import time
from collections import defaultdict

def run_benchmark(test_queries, num_results=10, iterations=3):
    """
    Run comprehensive benchmark across all search methods
    
    Args:
        test_queries: List of test query dictionaries
        num_results: Number of results per query
        iterations: Number of times to run each query for timing
        
    Returns:
        Dictionary with benchmark results
    """
    results = []
    
    print(f"Running benchmark with {len(test_queries)} queries...")
    print(f"Iterations per query: {iterations}")
    print(f"Results per query: {num_results}")
    print("="*80)
    
    for idx, test_query in enumerate(test_queries, 1):
        query_text = test_query["query"]
        category = test_query["category"]
        
        print(f"\n[{idx}/{len(test_queries)}] {category}: {query_text[:60]}...")
        
        query_results = {
            "query": query_text,
            "category": category,
            "keywords": test_query.get("keywords", [])
        }
        
        # Test each method
        methods = [
            ("ANN", lambda: search_ann(query_text, num_results)),
            ("Full-Text", lambda: search_fulltext(query_text, num_results)),
            ("Hybrid", lambda: search_hybrid(query_text, num_results))
        ]
        
        for method_name, search_func in methods:
            latencies = []
            result = None
            
            # Run multiple iterations for accurate timing
            for i in range(iterations):
                result = search_func()
                if result["success"]:
                    latencies.append(result["latency_ms"])
            
            if latencies:
                avg_latency = sum(latencies) / len(latencies)
                min_latency = min(latencies)
                max_latency = max(latencies)
                
                query_results[method_name] = {
                    "success": True,
                    "avg_latency_ms": avg_latency,
                    "min_latency_ms": min_latency,
                    "max_latency_ms": max_latency,
                    "num_results": result["num_results"],
                    "results": result.get("results", [])
                }
                
                print(f"  ✓ {method_name:12s}: {avg_latency:6.2f}ms (avg) | {result['num_results']} results")
            else:
                query_results[method_name] = {
                    "success": False,
                    "error": result.get("error", "Unknown error") if result else "No results"
                }
                print(f"  ✗ {method_name:12s}: FAILED")
        
        results.append(query_results)
    
    print("\n" + "="*80)
    print("✓ Benchmark complete!")
    
    return results

# Run the benchmark
benchmark_results = run_benchmark(
    TEST_QUERIES[:NUM_TEST_QUERIES],  # Use configured number of queries
    num_results=RESULTS_PER_QUERY,
    iterations=BENCHMARK_ITERATIONS
)

# Save results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_json = json.dumps(benchmark_results, indent=2, default=str)

print(f"\nBenchmark completed at {timestamp}")
print(f"Total queries tested: {len(benchmark_results)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Performance Analysis

# COMMAND ----------

from benchmark_code import render_benchmark_results
render_benchmark_results(benchmark_results)

# COMMAND ----------

# DBTITLE 1,Results Quality Comparison
# Analyze result overlap between methods
overlap_analysis = []

for result in benchmark_results:
    query = result["query"]
    
    # Get result IDs for each method
    ann_ids = set([r["chunk_id"] for r in result.get("ANN", {}).get("results", [])])
    ft_ids = set([r["chunk_id"] for r in result.get("Full-Text", {}).get("results", [])])
    hybrid_ids = set([r["chunk_id"] for r in result.get("Hybrid", {}).get("results", [])])
    
    # Calculate overlaps
    ann_ft_overlap = len(ann_ids & ft_ids)
    ann_hybrid_overlap = len(ann_ids & hybrid_ids)
    ft_hybrid_overlap = len(ft_ids & hybrid_ids)
    all_overlap = len(ann_ids & ft_ids & hybrid_ids)
    
    overlap_analysis.append({
        "Query": query[:40] + "...",
        "Category": result["category"],
        "ANN ∩ Full-Text": ann_ft_overlap,
        "ANN ∩ Hybrid": ann_hybrid_overlap,
        "Full-Text ∩ Hybrid": ft_hybrid_overlap,
        "All Three": all_overlap,
        "ANN Results": len(ann_ids),
        "Full-Text Results": len(ft_ids),
        "Hybrid Results": len(hybrid_ids)
    })

df_overlap = pd.DataFrame(overlap_analysis)

if not df_overlap.empty:
    print("="*80)
    print("RESULT OVERLAP ANALYSIS")
    print("="*80)
    print("\nHow many results are shared between methods?")
    print("(Higher overlap = methods agree on relevance)")
    print()
    
    display(df_overlap)
    
    # Visualize overlap
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    overlap_cols = ["ANN ∩ Full-Text", "ANN ∩ Hybrid", "Full-Text ∩ Hybrid", "All Three"]
    df_overlap[overlap_cols].mean().plot(kind='bar')
    plt.title("Average Result Overlap Between Methods")
    plt.ylabel("Number of Shared Results")
    plt.xlabel("Method Combination")
    plt.xticks(rotation=45, ha='right')
    
    plt.subplot(1, 2, 2)
    result_cols = ["ANN Results", "Full-Text Results", "Hybrid Results"]
    df_overlap[result_cols].mean().plot(kind='bar', color=['blue', 'green', 'orange'])
    plt.title("Average Number of Results Returned")
    plt.ylabel("Number of Results")
    plt.xlabel("Method")
    plt.xticks(rotation=45, ha='right')
    plt.axhline(y=RESULTS_PER_QUERY, color='r', linestyle='--', label='Target')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# COMMAND ----------

# DBTITLE 1,Interactive Query Selector
from benchmark_code import display_query_comparison as dqc

# Check if benchmark results exist
if 'benchmark_results' not in dir() or not benchmark_results:
    print("⚠️  No benchmark results available!")
    print("\nPlease run Cell 15 (Execute Benchmark for All Queries) first.")
    print("\nThe benchmark will:")
    print("  1. Test all search methods (ANN, Full-Text, Hybrid)")
    print("  2. Run multiple iterations for accurate timing")
    print("  3. Generate comparison data")
    print("\nOnce complete, this interactive widget will let you explore the results.")
else:
    # Create dropdown to select different queries
    from IPython.display import display as ipython_display
    from ipywidgets import Dropdown, VBox, Button, Output

    def create_query_selector():
        """
        Create interactive widget to select and compare queries
        """
        query_options = [
            (f"{i+1}. [{r['category']}] {r['query'][:60]}...", i)
            for i, r in enumerate(benchmark_results)
        ]
        
        dropdown = Dropdown(
            options=query_options,
            description='Select Query:',
            style={'description_width': '100px'},
            layout={'width': '800px'}
        )
        
        button = Button(description="Show Comparison")
        output = Output()
        
        def on_button_click(b):
            with output:
                output.clear_output()
                dqc(benchmark_results,dropdown.value)
        
        button.on_click(on_button_click)
        
        return VBox([dropdown, button, output])

    # Display interactive selector
    try:
        ipython_display(create_query_selector())
    except:
        print("Interactive widgets not available. Use dqc(index) instead.")
        print("\nAvailable queries:")
        for i, r in enumerate(benchmark_results):
            print(f"  {i}: [{r['category']}] {r['query'][:60]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Performance Summary & Recommendations

# COMMAND ----------

# DBTITLE 1,Generate Summary Report

from benchmark_code import generate_summary_report  # Ensure this function exists in the module or import from the correct location

# Generate report
generate_summary_report(benchmark_results)

# COMMAND ----------

# MAGIC %sql
# MAGIC ALTER TABLE gaurav_catalog.medical_research.silver_papers_embeddings
# MAGIC SET TBLPROPERTIES (
# MAGIC   'delta.enableDeletionVectors' = 'false'
# MAGIC );
# MAGIC
# MAGIC ALTER TABLE gaurav_catalog.medical_research.silver_papers_embeddings
# MAGIC SET TBLPROPERTIES (  'delta.columnMapping.mode' = 'name',
# MAGIC   'delta.enableIcebergCompatV2' = 'true',
# MAGIC   'delta.universalFormat.enabledFormats' = 'iceberg');
# MAGIC
# MAGIC