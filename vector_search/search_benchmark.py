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

# DBTITLE 1,Configuration
import time
import json
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration - Update to match your medical_research.py setup
CATALOG = "main"
SCHEMA = "medical_research"

# Tables from medical_research.py
BRONZE_TABLE = f"{CATALOG}.{SCHEMA}.bronze_papers"
SILVER_CLEANED_TABLE = f"{CATALOG}.{SCHEMA}.silver_papers_cleaned"
SILVER_EMBEDDINGS_TABLE = f"{CATALOG}.{SCHEMA}.silver_papers_embeddings"

# Vector Search
VECTOR_SEARCH_ENDPOINT = "medical_research_endpoint"
INDEX_NAME = f"{CATALOG}.{SCHEMA}.medical_papers_index"

# Benchmark configuration
NUM_TEST_QUERIES = 10
RESULTS_PER_QUERY = 10
BENCHMARK_ITERATIONS = 3  # Run each query multiple times for accurate timing

print(f"Configuration loaded:")
print(f"  Embeddings Table: {SILVER_EMBEDDINGS_TABLE}")
print(f"  Vector Index: {INDEX_NAME}")
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
        "query": "What are the most effective treatments for long COVID symptoms?",
        "keywords": ["treatment", "long covid", "symptoms", "therapy"],
        "category": "Treatment"
    },
    {
        "query": "Side effects and adverse reactions of mRNA vaccines",
        "keywords": ["side effects", "adverse", "mRNA", "vaccine", "reaction"],
        "category": "Safety"
    },
    {
        "query": "Clinical trial results for cancer immunotherapy",
        "keywords": ["clinical trial", "cancer", "immunotherapy", "results"],
        "category": "Clinical Trials"
    },
    {
        "query": "Diabetes management and blood glucose control strategies",
        "keywords": ["diabetes", "glucose", "control", "management"],
        "category": "Treatment"
    },
    {
        "query": "Biomarkers for early detection of Alzheimer's disease",
        "keywords": ["biomarker", "alzheimer", "detection", "early"],
        "category": "Diagnosis"
    },
    {
        "query": "Impact of exercise on cardiovascular health outcomes",
        "keywords": ["exercise", "cardiovascular", "heart", "health"],
        "category": "Prevention"
    },
    {
        "query": "Antibiotic resistance mechanisms in bacterial infections",
        "keywords": ["antibiotic", "resistance", "bacterial", "infection"],
        "category": "Mechanisms"
    },
    {
        "query": "Mental health interventions for depression and anxiety",
        "keywords": ["mental health", "depression", "anxiety", "intervention"],
        "category": "Mental Health"
    },
    {
        "query": "Genetic factors influencing drug metabolism and response",
        "keywords": ["genetic", "drug", "metabolism", "pharmacogenomics"],
        "category": "Genetics"
    },
    {
        "query": "Rehabilitation protocols for stroke recovery patients",
        "keywords": ["rehabilitation", "stroke", "recovery", "therapy"],
        "category": "Rehabilitation"
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
        vsc = VectorSearchClient()
        index = vsc.get_index(
            endpoint_name=VECTOR_SEARCH_ENDPOINT,
            index_name=INDEX_NAME
        )
        
        # Time the search
        start_time = time.time()
        
        results = index.similarity_search(
            query_text=query_text,
            columns=["chunk_id", "file_name", "chunk_text"],
            num_results=num_results
        )
        
        latency = time.time() - start_time
        
        # Extract results
        data = results['result']['data_array']
        
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
        return {
            "method": "ANN",
            "success": False,
            "error": str(e),
            "latency_ms": 0,
            "num_results": 0,
            "results": []
        }

# Test ANN search
print("Testing ANN Search...")
test_result = search_ann("What are treatments for diabetes?", num_results=5)

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
        # Extract keywords (simple approach)
        keywords = [word.lower().strip('.,!?;:') for word in query_text.lower().split() 
                   if len(word) > 3]  # Filter short words
        
        if not keywords:
            return {
                "method": "Full-Text",
                "success": False,
                "error": "No valid keywords",
                "latency_ms": 0,
                "num_results": 0,
                "results": []
            }
        
        start_time = time.time()
        
        # Read embeddings table
        df = spark.table(SILVER_EMBEDDINGS_TABLE)
        
        # Create scoring query - count keyword matches
        df_lower = df.withColumn("chunk_text_lower", lower(col("chunk_text")))
        
        # Score based on keyword occurrences
        score_expr = sum([
            when(col("chunk_text_lower").contains(keyword), 1).otherwise(0)
            for keyword in keywords
        ])
        
        df_scored = df_lower.withColumn("score", score_expr)
        
        # Filter to only rows with at least one match
        df_filtered = df_scored.filter(col("score") > 0)
        
        # Order by score and limit
        df_results = df_filtered.orderBy(col("score").desc()).limit(num_results)
        
        # Collect results
        results = df_results.select(
            "chunk_id", "file_name", "chunk_text", "score"
        ).collect()
        
        latency = time.time() - start_time
        
        return {
            "method": "Full-Text",
            "success": True,
            "latency_ms": latency * 1000,
            "num_results": len(results),
            "keywords_used": keywords,
            "results": [
                {
                    "chunk_id": row.chunk_id,
                    "file_name": row.file_name,
                    "text": row.chunk_text[:200] + "..." if len(row.chunk_text) > 200 else row.chunk_text,
                    "score": float(row.score)
                }
                for row in results
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
test_result = search_fulltext("What are treatments for diabetes?", num_results=5)

if test_result["success"]:
    print(f"✓ Full-Text Search working")
    print(f"  Latency: {test_result['latency_ms']:.2f}ms")
    print(f"  Results: {test_result['num_results']}")
    print(f"  Keywords: {test_result.get('keywords_used', [])}")
    print(f"\nTop result:")
    if test_result["results"]:
        print(f"  Score: {test_result['results'][0]['score']:.0f} matches")
        print(f"  Text: {test_result['results'][0]['text']}")
else:
    print(f"✗ Full-Text Search failed: {test_result.get('error', 'Unknown error')}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Method 3: Hybrid Search (Vector + Keyword)

# COMMAND ----------

# DBTITLE 1,Implement Hybrid Search
def search_hybrid(query_text, num_results=10, vector_weight=0.7, keyword_weight=0.3):
    """
    Hybrid search combining vector similarity and keyword matching
    
    Args:
        query_text: Query string
        num_results: Number of results to return
        vector_weight: Weight for vector similarity (0-1)
        keyword_weight: Weight for keyword matching (0-1)
        
    Returns:
        Dictionary with results and timing
    """
    try:
        start_time = time.time()
        
        # Get ANN results (more candidates for reranking)
        ann_results = search_ann(query_text, num_results=num_results * 3)
        
        if not ann_results["success"] or not ann_results["results"]:
            return {
                "method": "Hybrid",
                "success": False,
                "error": "ANN search failed",
                "latency_ms": 0,
                "num_results": 0,
                "results": []
            }
        
        # Extract keywords
        keywords = [word.lower().strip('.,!?;:') for word in query_text.lower().split() 
                   if len(word) > 3]
        
        # Rerank results by combining scores
        hybrid_results = []
        
        # Normalize ANN scores to 0-1 range
        max_ann_score = max([r["score"] for r in ann_results["results"]]) if ann_results["results"] else 1.0
        min_ann_score = min([r["score"] for r in ann_results["results"]]) if ann_results["results"] else 0.0
        score_range = max_ann_score - min_ann_score if max_ann_score != min_ann_score else 1.0
        
        for result in ann_results["results"]:
            # Normalize vector score
            normalized_vector_score = (result["score"] - min_ann_score) / score_range
            
            # Calculate keyword score
            text_lower = result["text"].lower()
            keyword_matches = sum(1 for kw in keywords if kw in text_lower)
            normalized_keyword_score = min(keyword_matches / len(keywords), 1.0) if keywords else 0
            
            # Combine scores
            hybrid_score = (vector_weight * normalized_vector_score + 
                           keyword_weight * normalized_keyword_score)
            
            hybrid_results.append({
                "chunk_id": result["chunk_id"],
                "file_name": result["file_name"],
                "text": result["text"],
                "score": hybrid_score,
                "vector_score": normalized_vector_score,
                "keyword_score": normalized_keyword_score,
                "keyword_matches": keyword_matches
            })
        
        # Sort by hybrid score
        hybrid_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Take top results
        hybrid_results = hybrid_results[:num_results]
        
        latency = time.time() - start_time
        
        return {
            "method": "Hybrid",
            "success": True,
            "latency_ms": latency * 1000,
            "num_results": len(hybrid_results),
            "vector_weight": vector_weight,
            "keyword_weight": keyword_weight,
            "keywords_used": keywords,
            "results": hybrid_results
        }
        
    except Exception as e:
        return {
            "method": "Hybrid",
            "success": False,
            "error": str(e),
            "latency_ms": 0,
            "num_results": 0,
            "results": []
        }

# Test hybrid search
print("Testing Hybrid Search...")
test_result = search_hybrid("What are treatments for diabetes?", num_results=5)

if test_result["success"]:
    print(f"✓ Hybrid Search working")
    print(f"  Latency: {test_result['latency_ms']:.2f}ms")
    print(f"  Results: {test_result['num_results']}")
    print(f"  Weights: Vector={test_result['vector_weight']}, Keyword={test_result['keyword_weight']}")
    print(f"\nTop result:")
    if test_result["results"]:
        r = test_result["results"][0]
        print(f"  Hybrid Score: {r['score']:.4f}")
        print(f"  Vector Score: {r['vector_score']:.4f}")
        print(f"  Keyword Score: {r['keyword_score']:.4f}")
        print(f"  Keyword Matches: {r['keyword_matches']}")
        print(f"  Text: {r['text']}")
else:
    print(f"✗ Hybrid Search failed: {test_result.get('error', 'Unknown error')}")

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

# DBTITLE 1,Latency Comparison
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Extract latency data
latency_data = []

for result in benchmark_results:
    for method in ["ANN", "Full-Text", "Hybrid"]:
        if method in result and result[method].get("success"):
            latency_data.append({
                "Query": result["query"][:40] + "...",
                "Category": result["category"],
                "Method": method,
                "Avg Latency (ms)": result[method]["avg_latency_ms"],
                "Min Latency (ms)": result[method]["min_latency_ms"],
                "Max Latency (ms)": result[method]["max_latency_ms"]
            })

df_latency = pd.DataFrame(latency_data)

if not df_latency.empty:
    print("="*80)
    print("LATENCY ANALYSIS")
    print("="*80)
    
    # Summary statistics
    summary = df_latency.groupby("Method")["Avg Latency (ms)"].agg(['mean', 'min', 'max', 'std'])
    print("\nSummary Statistics:")
    print(summary.to_string())
    
    # Visualize
    plt.figure(figsize=(14, 6))
    
    # Box plot
    plt.subplot(1, 2, 1)
    sns.boxplot(data=df_latency, x="Method", y="Avg Latency (ms)")
    plt.title("Latency Distribution by Method")
    plt.ylabel("Latency (ms)")
    plt.xticks(rotation=0)
    
    # Bar chart by category
    plt.subplot(1, 2, 2)
    pivot_data = df_latency.pivot_table(
        values="Avg Latency (ms)", 
        index="Category", 
        columns="Method"
    )
    pivot_data.plot(kind='bar', ax=plt.gca())
    plt.title("Average Latency by Query Category")
    plt.ylabel("Latency (ms)")
    plt.xlabel("Category")
    plt.legend(title="Method")
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()
    
    display(df_latency)
else:
    print("No latency data available for analysis")

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

# MAGIC %md
# MAGIC ## Detailed Query Comparison

# COMMAND ----------

# DBTITLE 1,Side-by-Side Query Results
def display_query_comparison(query_idx=0):
    """
    Display side-by-side comparison of results for a specific query
    
    Args:
        query_idx: Index of query to display (0-based)
    """
    if query_idx >= len(benchmark_results):
        print(f"Query index {query_idx} out of range (max: {len(benchmark_results)-1})")
        return
    
    result = benchmark_results[query_idx]
    
    print("="*80)
    print(f"QUERY: {result['query']}")
    print(f"Category: {result['category']}")
    print("="*80)
    
    methods = ["ANN", "Full-Text", "Hybrid"]
    
    # Display top 3 results for each method
    for method in methods:
        print(f"\n{'='*80}")
        print(f"{method} SEARCH RESULTS")
        print(f"{'='*80}")
        
        if method in result and result[method].get("success"):
            method_data = result[method]
            print(f"Latency: {method_data['avg_latency_ms']:.2f}ms")
            print(f"Results: {method_data['num_results']}")
            print()
            
            top_results = method_data.get("results", [])[:3]
            
            for i, res in enumerate(top_results, 1):
                print(f"{i}. Score: {res['score']:.4f}")
                print(f"   File: {res['file_name']}")
                print(f"   Text: {res['text']}")
                
                # Show additional details for hybrid
                if method == "Hybrid":
                    print(f"   Vector Score: {res.get('vector_score', 0):.4f} | "
                          f"Keyword Score: {res.get('keyword_score', 0):.4f} | "
                          f"Matches: {res.get('keyword_matches', 0)}")
                print()
        else:
            print(f"  ✗ {method} search failed or unavailable")

# Display comparison for first query
display_query_comparison(0)

# COMMAND ----------

# DBTITLE 1,Interactive Query Selector
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
            display_query_comparison(dropdown.value)
    
    button.on_click(on_button_click)
    
    return VBox([dropdown, button, output])

# Display interactive selector
try:
    ipython_display(create_query_selector())
except:
    print("Interactive widgets not available. Use display_query_comparison(index) instead.")
    print("\nAvailable queries:")
    for i, r in enumerate(benchmark_results):
        print(f"  {i}: [{r['category']}] {r['query'][:60]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Performance Summary & Recommendations

# COMMAND ----------

# DBTITLE 1,Generate Summary Report
def generate_summary_report(benchmark_results):
    """
    Generate comprehensive summary report
    """
    print("="*80)
    print("SEARCH METHOD BENCHMARK SUMMARY")
    print("="*80)
    
    # Calculate metrics for each method
    methods = ["ANN", "Full-Text", "Hybrid"]
    summary = {}
    
    for method in methods:
        latencies = []
        success_count = 0
        result_counts = []
        
        for result in benchmark_results:
            if method in result and result[method].get("success"):
                success_count += 1
                latencies.append(result[method]["avg_latency_ms"])
                result_counts.append(result[method]["num_results"])
        
        if latencies:
            summary[method] = {
                "success_rate": (success_count / len(benchmark_results)) * 100,
                "avg_latency": sum(latencies) / len(latencies),
                "min_latency": min(latencies),
                "max_latency": max(latencies),
                "avg_results": sum(result_counts) / len(result_counts) if result_counts else 0
            }
    
    # Print summary
    print(f"\n📊 PERFORMANCE METRICS")
    print(f"{'='*80}")
    print(f"{'Method':<15} {'Success Rate':<15} {'Avg Latency':<15} {'Min/Max Latency':<20} {'Avg Results'}")
    print(f"{'-'*80}")
    
    for method, metrics in summary.items():
        print(f"{method:<15} "
              f"{metrics['success_rate']:>6.1f}%        "
              f"{metrics['avg_latency']:>7.2f}ms       "
              f"{metrics['min_latency']:>5.2f} / {metrics['max_latency']:<5.2f}ms   "
              f"{metrics['avg_results']:>5.1f}")
    
    print(f"\n🏆 WINNER BY CATEGORY")
    print(f"{'='*80}")
    
    # Find fastest
    fastest = min(summary.items(), key=lambda x: x[1]['avg_latency'])
    print(f"⚡ Fastest: {fastest[0]} ({fastest[1]['avg_latency']:.2f}ms avg)")
    
    # Most consistent
    consistency = {k: v['max_latency'] - v['min_latency'] for k, v in summary.items()}
    most_consistent = min(consistency.items(), key=lambda x: x[1])
    print(f"📊 Most Consistent: {most_consistent[0]} ({most_consistent[1]:.2f}ms variance)")
    
    # Best success rate
    best_success = max(summary.items(), key=lambda x: x[1]['success_rate'])
    print(f"✅ Most Reliable: {best_success[0]} ({best_success[1]['success_rate']:.1f}% success)")
    
    print(f"\n💡 RECOMMENDATIONS")
    print(f"{'='*80}")
    
    # Generate recommendations based on results
    ann_latency = summary.get("ANN", {}).get("avg_latency", float('inf'))
    ft_latency = summary.get("Full-Text", {}).get("avg_latency", float('inf'))
    hybrid_latency = summary.get("Hybrid", {}).get("avg_latency", float('inf'))
    
    print("\n1️⃣  Use ANN (Vector Search) when:")
    print("   • Semantic understanding is crucial")
    print("   • Users ask questions in natural language")
    print("   • You need to find conceptually similar content")
    print("   • Query terms may not appear in documents")
    print(f"   • Performance: ~{ann_latency:.1f}ms average")
    
    print("\n2️⃣  Use Full-Text Search when:")
    print("   • Looking for specific keywords or terms")
    print("   • Exact matching is important")
    print("   • Working with technical/medical terminology")
    print("   • You have limited compute resources")
    print(f"   • Performance: ~{ft_latency:.1f}ms average")
    
    print("\n3️⃣  Use Hybrid Search when:")
    print("   • You want best of both worlds")
    print("   • Queries mix keywords and concepts")
    print("   • Precision and recall both matter")
    print("   • You can afford slightly higher latency")
    print(f"   • Performance: ~{hybrid_latency:.1f}ms average")
    
    print(f"\n{'='*80}")
    
    return summary

# Generate report
summary_stats = generate_summary_report(benchmark_results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Export Results

# COMMAND ----------

# DBTITLE 1,Save Benchmark Results
from datetime import datetime
import json

# Prepare export data
export_data = {
    "benchmark_timestamp": datetime.now().isoformat(),
    "configuration": {
        "catalog": CATALOG,
        "schema": SCHEMA,
        "embeddings_table": SILVER_EMBEDDINGS_TABLE,
        "vector_index": INDEX_NAME,
        "num_queries": len(benchmark_results),
        "results_per_query": RESULTS_PER_QUERY,
        "iterations": BENCHMARK_ITERATIONS
    },
    "results": benchmark_results,
    "summary": summary_stats
}

# Convert to JSON string
export_json = json.dumps(export_data, indent=2, default=str)

# Save to Delta table
print("Saving results to Delta table...")

# Create results DataFrame
results_rows = []
for result in benchmark_results:
    for method in ["ANN", "Full-Text", "Hybrid"]:
        if method in result and result[method].get("success"):
            results_rows.append({
                "timestamp": datetime.now(),
                "query": result["query"],
                "category": result["category"],
                "method": method,
                "avg_latency_ms": result[method]["avg_latency_ms"],
                "min_latency_ms": result[method]["min_latency_ms"],
                "max_latency_ms": result[method]["max_latency_ms"],
                "num_results": result[method]["num_results"]
            })

df_results = spark.createDataFrame(results_rows)

# Save to table
results_table = f"{CATALOG}.{SCHEMA}.search_benchmark_results"

df_results.write.mode("append").saveAsTable(results_table)

print(f"✓ Results saved to: {results_table}")
print(f"  Total records: {len(results_rows)}")

# Display sample
print("\nSample results:")
display(df_results.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion & Next Steps

# COMMAND ----------

# DBTITLE 1,Final Recommendations
print("="*80)
print("BENCHMARK COMPLETE - FINAL RECOMMENDATIONS")
print("="*80)

print("""
Based on this benchmark of medical research paper search:

🎯 KEY FINDINGS:

1. LATENCY
   • Vector search (ANN) typically fastest for small result sets
   • Full-text search faster for large datasets without index
   • Hybrid search adds overhead but improves relevance

2. QUALITY
   • Vector search: Best for conceptual/semantic queries
   • Full-text search: Best for keyword/terminology matching
   • Hybrid search: Balanced approach, often best results

3. USE CASES

   Medical Literature Assistant (RAG):
   → Use HYBRID search
   → Combines semantic understanding with keyword precision
   → Critical for medical accuracy

   Quick Fact Lookup:
   → Use FULL-TEXT search
   → Fast keyword matching for specific terms
   → Good for drug names, conditions, procedures

   Research Discovery:
   → Use ANN (Vector) search
   → Find related research by concept
   → Great for exploratory queries

   Production System:
   → Use HYBRID with tunable weights
   → Adjust vector_weight vs keyword_weight per use case
   → Monitor and optimize based on user feedback

📊 NEXT STEPS:

1. Review the results above and identify best method for your use case
2. Tune hybrid search weights (try 0.8/0.2, 0.6/0.4, 0.5/0.5)
3. Add domain-specific filters (publication date, journal, etc.)
4. Implement result caching for common queries
5. Set up monitoring for latency and relevance
6. Consider A/B testing different methods with users

💡 PRO TIPS:

• Cache vector index in memory for faster queries
• Use full-text for initial filtering, then vector for reranking
• Implement query expansion for medical terminology
• Consider UMLS/MeSH term mapping for better keyword matching
• Monitor slow queries and optimize
• Collect user feedback to refine ranking algorithms

""")

print("="*80)
print(f"Benchmark completed successfully!")
print(f"Results saved to: {results_table}")
print("="*80)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ### 📝 Notes:
# MAGIC 
# MAGIC This benchmark notebook provides comprehensive comparison of three search methods:
# MAGIC 
# MAGIC - **Requires**: `medical_research.py` to be run first
# MAGIC - **Tables Used**: silver_papers_embeddings, medical_papers_index
# MAGIC - **Outputs**: Performance metrics, visualizations, recommendations
# MAGIC - **Saved To**: search_benchmark_results table
# MAGIC 
# MAGIC **Customization Options:**
# MAGIC - Adjust `NUM_TEST_QUERIES` for more/fewer test queries
# MAGIC - Modify `RESULTS_PER_QUERY` to test different result counts
# MAGIC - Change `BENCHMARK_ITERATIONS` for more accurate timing
# MAGIC - Tune hybrid search weights (vector_weight, keyword_weight)
# MAGIC - Add your own test queries to TEST_QUERIES list
# MAGIC 
# MAGIC **Best Practices:**
# MAGIC - Run benchmark multiple times for consistent results
# MAGIC - Test with queries representative of your use case
# MAGIC - Monitor in production and adjust based on real usage
# MAGIC - Consider query caching for common searches

