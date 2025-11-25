# Databricks notebook source
# MAGIC %md
# MAGIC # Example Queries - MLB Prospects Vector Search
# MAGIC 
# MAGIC This notebook demonstrates various ways to query the MLB prospects vector search index.
# MAGIC 
# MAGIC **Prerequisites**: Run `mlb_prospects_vector_search.py` first to create the index.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

# DBTITLE 1,Configuration
from databricks.vector_search.client import VectorSearchClient
from databricks.sdk import WorkspaceClient
import pandas as pd
from pyspark.sql.functions import col

# Configuration (update to match your environment)
CATALOG = "main"
SCHEMA = "mlb_prospects"
VECTOR_SEARCH_ENDPOINT = "mlb_prospects_endpoint"
INDEX_NAME = f"{CATALOG}.{SCHEMA}.mlb_prospects_index"

# Initialize clients
vsc = VectorSearchClient()
w = WorkspaceClient()

# Get the index
index = vsc.get_index(
    endpoint_name=VECTOR_SEARCH_ENDPOINT,
    index_name=INDEX_NAME
)

print(f"✓ Connected to index: {INDEX_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic Queries

# COMMAND ----------

# DBTITLE 1,Simple Search Query
# Basic similarity search
results = index.similarity_search(
    query_text="Who are the top pitching prospects?",
    columns=["chunk_text", "file_name", "source_url"],
    num_results=5
)

# Display results
print("Top 5 Results:")
print("="*80)
for i, row in enumerate(results['result']['data_array'], 1):
    chunk_text, file_name, source_url, score = row
    print(f"\n{i}. Score: {score:.4f}")
    print(f"   Source: {file_name}")
    print(f"   Text: {chunk_text[:200]}...")
    print("-"*80)

# COMMAND ----------

# DBTITLE 1,Search with More Results
# Get more comprehensive results
results = index.similarity_search(
    query_text="college baseball prospects with power hitting",
    columns=["chunk_id", "chunk_text", "chunk_sequence"],
    num_results=10
)

# Convert to DataFrame for easier analysis
data = results['result']['data_array']
df = pd.DataFrame(data, columns=["chunk_id", "chunk_text", "chunk_sequence", "score"])

print(f"Found {len(df)} results")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Advanced Queries

# COMMAND ----------

# DBTITLE 1,Query with Filters
# Search with metadata filters (if your index supports it)
try:
    results = index.similarity_search(
        query_text="shortstop prospects",
        columns=["chunk_text", "chunk_length"],
        filters={"chunk_length": {"$gt": 200}},  # Only chunks > 200 chars
        num_results=5
    )
    
    print("Results with filters applied:")
    for i, row in enumerate(results['result']['data_array'], 1):
        chunk_text, chunk_length, score = row
        print(f"{i}. Length: {chunk_length}, Score: {score:.4f}")
        print(f"   {chunk_text[:150]}...\n")
except Exception as e:
    print(f"Filters not supported or error: {e}")

# COMMAND ----------

# DBTITLE 1,Multi-Query Search
# Compare results for multiple queries
queries = [
    "best power hitters in the draft",
    "top defensive players",
    "fastest runners and base stealers",
    "pitchers with high velocity"
]

all_results = []

for query in queries:
    results = index.similarity_search(
        query_text=query,
        columns=["chunk_text"],
        num_results=3
    )
    
    for row in results['result']['data_array']:
        chunk_text, score = row
        all_results.append({
            "query": query,
            "score": score,
            "text": chunk_text[:100] + "..."
        })

# Display as DataFrame
results_df = pd.DataFrame(all_results)
display(results_df.sort_values("score", ascending=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Analytical Queries

# COMMAND ----------

# DBTITLE 1,Find Similar Prospects
def find_similar_prospects(prospect_name, num_results=5):
    """
    Find prospects similar to a given prospect
    """
    # Search for the prospect
    results = index.similarity_search(
        query_text=f"information about {prospect_name}",
        columns=["chunk_text", "chunk_id"],
        num_results=num_results
    )
    
    print(f"Prospects similar to {prospect_name}:")
    print("="*80)
    
    for i, row in enumerate(results['result']['data_array'], 1):
        chunk_text, chunk_id, score = row
        print(f"\n{i}. Similarity: {score:.4f}")
        print(f"   {chunk_text[:250]}...")
        print("-"*80)
    
    return results

# Example usage
find_similar_prospects("Chase Burns", num_results=5)

# COMMAND ----------

# DBTITLE 1,Compare Two Prospects
def compare_prospects(prospect1, prospect2):
    """
    Compare two prospects by finding information about each
    """
    print(f"COMPARING: {prospect1} vs {prospect2}")
    print("="*80)
    
    # Search for first prospect
    results1 = index.similarity_search(
        query_text=prospect1,
        columns=["chunk_text"],
        num_results=3
    )
    
    # Search for second prospect
    results2 = index.similarity_search(
        query_text=prospect2,
        columns=["chunk_text"],
        num_results=3
    )
    
    print(f"\n📊 {prospect1}:")
    print("-"*80)
    for i, row in enumerate(results1['result']['data_array'], 1):
        print(f"{i}. {row[0][:200]}...\n")
    
    print(f"\n📊 {prospect2}:")
    print("-"*80)
    for i, row in enumerate(results2['result']['data_array'], 1):
        print(f"{i}. {row[0][:200]}...\n")

# Example usage
compare_prospects("power hitting prospect", "contact hitting prospect")

# COMMAND ----------

# DBTITLE 1,Aggregate Statistics on Search Results
def analyze_search_results(query, num_results=20):
    """
    Analyze search results statistics
    """
    results = index.similarity_search(
        query_text=query,
        columns=["chunk_text", "chunk_length", "chunk_sequence"],
        num_results=num_results
    )
    
    data = results['result']['data_array']
    df = pd.DataFrame(data, columns=["chunk_text", "chunk_length", "chunk_sequence", "score"])
    
    print(f"ANALYSIS: '{query}'")
    print("="*80)
    print(f"Total results: {len(df)}")
    print(f"Average score: {df['score'].mean():.4f}")
    print(f"Score range: {df['score'].min():.4f} - {df['score'].max():.4f}")
    print(f"Average chunk length: {df['chunk_length'].mean():.0f} chars")
    print(f"Chunk length range: {df['chunk_length'].min()} - {df['chunk_length'].max()} chars")
    
    # Score distribution
    print("\nScore Distribution:")
    print(df['score'].describe())
    
    return df

# Example usage
df_analysis = analyze_search_results("top draft picks", num_results=20)
display(df_analysis)

# COMMAND ----------

# MAGIC %md
# MAGIC ## RAG (Retrieval Augmented Generation) Examples

# COMMAND ----------

# DBTITLE 1,Simple RAG Pattern
def rag_query(question, num_context=3):
    """
    Simple RAG: Retrieve context and use it to answer questions
    """
    print(f"Question: {question}")
    print("="*80)
    
    # Step 1: Retrieve relevant context
    results = index.similarity_search(
        query_text=question,
        columns=["chunk_text"],
        num_results=num_context
    )
    
    # Extract context
    contexts = [row[0] for row in results['result']['data_array']]
    combined_context = "\n\n".join(contexts)
    
    print("\n📚 Retrieved Context:")
    print("-"*80)
    for i, ctx in enumerate(contexts, 1):
        print(f"\nContext {i}:")
        print(ctx[:300] + "...")
    
    # Step 2: Format for LLM (pseudo-code - you would send this to an LLM)
    prompt = f"""Based on the following context about MLB draft prospects, answer the question.

Context:
{combined_context}

Question: {question}

Answer:"""
    
    print("\n\n🤖 Prompt for LLM:")
    print("-"*80)
    print(prompt[:500] + "...")
    
    # In production, you would send 'prompt' to your LLM here
    # answer = your_llm_function(prompt)
    
    return {
        "question": question,
        "contexts": contexts,
        "prompt": prompt
    }

# Example usage
rag_query("What are the key strengths of the top pitching prospects?", num_context=3)

# COMMAND ----------

# DBTITLE 1,Multi-Step RAG with Follow-up
def multi_step_rag(initial_question, follow_up_questions):
    """
    Multi-step RAG: Use initial context for follow-up questions
    """
    print(f"Initial Question: {initial_question}")
    print("="*80)
    
    # Get initial context
    initial_results = index.similarity_search(
        query_text=initial_question,
        columns=["chunk_text", "chunk_id"],
        num_results=5
    )
    
    initial_context = [row[0] for row in initial_results['result']['data_array']]
    
    print("\n📚 Initial Context Retrieved:")
    print(f"Found {len(initial_context)} relevant chunks\n")
    
    # Process follow-up questions
    for i, follow_up in enumerate(follow_up_questions, 1):
        print(f"\nFollow-up {i}: {follow_up}")
        print("-"*80)
        
        # Search with refined context
        results = index.similarity_search(
            query_text=follow_up,
            columns=["chunk_text"],
            num_results=3
        )
        
        for j, row in enumerate(results['result']['data_array'], 1):
            chunk_text, score = row
            print(f"  {j}. Score: {score:.4f}")
            print(f"     {chunk_text[:150]}...\n")

# Example usage
multi_step_rag(
    initial_question="Tell me about the top prospects in the 2025 draft",
    follow_up_questions=[
        "Which of these prospects are pitchers?",
        "What are their velocity ranges?",
        "Which ones have college experience?"
    ]
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Specialized Search Patterns

# COMMAND ----------

# DBTITLE 1,Hybrid Search (Vector + Keyword)
def hybrid_search(query_text, keyword_filter=None, num_results=10):
    """
    Combine vector search with keyword filtering
    """
    # Vector search
    vector_results = index.similarity_search(
        query_text=query_text,
        columns=["chunk_id", "chunk_text"],
        num_results=num_results * 2  # Get more candidates
    )
    
    results = []
    for row in vector_results['result']['data_array']:
        chunk_id, chunk_text, score = row
        
        # Apply keyword filter if provided
        if keyword_filter:
            if keyword_filter.lower() in chunk_text.lower():
                results.append({
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "vector_score": score,
                    "keyword_match": True
                })
        else:
            results.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "vector_score": score,
                "keyword_match": False
            })
    
    # Return top results
    results_df = pd.DataFrame(results[:num_results])
    return results_df

# Example usage
print("Hybrid Search: Vector similarity + keyword filtering")
hybrid_results = hybrid_search(
    query_text="athletic players with good defense",
    keyword_filter="shortstop",
    num_results=5
)
display(hybrid_results)

# COMMAND ----------

# DBTITLE 1,Semantic Clustering of Results
def cluster_search_results(query, num_results=20):
    """
    Search and attempt to cluster results by similarity
    """
    results = index.similarity_search(
        query_text=query,
        columns=["chunk_text", "chunk_sequence"],
        num_results=num_results
    )
    
    data = results['result']['data_array']
    df = pd.DataFrame(data, columns=["chunk_text", "chunk_sequence", "score"])
    
    # Simple clustering based on score ranges
    df['cluster'] = pd.cut(df['score'], bins=3, labels=['Low', 'Medium', 'High'])
    
    print(f"CLUSTERED RESULTS: '{query}'")
    print("="*80)
    
    for cluster_name in ['High', 'Medium', 'Low']:
        cluster_df = df[df['cluster'] == cluster_name]
        if len(cluster_df) > 0:
            print(f"\n{cluster_name} Relevance Cluster ({len(cluster_df)} items):")
            print("-"*40)
            for _, row in cluster_df.head(3).iterrows():
                print(f"  Score: {row['score']:.4f}")
                print(f"  {row['chunk_text'][:100]}...\n")
    
    return df

# Example usage
clustered_results = cluster_search_results("pitching prospects with high upside", num_results=15)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Performance Testing

# COMMAND ----------

# DBTITLE 1,Query Latency Benchmark
import time

def benchmark_queries(queries, num_iterations=5):
    """
    Benchmark query performance
    """
    results = []
    
    for query in queries:
        latencies = []
        
        for _ in range(num_iterations):
            start = time.time()
            index.similarity_search(
                query_text=query,
                columns=["chunk_id", "chunk_text"],
                num_results=10
            )
            latency = time.time() - start
            latencies.append(latency)
        
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        
        results.append({
            "query": query[:50] + "..." if len(query) > 50 else query,
            "avg_latency": f"{avg_latency:.3f}s",
            "min_latency": f"{min_latency:.3f}s",
            "max_latency": f"{max_latency:.3f}s"
        })
    
    # Display results
    print("QUERY PERFORMANCE BENCHMARK")
    print("="*80)
    results_df = pd.DataFrame(results)
    display(results_df)
    
    return results_df

# Test queries
test_queries = [
    "top pitching prospects",
    "best defensive players in the draft",
    "power hitting college prospects",
    "high school players with upside",
    "prospects with good speed and contact skills"
]

benchmark_queries(test_queries, num_iterations=3)

# COMMAND ----------

# DBTITLE 1,Throughput Test
def throughput_test(query, duration_seconds=10):
    """
    Test how many queries can be processed in a given time
    """
    print(f"THROUGHPUT TEST")
    print(f"Query: '{query}'")
    print(f"Duration: {duration_seconds}s")
    print("="*80)
    
    start_time = time.time()
    query_count = 0
    
    while time.time() - start_time < duration_seconds:
        index.similarity_search(
            query_text=query,
            columns=["chunk_id"],
            num_results=10
        )
        query_count += 1
    
    elapsed = time.time() - start_time
    qps = query_count / elapsed
    
    print(f"\n✓ Completed {query_count} queries in {elapsed:.2f}s")
    print(f"✓ Throughput: {qps:.2f} queries/second")
    print(f"✓ Average latency: {1000/qps:.2f}ms per query")
    
    return qps

# Run throughput test
throughput_test("test query", duration_seconds=10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Utility Functions

# COMMAND ----------

# DBTITLE 1,Pretty Print Search Results
def pretty_search(query, num_results=5):
    """
    Nicely formatted search results
    """
    results = index.similarity_search(
        query_text=query,
        columns=["chunk_text", "file_name", "chunk_sequence"],
        num_results=num_results
    )
    
    print("🔍 SEARCH RESULTS")
    print("="*80)
    print(f"Query: '{query}'")
    print(f"Results: {num_results}")
    print("="*80)
    
    for i, row in enumerate(results['result']['data_array'], 1):
        chunk_text, file_name, chunk_sequence, score = row
        
        print(f"\n#{i} | Score: {score:.4f} | Source: {file_name} | Chunk: {chunk_sequence}")
        print("-"*80)
        print(chunk_text)
        print()

# Example usage
pretty_search("players with the best tools", num_results=3)

# COMMAND ----------

# DBTITLE 1,Export Results to Delta Table
def export_search_results(query, table_name, num_results=100):
    """
    Export search results to a Delta table for further analysis
    """
    results = index.similarity_search(
        query_text=query,
        columns=["chunk_id", "chunk_text", "file_name", "chunk_sequence"],
        num_results=num_results
    )
    
    # Convert to DataFrame
    data = results['result']['data_array']
    columns = ["chunk_id", "chunk_text", "file_name", "chunk_sequence", "score"]
    
    results_df = spark.createDataFrame(
        [dict(zip(columns, row)) for row in data]
    )
    
    # Add query metadata
    from pyspark.sql.functions import lit, current_timestamp
    results_df = results_df.withColumn("query", lit(query)) \
                           .withColumn("query_timestamp", current_timestamp())
    
    # Write to Delta
    results_df.write.mode("overwrite").saveAsTable(table_name)
    
    print(f"✓ Exported {len(data)} results to {table_name}")
    return results_df

# Example usage
export_table = f"{CATALOG}.{SCHEMA}.search_results_example"
exported_df = export_search_results(
    query="top prospects",
    table_name=export_table,
    num_results=50
)
display(exported_df)

# COMMAND ----------

# DBTITLE 1,Batch Query Processing
def batch_query(queries, num_results_per_query=5):
    """
    Process multiple queries and aggregate results
    """
    all_results = []
    
    print(f"Processing {len(queries)} queries...")
    
    for query in queries:
        results = index.similarity_search(
            query_text=query,
            columns=["chunk_text"],
            num_results=num_results_per_query
        )
        
        for row in results['result']['data_array']:
            chunk_text, score = row
            all_results.append({
                "query": query,
                "chunk_text": chunk_text,
                "score": score
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    print(f"✓ Processed {len(queries)} queries")
    print(f"✓ Total results: {len(all_results)}")
    
    return df

# Example usage
batch_queries = [
    "best pitchers",
    "top position players",
    "high school prospects",
    "college prospects",
    "international prospects"
]

batch_results = batch_query(batch_queries, num_results_per_query=3)
display(batch_results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC 
# MAGIC This notebook demonstrated:
# MAGIC 
# MAGIC 1. **Basic Queries**: Simple similarity search
# MAGIC 2. **Advanced Queries**: Filters, multi-query, analytical searches
# MAGIC 3. **RAG Patterns**: Retrieval augmented generation examples
# MAGIC 4. **Specialized Searches**: Hybrid search, clustering
# MAGIC 5. **Performance Testing**: Latency and throughput benchmarks
# MAGIC 6. **Utilities**: Export, batch processing, pretty printing
# MAGIC 
# MAGIC ### Next Steps:
# MAGIC 
# MAGIC - Integrate with your LLM for full RAG applications
# MAGIC - Build a chatbot using these patterns
# MAGIC - Create dashboards with search analytics
# MAGIC - Implement re-ranking for better results
# MAGIC - Add hybrid search with BM25
# MAGIC 
# MAGIC ### Resources:
# MAGIC 
# MAGIC - [Main Pipeline Notebook](mlb_prospects_vector_search.py)
# MAGIC - [README](README.md)
# MAGIC - [Troubleshooting Guide](TROUBLESHOOTING.md)
# MAGIC - [Architecture Docs](ARCHITECTURE.md)

