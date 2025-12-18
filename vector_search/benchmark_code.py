import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Extract latency data
latency_data = []

def render_benchmark_results(benchmark_results):
    latency_data = []
    score_data = []

    # -----------------------
    # Extract Latency + Score
    # -----------------------
    for result in benchmark_results:
        query = result.get("query", "")[:40] + "..."
        category = result.get("category", "Unknown")
        
        for method in ["ANN", "Full-Text", "Hybrid"]:
            if method in result and result[method].get("success"):

                # Latency
                latency_data.append({
                    "Query": query,
                    "Category": category,
                    "Method": method,
                    "Avg Latency (ms)": result[method]["avg_latency_ms"],
                    "Min Latency (ms)": result[method]["min_latency_ms"],
                    "Max Latency (ms)": result[method]["max_latency_ms"]
                })

                # ---------------------
                # Score Parsing Section
                # ---------------------
                # Score format example: x['score'] for x in result[method]['results']
                if "results" in result[method]:
                    for entry in result[method]["results"]:
                        if "score" in entry:
                            score_data.append({
                                "Query": query,
                                "Category": category,
                                "Method": method,
                                "Score": entry["score"]
                            })

    # Convert to DataFrames
    df_latency = pd.DataFrame(latency_data)
    df_score = pd.DataFrame(score_data)

    # -----------------------------
    # Render Latency Analysis
    if not df_latency.empty:
        print("="*80)
        print("LATENCY ANALYSIS")
        print("="*80)

        summary = df_latency.groupby("Method")["Avg Latency (ms)"].agg(['mean', 'min', 'max', 'std'])
        print("\nSummary Statistics:")
        print(summary.to_string())

        plt.figure(figsize=(14, 6))

        # Box plot
        plt.subplot(1, 2, 1)
        sns.boxplot(data=df_latency, x="Method", y="Avg Latency (ms)")
        plt.title("Latency Distribution by Method")
        plt.ylabel("Latency (ms)")
        plt.xticks(rotation=0)

        # Bar chart by category
        plt.subplot(1, 2, 2)
        pivot_latency = df_latency.pivot_table(
            values="Avg Latency (ms)",
            index="Category",
            columns="Method"
        )
        pivot_latency.plot(kind='bar', ax=plt.gca())
        plt.title("Average Latency by Query Category")
        plt.ylabel("Latency (ms)")
        plt.xlabel("Category")
        plt.xticks(rotation=45, ha='right')
        plt.legend(title="Method")

        plt.tight_layout()
        display(df_latency)
    else:
        print("No latency data available for analysis")

    # -----------------------------
    # Render Score Analysis
    # -----------------------------
    if not df_score.empty:
        print("\n" + "="*80)
        print("SCORE ANALYSIS")
        print("="*80)

        score_summary = df_score.groupby("Method")["Score"].agg(['mean', 'min', 'max', 'std'])
        print("\nScore Summary Statistics:")
        print(score_summary.to_string())

        plt.figure(figsize=(14, 6))

        # Box plot of scores
        plt.subplot(1, 2, 1)
        sns.boxplot(data=df_score, x="Method", y="Score")
        plt.title("Score Distribution by Method")
        plt.ylabel("Score")
        plt.xticks(rotation=0)

        # Bar chart by category (mean score)
        plt.subplot(1, 2, 2)
        pivot_score = df_score.pivot_table(
            values="Score",
            index="Category",
            columns="Method"
        )
        pivot_score.plot(kind='bar', ax=plt.gca())
        plt.title("Average Score by Query Category")
        plt.ylabel("Score")
        plt.xlabel("Category")
        plt.xticks(rotation=45, ha='right')
        plt.legend(title="Method")

        plt.tight_layout()
        display(df_score)
    else:
        print("No score data available for analysis")

    return plt.show()
  


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
    
        return summary


def display_query_comparison(benchmark_results,query_idx=0):
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



