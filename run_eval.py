# run_eval.py
import sys
from pathlib import Path
import time
import pandas as pd
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.engine_load import RAGEngine


def main():
    print("🚀 Running Full Vantage Core RAG Evaluation...\n")
    
    engine = RAGEngine()   # loads everything once

    test_queries = [
        "What is the fiscal deficit target for 2026-27?",
        "Summarize the key highlights of Economic Survey 2025-26.",
        "What are the major allocations in Budget at a Glance 2026-27?",
        "How much is allocated to defence?",
        "What is the projected GDP growth for FY 2026-27?",
        "Explain revenue and capital receipts in the budget.",
        "What are the major agriculture schemes announced?",
        "What is the total expenditure for 2026-27?",
        "How does the budget address inflation?",
        "Key recommendations of Economic Survey on employment?",
        "Compare capital expenditure this year vs last year.",
        "What is the tax-to-GDP ratio target?",
        "Any new green energy initiatives?",
        "What is the fiscal deficit as % of GDP?",
    ]

    results = []
    empty_count = 0
    print(f"Testing {len(test_queries)} diverse queries (silent mode)...\n")

    for i, query in enumerate(test_queries, 1):
        print(f"[{i:2d}/{len(test_queries)}] Running: {query[:70]}...")
        
        metrics = engine.ask(query, verbose=False)   # silent + returns all metrics
        
        row = {
            "Query": query[:55] + "..." if len(query) > 55 else query,
            "Total_Latency_s": metrics.get("total_latency", 0),
            "Retrieval_s": metrics.get("retrieval_time", 0),
            "Generation_s": metrics.get("generation_time", 0),
            "Top_Rerank_Score": metrics.get("top_rerank_score", 0),
            "Avg_Rerank_Score": metrics.get("avg_rerank_score", 0),
            "Docs_Retrieved": metrics.get("docs_retrieved", 0),
            "Input_Tokens": metrics.get("input_tokens", 0),
            "Output_Tokens": metrics.get("output_tokens", 0),
            "Total_Tokens": metrics.get("total_tokens", 0),
            "Tokens_per_Second": metrics.get("tokens_per_second", 0),
            "Peak_VRAM_MB": metrics.get("peak_vram_mb", 0),
        }
        
        if metrics.get("empty_retrieval", False):
            empty_count += 1
            row["Empty_Retrieval"] = "YES"
        else:
            row["Empty_Retrieval"] = "No"
            
        results.append(row)

    df = pd.DataFrame(results)

    # Professional Summary Table
    print("\n" + "="*120)
    print("📊 FINAL VANTAGE CORE RAG EVALUATION REPORT")
    print("="*120)
    print(tabulate(df, headers="keys", tablefmt="pretty", floatfmt=".2f"))

    print("\n📈 KEY STATISTICS (Industry Standard)")
    print(f"   Average Total Latency     : {df['Total_Latency_s'].mean():.2f} s")
    print(f"   Avg Generation Time       : {df['Generation_s'].mean():.2f} s")
    print(f"   Avg Retrieval Time        : {df['Retrieval_s'].mean():.2f} s")
    print(f"   Avg Tokens per Query      : {df['Total_Tokens'].mean():.0f}")
    print(f"   Median (P50) Latency      : {df['Total_Latency_s'].median():.2f} s")
    print(f"   P95 Latency               : {df['Total_Latency_s'].quantile(0.95):.2f} s")
    print(f"   Empty Retrieval Rate      : {empty_count/len(test_queries)*100:.1f}%")
    print(f"   Avg Top Rerank Score      : {df['Top_Rerank_Score'].mean():.4f}")
    print(f"   Avg Rerank Score (all)    : {df['Avg_Rerank_Score'].mean():.4f}")
    print(f"   Avg Tokens/sec            : {df['Tokens_per_Second'].mean():.1f}")
    print(f"   Avg Peak VRAM             : {df['Peak_VRAM_MB'].mean():.1f} MB")

    df.to_csv("evaluation_results.csv", index=False)
    print("\n✅ Full detailed results saved to evaluation_results.csv")
    print("   You can now open this CSV in Excel/Google Sheets for deeper analysis.")



if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    total_seconds = time.time() - start_time
    minutes = int(total_seconds // 60)
    seconds = round(total_seconds % 60, 2)
    print(f"\n{'='*50}")
    print(f"✅ Total Evaluation Runtime: {minutes} min {seconds} sec")
    print(f"{'='*50}")