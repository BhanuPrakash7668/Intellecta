# In test_workflow.py

import asyncio
import os
import json
from src.core.graph import research_workflow

async def main():
    """
    Main function to run a simple test of the research workflow.
    """
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ ERROR: Please set your OPENAI_API_KEY environment variable before running.")
        return

    test_query = "The role of attention mechanisms in natural language processing"
    kwargs = {
        "max_results": 5,
        "focus_areas": ["attention mechanism", "natural language processing"]
    }

    print(f"--- 🚀 Starting Test Workflow for query: '{test_query}' ---")
    final_result = await research_workflow.run_research_workflow(query=test_query, **kwargs)

    print("\n--- ✅ Workflow Complete ---")
    print(f"Session ID: {final_result.get('session_id')}")
    print(f"Status: {final_result.get('status')}")

    # --- ADD THIS SECTION TO PRINT RETRIEVED PAPERS ---
    if final_result.get("retrieved_papers"):
        print("\n--- 📚 Retrieved Papers ---")
        papers = final_result["retrieved_papers"]
        for i, paper in enumerate(papers):
            print(f"[{i+1}] Title: {paper.get('title')}")
            print(f"    Authors: {', '.join(paper.get('authors', []))}")
            print(f"    Source: {paper.get('source')}")
            print(f"    URL: {paper.get('url')}")
    # --------------------------------------------------
    
    if final_result.get("research_report"):
        report_str = json.dumps(final_result["research_report"], indent=2)
        print("\n--- 📋 Final Research Report ---")
        print(report_str)
        
    elif final_result.get("error_messages"):
        print(f"\n--- ❌ Workflow failed with errors ---")
        for error in final_result["error_messages"]:
            print(f"- {error}")

if __name__ == "__main__":
    asyncio.run(main())