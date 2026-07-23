import os
import asyncio
from anthropic import Anthropic
from datetime import datetime

client = Anthropic(
    # This is the default and can be omitted
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)

# Subagent 1: Web Search
def web_search_subagent(brief: str, query: str) -> dict:
    """
    brief: Full context and instructions for this subagent
    query: The specific search task
    Returns: {result, timestamp, subagent_id}
    """
    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": f"{brief}\n\nSearch for information about: {query}"}
        ]
    )
    return {
        "result": response.content[0].text if response.content else "No results found.",
        "timestamp": datetime.now(),
        "subagent_id": "web_search"
    }

# Subagent 2: Document Analysis
def document_analysis_subagent(brief: str, document: str, query: str) -> dict:
    """
    brief: Full context and instructions
    document: The text to analyze (passed by coordinator)
    query: The specific analysis task
    Returns: {result, timestamp, subagent_id}
    """
    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": f"{brief}\n\nAnalyze the following document for information related to {query}: {document}"}
        ]
    )
    return {
        "result": response.content[0].text if response.content else "No results found.",
        "timestamp": datetime.now(),
        "subagent_id": "document_analysis"
    }

def make_web_search_brief(user_query: str) -> str:
    """Create a concise brief for the web search subagent based on the user query."""
    return (
        "You are a Web Search Agent. Use the user query to search for the most relevant, "
        "recent, and authoritative information available. Present findings clearly and briefly, "
        "including any implications or business impact when appropriate."
        f"\n\nUser query: {user_query}"
    )


def make_document_analysis_brief(user_query: str) -> str:
    """Create a document analysis brief from the user query."""
    return (
        "You are a Document Analysis Agent. Review the provided document carefully and "
        "extract the information that directly answers the user's request. Focus on facts, "
        "important conclusions, and any relevant context or implications."
        f"\n\nUser query: {user_query}"
    )


async def _run_subagents_parallel(
    web_search_brief: str,
    doc_analysis_brief: str,
    user_query: str,
    sample_document: str,
) -> tuple[dict, dict]:
    return await asyncio.gather(
        asyncio.to_thread(web_search_subagent, web_search_brief, user_query),
        asyncio.to_thread(document_analysis_subagent, doc_analysis_brief, sample_document, user_query),
    )


def synthesize_results(user_query: str, web_result: dict, doc_result: dict) -> str:
    """Synthesize subagent outputs into a single coordinator response."""
    prompt = (
        "You are a coordinator agent. Combine the results from a web search and a document analysis into "
        "a concise, coherent answer to the user's original question. Highlight the most relevant facts and "
        "note when the sources agree or differ."
        f"\n\nUser query: {user_query}"
        f"\n\nWeb search result:\n{web_result['result']}"
        f"\n\nDocument analysis result:\n{doc_result['result']}"
    )

    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.content[0].text if response.content else "No synthesis available."


def coordinator(user_query: str) -> dict:
    """
    Receives a broad query, delegates to both subagents in parallel,
    synthesizes results.
    
    Decompose, delegate in parallel, synthesize.
    Returns: {synthesis, timings, subagent_results}
    """
    # Step 1: Define briefs based on user_query
    web_search_brief = make_web_search_brief(user_query)
    doc_analysis_brief = make_document_analysis_brief(user_query)
    
    # Step 2: Define sample document (coordinator provides this)
    sample_document = """
    Quantum Computing: Hardware Breakthroughs and Business Impact
    
    Recent advances in quantum computing hardware have accelerated significantly. 
    Superconducting qubits remain the dominant approach, with companies like IBM, Google, 
    and Rigetti pushing error correction improvements. Ion-trap systems from IonQ and 
    Honeywell show promise for stability. Photonic quantum computing is emerging as an 
    alternative with lower cooling requirements.
    
    Key developments:
    - Google's Willow chip (2024): 99.7% error suppression, approaching practical utility
    - IBM's modular approach: scaling to 1000+ qubits by 2025
    - IonQ's trapped-ion systems: higher fidelity but slower gate speeds
    - Amazon Braket: quantum-as-a-service making hardware accessible
    
    Business implications include potential disruption in drug discovery, materials science, 
    cryptography, and optimization problems. Early adopters in finance and pharma are 
    exploring use cases. However, practical quantum advantage remains 3-5 years away for 
    most real-world applications.
    """
    
    # Step 3: Launch both subagents in parallel
    start = datetime.now()
    web_result, doc_result = asyncio.run(
        _run_subagents_parallel(web_search_brief, doc_analysis_brief, user_query, sample_document)
    )
    end = datetime.now()
    
    # Step 4: Measure time and collect results
    timings = {
        "total": (end - start).total_seconds(),
        "web_search": web_result["timestamp"],
        "document_analysis": doc_result["timestamp"],
    }
    
    # Step 5: Synthesize
    synthesis = synthesize_results(user_query, web_result, doc_result)
    return {
        "synthesis": synthesis,
        "timings": timings,
        "subagent_results": {
            "web_search": web_result,
            "document_analysis": doc_result,
        },
    }

def check_coverage_and_refine(
    user_query: str,
    initial_synthesis: str,
    web_result: dict,
    doc_result: dict,
) -> dict:
    """
    Checks if initial synthesis fully covered the user query.
    If gaps detected, re-delegates targeted queries to fill them.
    
    Returns: {refined_synthesis, gaps_detected, follow_up_queries, additional_results}
    """
    # Step 1: Ask AI to identify gaps
    gap_detection_prompt = (
        "Review this user query and synthesis. Identify any aspects NOT adequately covered:\n"
        f"User query: {user_query}\n"
        f"Current synthesis: {initial_synthesis}\n"
        "List gaps concisely (e.g., 'missing timeline', 'no competitive landscape')."
    )
    
    gap_response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=512,
        messages=[{"role": "user", "content": gap_detection_prompt}]
    )
    gaps = gap_response.content[0].text if gap_response.content else ""
    
    # Step 2: If gaps exist, re-delegate
    if "no gaps" not in gaps.lower() and gaps.strip():
        # Generate follow-up query based on gaps
        follow_up_query = f"Based on gaps in coverage: {gaps}. Provide targeted information to fill these gaps."
        
        # Re-delegate (e.g., web search for new angle)
        follow_up_result = web_search_subagent(make_web_search_brief(follow_up_query), follow_up_query)
        
        # Re-synthesize with new findings
        refined_synthesis = synthesize_results(
            user_query,
            web_result,  # original
            {"result": follow_up_result["result"] + "\n" + doc_result["result"]}  # combined
        )
        
        return {
            "refined_synthesis": refined_synthesis,
            "gaps_detected": gaps,
            "follow_up_query": follow_up_query,
            "additional_results": follow_up_result,
        }

if __name__ == "__main__":
    user_query = "What are the latest developments in quantum computing hardware, and what are the business implications?"
    result = coordinator(user_query)
    
    print("=" * 80)
    print("COORDINATOR SYNTHESIS:")
    print("=" * 80)
    print(result["synthesis"])
    print("\n" + "=" * 80)
    print("TIMING COMPARISON:")
    print("=" * 80)
    print(f"Parallel execution:   {result['timings']['total']:.2f} seconds")
    # print(f"Sequential execution: {result['timings']['total']:.2f} seconds")
    # print(f"Speedup factor:       {result['timings']['total']:.2f}x")

    refined_result = check_coverage_and_refine(
        user_query,
        result["synthesis"],
        result["subagent_results"]["web_search"],
        result["subagent_results"]["document_analysis"],
    )

    print("=" * 80)
    print("REFINED COORDINATOR SYNTHESIS:")
    print("=" * 80)
    print(f"Refined synthesis: {refined_result['refined_synthesis']}")
    print(f"Gaps detected: {refined_result['gaps_detected']}")
    print(f"Follow-up query: {refined_result['follow_up_query']}")
    print(f"Additional results: {refined_result['additional_results']['result']}")
    print("\n" + "=" * 80)
