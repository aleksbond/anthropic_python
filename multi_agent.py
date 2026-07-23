import os
from anthropic import Anthropic

client = Anthropic(
    # This is the default and can be omitted
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)

def coordinator_call(user_query, subagent_results=None):
    """
    This function simulates a coordinator agent that takes a user query and optional subagent results,
    and returns a response from the Anthropic API.
    
    :param user_query: The query from the user.
    :param subagent_results: Optional results from subagents to provide context.
    :return: The response from the Anthropic API.
    """
    messages = [
        {
            "role": "user",
            "content": user_query,
        }
    ]
    
    if subagent_results:
        messages.append({
            "role": "assistant",
            "content": f"Subagent results: {subagent_results}"
        })
    
    message = client.messages.create(
        max_tokens=1024,
        messages=messages,
        model="claude-opus-4-8",
    )
    
    response_text = ""
    for block in message.content:
        if block.type == "text":
            response_text += block.text
    
    return response_text

def web_search_subagent(brief):
    """
    This function simulates a web search subagent that takes a brief query and returns search results.
    
    :param brief: The brief query for the web search.
    :return: Simulated search results.
    """
    # In a real implementation, this would perform a web search and return results.
    # Here, we simulate it with a placeholder response.
    simulated_results = f"Simulated search results for query: '{brief}'"
    return simulated_results

def document_analysis_subagent(brief):
    """
    This function simulates a document analysis subagent that takes a brief query and returns analysis results.
    
    :param brief: The brief query for document analysis.
    :return: Simulated analysis results.
    """
    # In a real implementation, this would analyze documents and return results.
    # Here, we simulate it with a placeholder response.
    simulated_results = f"Simulated document analysis results for query: '{brief}'"
    return simulated_results


def parse_delegation(coordinator_response):
    """Parse coordinator output for a simple delegation decision."""
    normalized = coordinator_response.strip().upper()
    if "SEARCH" in normalized:
        return "SEARCH"
    if "DOCUMENT" in normalized:
        return "DOCUMENT"
    return "NONE"


def orchestrator_query(user_query):
    """Call the coordinator, capture subagent results, and return both."""
    coordinator_response = coordinator_call(user_query)
    delegation = parse_delegation(coordinator_response)
    subagent_results = {}

    if delegation == "SEARCH":
        subagent_results["search"] = web_search_subagent(user_query)
    elif delegation == "DOCUMENT":
        subagent_results["document"] = document_analysis_subagent(user_query)

    return {
        "user_query": user_query,
        "coordinator_response": coordinator_response,
        "delegation": delegation,
        "subagent_results": subagent_results,
    }


def orchestrator_loop():
    """Simple loop that calls the coordinator and returns captured subagent results."""
    print("Starting simple orchestrator loop. Type 'exit' to stop.")

    while True:
        user_query = input("\nUser query: ").strip()
        if not user_query:
            continue
        if user_query.lower() in {"exit", "quit", "q"}:
            print("Exiting orchestrator.")
            break

        result = orchestrator_query(user_query)
        print("\nCoordinator decision:")
        print(result["coordinator_response"])
        print("Delegation:", result["delegation"])
        print("Subagent results:", result["subagent_results"])


orchestrator_loop()