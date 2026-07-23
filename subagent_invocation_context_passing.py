import os
import asyncio
from anthropic import Anthropic
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional

client = Anthropic(
    # This is the default and can be omitted
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)


@dataclass
class AgentDefinition:
    name: str
    description: str
    tools: List[str] = field(default_factory=list)
    must_not_have: List[str] = field(default_factory=list)
    system_prompt: str = ""
    output_format: Optional[Dict] = None  # Optional structured output format


# Agent 1: SearchAgent
SearchAgent = AgentDefinition(
    name="SearchAgent",
    description=(
        "Invoke when the user query requires up-to-date or external information from the web. "
        "Performs targeted searches and returns concise findings."
    ),
    tools=["web_search", "read_document"],
    must_not_have=["direct_file_write", "database_write", "privileged_system_access"],
    system_prompt=(
        "Perform precise web searches; prioritize recent, authoritative sources and cite them succinctly. "
        "Return results ONLY in this JSON format:\n"
        "{\n"
        '  "documents": [\n'
        '    {\n'
        '      "source_url": "https://example.com",\n'
        '      "document_name": "Article Title or Filename",\n'
        '      "page_number": 1,\n'
        '      "content": "Full extracted text"\n'
        "    }\n"
        "  ]\n"
        "}"
    ),
    output_format={  # Structured constraint (if your API supports it)
        "type": "object",
        "properties": {
            "documents": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source_url": {"type": "string"},
                        "document_name": {"type": "string"},
                        "page_number": {"type": "integer"},
                        "content": {"type": "string"}
                    },
                    "required": ["source_url", "document_name", "page_number", "content"]
                }
            }
        },
        "required": ["documents"]
    },
)


# Agent 2: AnalysisAgent
AnalysisAgent = AgentDefinition(
    name="AnalysisAgent",
    description=(
        "Invoke to extract claims, facts, and evidence from documents provided by SearchAgent. Map each claim to its source. "
        "Extract facts, summarize sections, and answer queries grounded in the document."
    ),
    tools=["read_document"],
    must_not_have=["web_search", "unrestricted_network_access"],
    system_prompt=(
        "You will be provided with `search_results` from the SearchAgent in JSON format containing a `documents` array (each item includes: source_url, document_name, page_number, content). "
        "Use the search_results as input: cross-check claims in the provided document(s), map each extracted claim to its source(s) from search_results, and return factual extracts, concise summaries, and exact citations when available. "
        "When citing, include the source_url and document_name for each claim."
        "Return results ONLY in this JSON format:\n"
        "{\n"
        '  "claims": [\n'
        '    {\n'
        '      "source_url": "https://example.com",\n'
        '      "document_name": "Article Title or Filename",\n'
        '      "page_number": 1,\n'
        '      "claim_text": "Full extracted text"\n'
        "    }\n"
        "  ]\n"
        "}"
    ),
    output_format={  # Structured constraint (if your API supports it)
        "type": "object",
        "properties": {
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source_url": {"type": "string"},
                        "document_name": {"type": "string"},
                        "page_number": {"type": "integer"},
                        "claim_text": {"type": "string"}
                    },
                    "required": ["source_url", "document_name", "page_number", "claim_text"]
                }
            }
        },
        "required": ["claims"]
    },
)


# Agent 3: SynthesisAgent
SynthesisAgent = AgentDefinition(
    name="SynthesisAgent",
    description=(
        "Invoke first for new user queries: decide decomposition, select/delgate subagents, and synthesize results. "
        "Coordinates subagents and produces the final answer."
    ),
    tools=["orchestrator_api", "task_queue", "synthesis_model", "logger"],
    must_not_have=["direct_db_write", "shell_access", "privileged_keys"],
    system_prompt=(
        "Decompose queries, pick the right subagents, and synthesize their outputs into a clear final response."
    ),
    output_format=None
)


# Exported list of agent definitions
AGENT_DEFINITIONS: List[AgentDefinition] = [SearchAgent, AnalysisAgent, SynthesisAgent]

