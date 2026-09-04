"""LangGraph Workflow Definition for Travel Policy Agent.

This module constructs the complete StateGraph, wires nodes with standard
and conditional edges, and compiles the workflow with LangSmith tracing.
"""

from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

from src.state import AgentState
from src.nodes import (
    rewrite_query_node,
    retrieve_docs_node,
    validate_context_node,
    generate_answer_node,
    evaluate_answer_node,
    fallback_node,
)


def increment_retry_node(state: AgentState) -> dict:
    """Explicit state update node that increments the retry counter before re-looping."""
    current_retries = state.get("retry_count", 0)
    new_retries = current_retries + 1
    trace_logs = list(state.get("trace_logs", []))
    trace_logs.append(f"[Node: increment_retry] Retry counter incremented: {current_retries} -> {new_retries}")
    return {
        "retry_count": new_retries,
        "trace_logs": trace_logs,
    }


# -------------------------------------------------------------------------
# Conditional Routing Functions
# -------------------------------------------------------------------------
def route_after_context_validation(
    state: AgentState,
) -> Literal["generate_answer", "increment_retry", "fallback"]:
    """Conditional Edge: Decides where to route after evaluating context relevance.
    
    Routes:
    - 'generate_answer': Context is relevant -> proceed to answer generation.
    - 'increment_retry': Context is NOT relevant and retry budget remains -> retry via query rewriting.
    - 'fallback': Context is NOT relevant and max retries exceeded -> fallback polite response.
    """
    is_relevant = state.get("is_context_relevant", False)
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    if is_relevant:
        return "generate_answer"
    elif retry_count < max_retries:
        return "increment_retry"
    else:
        return "fallback"


def route_after_answer_evaluation(
    state: AgentState,
) -> Literal["__end__", "increment_retry", "fallback"]:
    """Conditional Edge: Decides where to route after quality-checking the answer.
    
    Routes:
    - END: Answer is verified to be faithful and grounded.
    - 'increment_retry': Answer is ungrounded/hallucinated and retry budget remains -> re-query.
    - 'fallback': Answer failed evaluation and max retries exceeded -> fallback response.
    """
    is_grounded = state.get("is_answer_grounded", True)
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    if is_grounded:
        return END
    elif retry_count < max_retries:
        return "increment_retry"
    else:
        return "fallback"


# -------------------------------------------------------------------------
# Graph Construction & Compilation
# -------------------------------------------------------------------------
def create_travel_policy_graph() -> CompiledStateGraph:
    """Builds and compiles the Travel Policy LangGraph workflow.
    
    Graph Topology:
    
    START 
      │
      ▼
    query_rewriter ◄────────────┐ (on retry)
      │                         │
      ▼                         │
    retriever                   │
      │                         │
      ▼                         │
    context_validator           │
      │                         │
      ├── [Relevant: False & retry < max] ──► increment_retry ──┘
      ├── [Relevant: False & retry >= max] ─► fallback ──► END
      │
      └── [Relevant: True]
            │
            ▼
    answer_generator
            │
            ▼
    answer_evaluator
            │
            ├── [Grounded: False & retry < max] ──► increment_retry ──┘
            ├── [Grounded: False & retry >= max] ─► fallback ──► END
            │
            └── [Grounded: True] ──► END
    """
    workflow = StateGraph(AgentState)

    # 1. Register Nodes
    workflow.add_node("query_rewriter", rewrite_query_node)
    workflow.add_node("retriever", retrieve_docs_node)
    workflow.add_node("context_validator", validate_context_node)
    workflow.add_node("answer_generator", generate_answer_node)
    workflow.add_node("answer_evaluator", evaluate_answer_node)
    workflow.add_node("increment_retry", increment_retry_node)
    workflow.add_node("fallback", fallback_node)

    # 2. Add Fixed Edges
    workflow.add_edge(START, "query_rewriter")
    workflow.add_edge("query_rewriter", "retriever")
    workflow.add_edge("retriever", "context_validator")
    workflow.add_edge("answer_generator", "answer_evaluator")
    workflow.add_edge("increment_retry", "query_rewriter")
    workflow.add_edge("fallback", END)

    # 3. Add Conditional Edges
    workflow.add_conditional_edges(
        "context_validator",
        route_after_context_validation,
        {
            "generate_answer": "answer_generator",
            "increment_retry": "increment_retry",
            "fallback": "fallback",
        },
    )

    workflow.add_conditional_edges(
        "answer_evaluator",
        route_after_answer_evaluation,
        {
            END: END,
            "increment_retry": "increment_retry",
            "fallback": "fallback",
        },
    )

    # 4. Compile Graph
    return workflow.compile()
