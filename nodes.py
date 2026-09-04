"""LangGraph Nodes Implementation for Travel Policy Agent.

Each function represents an isolated node in the LangGraph workflow:
1. rewrite_query_node  (LLM)       - Refines/optimizes search query based on feedback
2. retrieve_docs_node  (Vector DB) - Retrieves top-k chunks from FAISS vector store
3. validate_context_node (LLM)     - Grades context relevance against the query
4. generate_answer_node  (LLM)     - Synthesizes grounded answer from context
5. evaluate_answer_node  (LLM)     - Assesses answer faithfulness and completeness
6. fallback_node        (Python)   - Handles out-of-scope / missing policy topics
"""

import json
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

from src.state import AgentState
from src.llm import get_llm
from src.retriever import PolicyRetriever

# Lazy global retriever instance
_retriever = None


def get_policy_retriever() -> PolicyRetriever:
    global _retriever
    if _retriever is None:
        _retriever = PolicyRetriever()
    return _retriever


# -------------------------------------------------------------------------
# Node 1: Query Rewriter (LLM)
# -------------------------------------------------------------------------
def rewrite_query_node(state: AgentState) -> Dict[str, Any]:
    """Node 1: Rewrites the user query to optimize vector retrieval.
    
    Why LLM? Natural language queries can be colloquial, vague, or conversational.
    The LLM converts them into precise keyword-rich search queries for the vector store.
    If this is a retry attempt, it incorporates feedback to explore alternate terms.
    """
    original_q = state.get("original_question", "")
    retry_count = state.get("retry_count", 0)
    feedback = state.get("evaluation_feedback", "")
    trace_logs = list(state.get("trace_logs", []))

    if retry_count == 0:
        # First pass: standard reformulation
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an AI query optimizer for an airline policy search engine. "
                "Your job is to rewrite the user query into a clear, search-friendly query "
                "focused on airline rules, baggage, check-in, cancellation, refunds, pets, or travel guidelines. "
                "Output ONLY the rewritten search query and nothing else."
            )),
            ("user", "User question: {question}"),
        ])
        chain = prompt | get_llm(temperature=0.0) | StrOutputParser()
        rewritten = chain.invoke({"question": original_q}).strip().strip('"')
        log_msg = f"[Node: query_rewriter] Optimized query: '{rewritten}'"
    else:
        # Retry pass: adjust search terms using feedback
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an AI query optimizer for an airline policy search engine. "
                "The previous retrieval attempt failed or produced insufficient context. "
                "Rewrite the query using alternative airline terminology, synonyms, or related policy concepts "
                "to find the relevant policy section. "
                "Output ONLY the revised search query and nothing else."
            )),
            ("user", (
                "Original Question: {original_question}\n"
                "Previous Query: {previous_query}\n"
                "Feedback / Reason for Retry: {feedback}\n\n"
                "Revised Search Query:"
            )),
        ])
        chain = prompt | get_llm(temperature=0.2) | StrOutputParser()
        rewritten = chain.invoke({
            "original_question": original_q,
            "previous_query": state.get("rewritten_question", original_q),
            "feedback": feedback,
        }).strip().strip('"')
        log_msg = f"[Node: query_rewriter] (Retry #{retry_count}) Rephrased query with feedback: '{rewritten}'"

    trace_logs.append(log_msg)
    return {
        "rewritten_question": rewritten,
        "trace_logs": trace_logs,
    }


# -------------------------------------------------------------------------
# Node 2: Retriever (Vector DB / Python Logic)
# -------------------------------------------------------------------------
def retrieve_docs_node(state: AgentState) -> Dict[str, Any]:
    """Node 2: Retrieves relevant airline policy chunks from the vector database.
    
    Why Vector DB / Python Logic? Pure deterministic retrieval of top-k semantic
    vectors without invoking an unnecessary LLM.
    """
    query = state.get("rewritten_question") or state.get("original_question", "")
    trace_logs = list(state.get("trace_logs", []))

    retriever = get_policy_retriever()
    docs = retriever.retrieve(query)
    context_text = PolicyRetriever.format_docs(docs)

    log_msg = f"[Node: retriever] Retrieved {len(docs)} document chunk(s) for query: '{query}'"
    trace_logs.append(log_msg)

    return {
        "retrieved_docs": docs,
        "context_text": context_text,
        "trace_logs": trace_logs,
    }


# -------------------------------------------------------------------------
# Node 3: Context Validator (LLM / Structured Assessment)
# -------------------------------------------------------------------------
def validate_context_node(state: AgentState) -> Dict[str, Any]:
    """Node 3: Validates if the retrieved chunks contain relevant information.
    
    Why LLM? Semantic relevance grading ensures we don't attempt to hallucinate
    answers when the vector store returns weakly matching or off-topic text.
    """
    question = state.get("original_question", "")
    context = state.get("context_text", "")
    docs = state.get("retrieved_docs", [])
    trace_logs = list(state.get("trace_logs", []))

    # Short-circuit if zero documents retrieved
    if not docs or not context.strip() or context == "No documents found.":
        log_msg = "[Node: context_validator] No documents retrieved -> context marked IRRELEVANT."
        trace_logs.append(log_msg)
        return {
            "is_context_relevant": False,
            "evaluation_feedback": "No documents retrieved from vector store.",
            "trace_logs": trace_logs,
        }

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a strict context evaluator for airline policy questions. "
            "Determine whether the retrieved context contains useful rules, policies, or guidelines "
            "to answer the user's question.\n"
            "Respond in JSON format with exactly two keys:\n"
            '{{"is_relevant": true/false, "reason": "brief explanation"}}'
        )),
        ("user", "User Question: {question}\n\nRetrieved Context:\n{context}\n\nJSON Evaluation:"),
    ])

    chain = prompt | get_llm(temperature=0.0) | StrOutputParser()
    raw_response = chain.invoke({"question": question, "context": context})

    try:
        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        data = json.loads(cleaned)
        is_relevant = bool(data.get("is_relevant", False))
        reason = str(data.get("reason", "No reason provided."))
    except Exception:
        # Fallback heuristic if LLM output isn't strict JSON
        is_relevant = "true" in raw_response.lower()
        reason = raw_response.strip()

    log_msg = f"[Node: context_validator] Relevance: {is_relevant} | Reason: {reason}"
    trace_logs.append(log_msg)

    return {
        "is_context_relevant": is_relevant,
        "evaluation_feedback": reason if not is_relevant else "",
        "trace_logs": trace_logs,
    }


# -------------------------------------------------------------------------
# Node 4: Answer Generator (LLM)
# -------------------------------------------------------------------------
def generate_answer_node(state: AgentState) -> Dict[str, Any]:
    """Node 4: Generates a factual, grounded answer strictly based on policy context.
    
    Why LLM? Synthesizes complex policy clauses into clear, polite, structured advice
    for the passenger while maintaining factual adherence.
    """
    question = state.get("original_question", "")
    context = state.get("context_text", "")
    trace_logs = list(state.get("trace_logs", []))

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a helpful and professional Airline Travel Policy Assistant. "
            "Answer the user's question strictly using the provided airline policy context.\n"
            "- Be direct, polite, and precise.\n"
            "- If applicable, mention specific limits, fees, requirements, or conditions stated in the policy.\n"
            "- Cite the source document or section if present.\n"
            "- Do NOT fabricate rules or make assumptions not supported by the context."
        )),
        ("user", "User Question: {question}\n\nAirline Policy Context:\n{context}\n\nAnswer:"),
    ])

    chain = prompt | get_llm(temperature=0.0) | StrOutputParser()
    answer = chain.invoke({"question": question, "context": context}).strip()

    log_msg = f"[Node: answer_generator] Generated answer ({len(answer)} chars)."
    trace_logs.append(log_msg)

    return {
        "answer": answer,
        "trace_logs": trace_logs,
    }


# -------------------------------------------------------------------------
# Node 5: Answer Evaluator (LLM)
# -------------------------------------------------------------------------
def evaluate_answer_node(state: AgentState) -> Dict[str, Any]:
    """Node 5: Assesses the generated answer for hallucination and completeness.
    
    Why LLM? Acts as an automated quality assurance guardrail to ensure the answer
    is truthful to the context and directly resolves the user's prompt.
    """
    question = state.get("original_question", "")
    context = state.get("context_text", "")
    answer = state.get("answer", "")
    trace_logs = list(state.get("trace_logs", []))

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an AI quality assurance evaluator for airline customer support. "
            "Evaluate whether the generated answer is faithful (grounded in the context without hallucinations) "
            "and directly answers the user's question.\n"
            "Respond in JSON format with exactly two keys:\n"
            '{{"is_grounded": true/false, "feedback": "brief feedback"}}'
        )),
        ("user", (
            "User Question: {question}\n\n"
            "Context:\n{context}\n\n"
            "Generated Answer:\n{answer}\n\n"
            "JSON Evaluation:"
        )),
    ])

    chain = prompt | get_llm(temperature=0.0) | StrOutputParser()
    raw_response = chain.invoke({
        "question": question,
        "context": context,
        "answer": answer,
    })

    try:
        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        data = json.loads(cleaned)
        is_grounded = bool(data.get("is_grounded", True))
        feedback = str(data.get("feedback", ""))
    except Exception:
        is_grounded = "true" in raw_response.lower()
        feedback = raw_response.strip()

    log_msg = f"[Node: answer_evaluator] Grounded & Valid: {is_grounded} | Feedback: {feedback}"
    trace_logs.append(log_msg)

    return {
        "is_answer_grounded": is_grounded,
        "evaluation_feedback": feedback if not is_grounded else "",
        "trace_logs": trace_logs,
    }


# -------------------------------------------------------------------------
# Node 6: Fallback Node (Python Logic)
# -------------------------------------------------------------------------
def fallback_node(state: AgentState) -> Dict[str, Any]:
    """Node 6: Handles cases where no valid policy information was found.
    
    Why Python Logic? Prevents hallucinations by returning a standardized, polite
    refusal when policies lack the requested topic.
    """
    question = state.get("original_question", "")
    feedback = state.get("evaluation_feedback", "")
    trace_logs = list(state.get("trace_logs", []))

    fallback_answer = (
        f"I'm sorry, but I could not find specific guidelines regarding '{question}' "
        "in the provided airline travel policy document.\n\n"
        "💡 Recommendation: Please consult the airline's official customer support desk "
        "or verify the terms on their website for up-to-date details."
    )

    log_msg = f"[Node: fallback_node] Routed to fallback due to lack of policy coverage. Feedback: {feedback}"
    trace_logs.append(log_msg)

    return {
        "answer": fallback_answer,
        "trace_logs": trace_logs,
    }
