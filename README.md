# TRAVEL-POLICY-AGENT-USING-LANGGRAPH-RAG-AND-LANGSMITH

1. Project Overview

This project is a AI Travel Policy Agent built using LangGraph, RAG, FAISS, and LangSmith.

The agent answers travel policy-related questions by retrieving relevant information from an airline policy PDF and generating responses based on the retrieved context.

The main focus of this project is to explore LangGraph architecture, state management, nodes, conditional edges, answer evaluation, and retry/revision loops.

The project uses free LLM options and local Hugging Face embeddings, making it suitable for a practical learning and portfolio project.

2. Objectives

* Build a RAG-based Travel Policy Agent.
* Learn and implement LangGraph architecture.
* Implement state management using AgentState.
* Create modular LangGraph nodes.
* Implement conditional routing.
* Validate retrieved context before answer generation.
* Evaluate generated answers.
* Implement retry and revision loops.
* Integrate LangSmith for observability.
* Use free LLM and local embedding models.

3. Knowledge Base

The agent uses an airline policy PDF as its knowledge base.
The knowledge base contains information related to areas such as:

* Baggage Policy
* Check-in Policy
* Cancellation Policy
* Refund Policy
* Boarding Policy
* Pet Policy
* Travel Documents
* International Travel Guidelines

The PDF is loaded, split into smaller chunks, converted into embeddings, and stored in a local FAISS vector store.

 4. Tools & Technologies

* Python
* LangGraph
* LangChain
* LangSmith
* FAISS
* Hugging Face Embeddings
* Qwen 2.5
* PyPDF
* NumPy
* python-dotenv

#5. Methodology

Step 1: Document Ingestion
The airline policy PDF is loaded using PyPDF and divided into smaller text chunks.

Step 2: Embedding Generation
The document chunks are converted into vector embeddings using a Hugging Face embedding model running locally on CPU.

Step 3: Vector Store
The generated embeddings are stored in a local FAISS vector database for similarity-based document retrieval.

Step 4: Query Rewriting
The user's question is processed by the `query_rewriter` node to improve the query before retrieval.

Step 5: Document Retrieval
The retriever node searches the FAISS vector store and retrieves relevant policy information.

Step 6: Context Validation
The context_validator node checks whether the retrieved context is relevant to the user's question.
If the context is not relevant and retries are available, the workflow performs another iteration.

Step 7: Answer Generation
If relevant context is available, the `generate_answer` node generates a response using the retrieved policy information.

Step 8: Answer Evaluation
The evaluate_answer node evaluates whether the generated response is sufficiently grounded in the retrieved context.

Step 9: Retry / Revision
If the answer does not meet the required conditions and retries are available, the workflow performs another iteration.

Step 10: Fallback
If the maximum number of retries is reached, the workflow moves to the fallback path.

6. LangGraph Workflow
                 START
                           │
                           ▼
                  ┌─────────────────┐
                  │  query_rewriter │◄──────────────┐
                  └────────┬────────┘               │
                           │                        │
                           ▼                        │
                  ┌─────────────────┐               │
                  │    retriever    │               │
                  └────────┬────────┘               │
                           │                        │
                           ▼                        │
                  ┌────────────────────┐            │
                  │ context_validator  │            │
                  └─────────┬──────────┘            │
                            │                       │
                  Context Relevant?                │
                       /          \                 │
                     No            Yes              │
                     │              │               │
                     ▼              ▼               │
              increment_retry   generate_answer     │
                     │              │               │
                     └──────────────┼───────────────┤
                                    │
                                    ▼
                           evaluate_answer
                                    │
                          Answer Grounded?
                              /        \
                            No          Yes
                            │            │
                            ▼            ▼
                     increment_retry  fallback
                            │            │
                            └────────────┘
                                    │
                                    ▼
                                   END
 7. LangGraph Concepts Demonstrated

| Concept                   | Location        | Purpose                                                             |
| :------------------------ | :-------------- | :------------------------------------------------------------------ |
| StateGraph           | src/graph.py  | Defines and orchestrates the workflow                               |
| AgentState            | src/state.py  | Maintains query, documents, validation and evaluation information   |
| Nodes                | src/nodes.py  | Performs individual processing tasks                                |
| Conditional Edges    | src/graph.py  | Controls workflow routing based on conditions                       |
| Retry / Revision Loop | src/graph.py  | Re-runs the workflow when context or answer quality is insufficient |
| LangSmith Tracing     | src/config.py | Provides execution tracing and observability                        |

8. LangSmith Observability

LangSmith is integrated to monitor and debug the LangGraph workflow.
It provides visibility into:

* Graph execution
* Individual node execution
* LLM calls
* Retrieval steps
* Execution flow
* Latency
* Token usage

This helps understand how a user query moves through the different stages of the agent.

9. Project Structure

travel-policy-agent/
│
├── data/
│   └── airline-guidelines.pdf
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── state.py
│   ├── ingestion.py
│   ├── retriever.py
│   ├── llm.py
│   ├── nodes.py
│   └── graph.py
│
├── vectorstore/
│   └── FAISS index
│
├── .env.example
├── .gitignore
├── requirements.txt
├── main.py
└── README.md

10. Configuration

The project is designed to work with **free model options**.
Example .env configuration:

LLM_PROVIDER="google"

GOOGLE_API_KEY="your_api_key_here"
GOOGLE_MODEL_NAME="gemini-1.5-flash"

EMBEDDING_PROVIDER="huggingface"

LANGCHAIN_TRACING_V2="true"
LANGCHAIN_API_KEY="your_langsmith_api_key"
LANGCHAIN_PROJECT="travel-policy-agent"

11. Installation & Usage
1. Install Dependencies


pip install -r requirements.txt

2. Configure Environment Variables

Create a .env file based on .env.example and add the required API keys.

3. Add the Airline Policy PDF

Place the airline policy PDF inside the `data/` directory.

4. Create the FAISS Vector Store

python -m src.ingestion

5. Start the Agent

python main.py

12. Example Queries

What is the cabin baggage limit?

Can I bring my pet on board?

What are the check-in requirements?

What is the cancellation policy?

What documents are required for international travel?

What is the baggage allowance?

13. Key Features

* RAG-based travel policy question answering.
* Modular LangGraph architecture.
* State-based workflow management.
* Query rewriting.
* FAISS similarity search.
* Context relevance validation.
* Answer evaluation.
* Conditional routing.
* Retry and revision loops.
* LangSmith observability.
* Terminal-based interaction.
* Local CPU-based embeddings.
* Free LLM / free-tier model support.

14. Results

The Travel Policy Agent successfully processes travel-related questions through a structured LangGraph workflow.

Instead of directly generating an answer, the agent performs multiple stages including **query rewriting, document retrieval, context validation, answer generation, and answer evaluation**.

The conditional routing and retry mechanism allow the workflow to revise its process when the retrieved context or generated answer does not meet the required conditions.

LangSmith provides visibility into the execution flow, making the agent easier to monitor and debug.

15. Conclusion

This project demonstrates how **LangGraph can be used to build a structured RAG-based AI Agent** rather than a simple linear chatbot.

The project provided practical experience with **StateGraph, state management, nodes, conditional edges, RAG, FAISS, retry/revision loops, answer evaluation, and LangSmith observability**.

The project was implemented using **free model options and local embeddings**, making it suitable as a practical learning and portfolio project.

 16. Future Improvements

* Add multiple airline policy documents.
* Improve query rewriting.
* Add conversation memory.
* Implement hybrid search.
* Add streaming responses.
* Improve answer evaluation.
* Add advanced LangSmith evaluations.
* Develop a web-based interface.
* Add support for multiple airlines.

 17. Author

Surabhi Suresh
AI Intern — SaaSvaap Techies Pvt. Ltd.

