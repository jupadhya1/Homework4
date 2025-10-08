# -*- coding: utf-8 -*-
"""week_04_hw_advanced_a2a.ipynb

EU AI Act RAG Evaluation - Advanced A2A Implementation
Using Agents as Assistants pattern and advanced evaluation library
"""

# Install required dependencies
# %pip install -U --quiet langsmith langchain langchain-openai langchain-community python-dotenv openai tiktoken pypdf langgraph

# Import necessary libraries
import os
from typing import Dict, List, Optional, Sequence
from typing_extensions import Annotated, TypedDict
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from langsmith import Client, traceable, evaluate
from langsmith.evaluation import LangChainStringEvaluator
from langsmith.schemas import Run, Example

# Load environment variables
load_dotenv()

# Set up environment
os.environ['LANGSMITH_TRACING'] = 'true'
os.environ['LANGSMITH_API_KEY'] = 'your_api_key'
os.environ['LANGSMITH_PROJECT'] = 'eu-ai-act-a2a-evaluation'
os.environ['OPENAI_API_KEY'] = 'sk-your-api-key'

client = Client()
print("✓ Environment configured successfully!")

# =============================================================================
# PART 1: DOCUMENT PROCESSING & KNOWLEDGE BASE
# =============================================================================

def load_and_process_documents(pdf_path: str = "eu_ai_act.pdf") -> InMemoryVectorStore:
    """Load EU AI Act PDF and create vector store with optimized chunking."""
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")
    
    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"✓ Loaded {len(documents)} pages from EU AI Act PDF")
    
    # Advanced chunking strategy
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
    )
    
    doc_splits = text_splitter.split_documents(documents)
    print(f"✓ Created {len(doc_splits)} optimized chunks")
    
    # Create vector store with metadata
    vectorstore = InMemoryVectorStore.from_documents(
        documents=doc_splits,
        embedding=OpenAIEmbeddings(model="text-embedding-3-small")
    )
    
    return vectorstore

# Initialize knowledge base
vectorstore = load_and_process_documents()
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)
print("✓ Knowledge base initialized!")

# =============================================================================
# PART 2: AGENTS AS ASSISTANTS (A2A) IMPLEMENTATION
# =============================================================================

# Define the agent state
class AgentState(TypedDict):
    """State for the EU AI Act compliance assistant agent."""
    messages: Sequence[BaseMessage]
    question: str
    retrieved_docs: List[Document]
    answer: str
    metadata: Dict

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Create retrieval tool
@traceable(name="retrieve_documents")
def retrieve_eu_ai_act_docs(query: str) -> str:
    """
    Retrieve relevant EU AI Act documentation based on the query.
    
    Args:
        query: The compliance question or topic to search for
        
    Returns:
        Formatted string containing relevant EU AI Act provisions
    """
    docs = retriever.invoke(query)
    
    # Format documents with metadata
    formatted_docs = []
    for i, doc in enumerate(docs, 1):
        formatted_docs.append(
            f"Document {i}:\n{doc.page_content}\n"
        )
    
    return "\n---\n".join(formatted_docs)

# Convert to LangChain tool
from langchain_core.tools import tool

@tool
def search_eu_ai_act(query: str) -> str:
    """Search the EU AI Act for relevant compliance information and legal provisions."""
    return retrieve_eu_ai_act_docs(query)

tools = [search_eu_ai_act]

# Bind tools to LLM
llm_with_tools = llm.bind_tools(tools)

# =============================================================================
# LANGGRAPH AGENT WORKFLOW
# =============================================================================

def create_eu_ai_act_agent():
    """Create an advanced A2A agent using LangGraph for EU AI Act compliance."""
    
    # Define nodes
    def agent_node(state: AgentState) -> AgentState:
        """Main reasoning node that decides whether to use tools or provide answer."""
        messages = state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": messages + [response]}
    
    def tool_node(state: AgentState) -> AgentState:
        """Execute tools and return results."""
        messages = state["messages"]
        last_message = messages[-1]
        
        # Execute tool calls
        tool_outputs = []
        for tool_call in last_message.tool_calls:
            tool_output = search_eu_ai_act.invoke(tool_call["args"])
            tool_outputs.append({
                "tool_call_id": tool_call["id"],
                "output": tool_output
            })
        
        # Create tool messages
        from langchain_core.messages import ToolMessage
        tool_messages = [
            ToolMessage(content=output["output"], tool_call_id=output["tool_call_id"])
            for output in tool_outputs
        ]
        
        return {"messages": messages + tool_messages}
    
    def should_continue(state: AgentState) -> str:
        """Determine whether to continue to tools or end."""
        messages = state["messages"]
        last_message = messages[-1]
        
        # If there are tool calls, continue to tools
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        # Otherwise end
        return "end"
    
    # Build graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    
    # Set entry point
    workflow.set_entry_point("agent")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )
    
    # Add edge from tools back to agent
    workflow.add_edge("tools", "agent")
    
    # Compile with memory
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

# Create agent
eu_ai_act_agent = create_eu_ai_act_agent()
print("✓ Advanced A2A agent created!")

# =============================================================================
# PART 3: ADVANCED EVALUATION DATASET
# =============================================================================

evaluation_dataset = [
    {
        "inputs": {
            "question": "What are the prohibited AI practices under the EU AI Act?"
        },
        "outputs": {
            "answer": "The EU AI Act prohibits certain AI practices that pose unacceptable risks to fundamental rights and safety, including AI systems that deploy subliminal techniques to manipulate behavior, exploit vulnerabilities of specific groups, enable social scoring by public authorities, perform real-time remote biometric identification in public spaces for law enforcement (with limited exceptions), and use biometric categorization systems based on sensitive characteristics."
        },
        "metadata": {
            "category": "prohibited_practices",
            "complexity": "high",
            "requires_legal_precision": True
        }
    },
    {
        "inputs": {
            "question": "What is a high-risk AI system according to the EU AI Act?"
        },
        "outputs": {
            "answer": "High-risk AI systems are those that pose significant risks to health, safety, or fundamental rights. They include AI systems used in critical infrastructure, educational or vocational training, employment, essential private and public services, law enforcement, migration and border control, and administration of justice. These systems are subject to strict requirements including risk management, data governance, technical documentation, and human oversight."
        },
        "metadata": {
            "category": "definitions",
            "complexity": "medium",
            "requires_legal_precision": True
        }
    },
    {
        "inputs": {
            "question": "What are the requirements for providers of high-risk AI systems?"
        },
        "outputs": {
            "answer": "Providers of high-risk AI systems must establish a risk management system, ensure appropriate data governance and management practices, maintain technical documentation, implement logging capabilities for traceability, provide transparency and information to deployers, ensure human oversight measures, achieve appropriate levels of accuracy, robustness and cybersecurity, and establish a quality management system. They must also register their systems in the EU database."
        },
        "metadata": {
            "category": "compliance",
            "complexity": "high",
            "requires_legal_precision": True
        }
    },
    {
        "inputs": {
            "question": "What is the conformity assessment procedure for high-risk AI systems?"
        },
        "outputs": {
            "answer": "The conformity assessment procedure involves either internal control (for most high-risk AI systems) or third-party assessment by a notified body (for specific high-risk systems like biometric identification). Providers must demonstrate compliance with requirements through technical documentation, quality management systems, and post-market monitoring. Upon successful assessment, they affix the CE marking and draw up an EU declaration of conformity."
        },
        "metadata": {
            "category": "procedures",
            "complexity": "high",
            "requires_legal_precision": True
        }
    },
    {
        "inputs": {
            "question": "What are the transparency requirements for AI systems?"
        },
        "outputs": {
            "answer": "The EU AI Act requires transparency for AI systems that interact with humans, emotion recognition systems, and biometric categorization systems. Providers must ensure users are informed they are interacting with AI. For AI-generated or manipulated content (deepfakes), there must be clear disclosure that the content was artificially generated or manipulated. General-purpose AI models must also provide technical documentation and comply with transparency obligations."
        },
        "metadata": {
            "category": "transparency",
            "complexity": "medium",
            "requires_legal_precision": True
        }
    },
    {
        "inputs": {
            "question": "What are the obligations of deployers of high-risk AI systems?"
        },
        "outputs": {
            "answer": "Deployers must use high-risk AI systems according to instructions, ensure human oversight, monitor system operation for risks, report serious incidents to providers and authorities, keep logs automatically generated by the system, and conduct a data protection impact assessment when required. They must also ensure input data is relevant and representative for the system's intended purpose."
        },
        "metadata": {
            "category": "compliance",
            "complexity": "medium",
            "requires_legal_precision": True
        }
    },
    {
        "inputs": {
            "question": "What penalties can be imposed for non-compliance with the EU AI Act?"
        },
        "outputs": {
            "answer": "The EU AI Act establishes a tiered system of administrative fines. For prohibited AI practices, fines can reach up to 35 million EUR or 7% of total worldwide annual turnover. For violations of obligations for high-risk systems, fines can be up to 15 million EUR or 3% of turnover. For supplying incorrect information, fines can reach 7.5 million EUR or 1% of turnover, whichever is higher."
        },
        "metadata": {
            "category": "enforcement",
            "complexity": "medium",
            "requires_legal_precision": True
        }
    },
    {
        "inputs": {
            "question": "What is the role of the AI Office established by the EU AI Act?"
        },
        "outputs": {
            "answer": "The AI Office, established within the European Commission, coordinates implementation and enforcement of the regulation at EU level. It supervises general-purpose AI models, maintains the EU database of high-risk AI systems, supports national authorities, promotes AI literacy and public awareness, and facilitates the development of standards and technical specifications for AI systems."
        },
        "metadata": {
            "category": "governance",
            "complexity": "medium",
            "requires_legal_precision": False
        }
    },
    {
        "inputs": {
            "question": "How does the EU AI Act address general-purpose AI models?"
        },
        "outputs": {
            "answer": "The EU AI Act introduces specific obligations for general-purpose AI models, particularly those with systemic risk (trained with compute above 10^25 FLOPs). Providers must conduct model evaluations, assess systemic risks, implement adversarial testing, report serious incidents, ensure cybersecurity, and provide technical documentation. They must also comply with copyright law regarding training data and publish summaries of content used for training."
        },
        "metadata": {
            "category": "general_purpose_ai",
            "complexity": "high",
            "requires_legal_precision": True
        }
    },
    {
        "inputs": {
            "question": "What are the regulatory sandbox provisions in the EU AI Act?"
        },
        "outputs": {
            "answer": "The EU AI Act establishes AI regulatory sandboxes that provide controlled environments for developing, testing, and validating innovative AI systems before market placement. National authorities establish and supervise these sandboxes, which allow participants to test AI under regulatory oversight with reduced compliance burdens. Sandboxes facilitate innovation while ensuring safety and fundamental rights protection, and participants receive guidance from supervisory authorities."
        },
        "metadata": {
            "category": "innovation",
            "complexity": "medium",
            "requires_legal_precision": False
        }
    }
]

# Create dataset in LangSmith
dataset_name = "eu-ai-act-a2a-advanced-eval"

try:
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Advanced EU AI Act evaluation dataset for A2A agent with comprehensive coverage"
    )
    client.create_examples(
        dataset_id=dataset.id,
        examples=evaluation_dataset
    )
    print(f"✓ Dataset '{dataset_name}' created with {len(evaluation_dataset)} examples!")
except Exception as e:
    if "already exists" in str(e):
        dataset = client.read_dataset(dataset_name=dataset_name)
        print(f"✓ Using existing dataset: {dataset.id}")
    else:
        print(f"Error: {e}")

# =============================================================================
# PART 4: ADVANCED EVALUATORS USING LANGSMITH LIBRARY
# =============================================================================

# 1. CORRECTNESS EVALUATOR (LLM-as-Judge)
correctness_prompt = """You are an expert legal analyst evaluating EU AI Act compliance responses.

Compare the STUDENT ANSWER against the REFERENCE ANSWER for factual accuracy.

Evaluation Criteria:
- Legal accuracy of EU AI Act provisions
- Correct interpretation of regulatory requirements
- No contradictions with reference answer
- Additional accurate information is acceptable

Score: 
- 1.0: Completely accurate and legally precise
- 0.7-0.9: Mostly accurate with minor omissions
- 0.4-0.6: Partially accurate with some errors
- 0.0-0.3: Largely inaccurate or misleading

REFERENCE ANSWER: {reference}
STUDENT ANSWER: {prediction}

Provide your score as a float between 0 and 1."""

correctness_evaluator = LangChainStringEvaluator(
    "labeled_score_string",
    config={
        "criteria": {
            "accuracy": "The answer is factually accurate regarding EU AI Act provisions",
            "completeness": "The answer covers the key points from the reference",
            "precision": "Legal terminology and concepts are correctly applied"
        },
        "normalize_by": 1.0,
    },
    prepare_data=lambda run, example: {
        "prediction": run.outputs["answer"],
        "reference": example.outputs["answer"],
        "input": example.inputs["question"]
    }
)

# 2. RELEVANCE EVALUATOR
relevance_prompt = """Evaluate if the answer directly addresses the EU AI Act question.

Criteria:
- Directly answers the specific question asked
- Stays focused on relevant legal provisions
- Provides actionable compliance information
- Appropriate level of detail

Score:
- 1.0: Perfectly relevant and focused
- 0.5-0.9: Mostly relevant with minor tangents
- 0.0-0.4: Off-topic or doesn't address question

QUESTION: {input}
ANSWER: {prediction}

Score (0-1):"""

relevance_evaluator = LangChainStringEvaluator(
    "score_string",
    config={
        "criteria": {
            "relevance": "The answer directly addresses the question and stays on topic"
        },
        "normalize_by": 1.0,
    },
    prepare_data=lambda run, example: {
        "prediction": run.outputs["answer"],
        "input": example.inputs["question"]
    }
)

# 3. GROUNDEDNESS EVALUATOR
def groundedness_evaluator(run: Run, example: Example) -> dict:
    """
    Evaluates if the answer is grounded in retrieved documents.
    Custom evaluator that checks for hallucinations.
    """
    # Get the answer and retrieved documents
    answer = run.outputs.get("answer", "")
    retrieved_docs = run.outputs.get("retrieved_docs", [])
    
    if not retrieved_docs:
        return {
            "key": "groundedness",
            "score": 0.0,
            "comment": "No documents retrieved"
        }
    
    # Combine retrieved content
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # Use LLM to evaluate groundedness
    grounding_prompt = f"""Evaluate if the ANSWER is fully supported by the CONTEXT from EU AI Act documents.

Check for:
- All claims supported by context
- No fabricated information
- Reasonable inferences from context
- No contradictions

CONTEXT:
{context}

ANSWER:
{answer}

Score:
- 1.0: Fully grounded, no hallucinations
- 0.5-0.9: Mostly grounded with minor unsupported claims
- 0.0-0.4: Contains significant hallucinations

Provide only a score between 0 and 1:"""
    
    result = llm.invoke([SystemMessage(content=grounding_prompt)])
    
    try:
        score = float(result.content.strip())
    except:
        score = 0.5  # Default if parsing fails
    
    return {
        "key": "groundedness",
        "score": score,
        "comment": f"Evaluated against {len(retrieved_docs)} retrieved documents"
    }

# 4. RETRIEVAL QUALITY EVALUATOR
def retrieval_quality_evaluator(run: Run, example: Example) -> dict:
    """
    Evaluates the quality and relevance of retrieved documents.
    """
    question = example.inputs.get("question", "")
    retrieved_docs = run.outputs.get("retrieved_docs", [])
    
    if not retrieved_docs:
        return {
            "key": "retrieval_quality",
            "score": 0.0,
            "comment": "No documents retrieved"
        }
    
    # Combine retrieved content
    docs_text = "\n---\n".join([f"Doc {i+1}: {doc.page_content[:300]}..." 
                                 for i, doc in enumerate(retrieved_docs)])
    
    retrieval_prompt = f"""Evaluate if the RETRIEVED DOCUMENTS are relevant to the QUESTION.

Criteria:
- Documents contain EU AI Act provisions relevant to question
- Documents provide sufficient context to answer
- Minimal irrelevant information

QUESTION:
{question}

RETRIEVED DOCUMENTS:
{docs_text}

Score:
- 1.0: Highly relevant documents
- 0.5-0.9: Mostly relevant
- 0.0-0.4: Largely irrelevant

Provide only a score between 0 and 1:"""
    
    result = llm.invoke([SystemMessage(content=retrieval_prompt)])
    
    try:
        score = float(result.content.strip())
    except:
        score = 0.5
    
    return {
        "key": "retrieval_quality",
        "score": score,
        "comment": f"Evaluated {len(retrieved_docs)} retrieved documents"
    }

# 5. COMPLETENESS EVALUATOR
def completeness_evaluator(run: Run, example: Example) -> dict:
    """
    Evaluates if the answer is comprehensive and complete.
    """
    answer = run.outputs.get("answer", "")
    reference = example.outputs.get("answer", "")
    question = example.inputs.get("question", "")
    
    completeness_prompt = f"""Evaluate the COMPLETENESS of the answer compared to the reference.

QUESTION:
{question}

REFERENCE (complete answer):
{reference}

STUDENT ANSWER:
{answer}

Score:
- 1.0: Covers all key points comprehensively
- 0.7-0.9: Covers most key points
- 0.4-0.6: Covers some key points
- 0.0-0.3: Misses most key points

Provide only a score between 0 and 1:"""
    
    result = llm.invoke([SystemMessage(content=completeness_prompt)])
    
    try:
        score = float(result.content.strip())
    except:
        score = 0.5
    
    return {
        "key": "completeness",
        "score": score,
        "comment": "Compared against reference answer"
    }

print("✓ Advanced evaluators created!")

# =============================================================================
# PART 5: TARGET FUNCTION FOR A2A AGENT
# =============================================================================

@traceable(name="eu_ai_act_assistant")
def target_function(inputs: dict) -> dict:
    """
    Target function that runs the A2A agent and returns structured outputs.
    
    Args:
        inputs: Dictionary with 'question' key
        
    Returns:
        Dictionary with 'answer' and 'retrieved_docs'
    """
    question = inputs["question"]
    
    # Create initial message
    initial_message = HumanMessage(content=f"""You are an expert EU AI Act compliance consultant. 
Answer the following question using the search_eu_ai_act tool to retrieve relevant information.

Provide a clear, accurate, and comprehensive answer based on the retrieved EU AI Act provisions.

Question: {question}""")
    
    # Run agent
    config = {"configurable": {"thread_id": f"eval_{hash(question)}"}}
    
    result = eu_ai_act_agent.invoke(
        {
            "messages": [initial_message],
            "question": question,
            "retrieved_docs": [],
            "answer": "",
            "metadata": {}
        },
        config=config
    )
    
    # Extract answer from final message
    final_message = result["messages"][-1]
    answer = final_message.content if hasattr(final_message, "content") else str(final_message)
    
    # Get retrieved documents
    retrieved_docs = retriever.invoke(question)
    
    return {
        "answer": answer,
        "retrieved_docs": retrieved_docs,
        "message_count": len(result["messages"]),
        "used_tools": any(hasattr(msg, "tool_calls") and msg.tool_calls 
                         for msg in result["messages"] if hasattr(msg, "tool_calls"))
    }

print("✓ Target function created!")

# =============================================================================
# PART 6: TEST INDIVIDUAL EVALUATORS
# =============================================================================

print("\n" + "="*80)
print("TESTING EVALUATORS")
print("="*80)

test_question = "What are the requirements for providers of high-risk AI systems?"
test_result = target_function({"question": test_question})

print(f"\nQuestion: {test_question}")
print(f"\nAnswer: {test_result['answer'][:300]}...")
print(f"\nRetrieved {len(test_result['retrieved_docs'])} documents")
print(f"Used tools: {test_result['used_tools']}")

# =============================================================================
# PART 7: RUN COMPREHENSIVE EVALUATION
# =============================================================================

print("\n" + "="*80)
print("RUNNING ADVANCED A2A EVALUATION")
print("="*80)

try:
    experiment_results = evaluate(
        target_function,
        data=dataset_name,
        evaluators=[
            correctness_evaluator,
            relevance_evaluator,
            groundedness_evaluator,
            retrieval_quality_evaluator,
            completeness_evaluator
        ],
        experiment_prefix="eu-ai-act-a2a-advanced",
        metadata={
            "version": "gpt-4o",
            "architecture": "langgraph_a2a",
            "agent_type": "react_with_tools",
            "domain": "eu_ai_act_compliance",
            "embedding_model": "text-embedding-3-small"
        },
        max_concurrency=2,
        num_repetitions=1
    )
    
    print("\n" + "="*80)
    print("✓ EVALUATION COMPLETE!")
    print("="*80)
    print(f"View results: {os.environ.get('LANGSMITH_PROJECT')}")
    
except Exception as e:
    print(f"Error during evaluation: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# PART 8: ANALYSIS AND INSIGHTS
# =============================================================================

print("\n" + "="*80)
print("ADVANCED A2A EVALUATION SUMMARY")
print("="*80)

print("""
✅ IMPLEMENTATION HIGHLIGHTS:

1. AGENTS AS ASSISTANTS (A2A) PATTERN:
   - LangGraph state machine for complex workflows
   - Tool-using agent with retrieval capabilities
   - Memory-enabled for conversation context
   - Conditional logic for tool invocation

2. ADVANCED EVALUATION FRAMEWORK:
   ✓ Correctness: LangChain labeled score evaluator
   ✓ Relevance: LangChain score string evaluator
   ✓ Groundedness: Custom LLM-based hallucination detector
   ✓ Retrieval Quality: Custom retrieval assessment
   ✓ Completeness: Custom comprehensive coverage evaluator

3. PRODUCTION-READY FEATURES:
   - Structured state management with TypedDict
   - Traceable functions for debugging
   - Comprehensive metadata tracking
   - Error handling and graceful degradation
   - Memory checkpointing for conversation continuity

4. EVALUATION METRICS:
   - 5 distinct evaluators covering all RAG dimensions
   - Normalized scores (0-1) for easy comparison
   - Custom evaluators for domain-specific needs
   - Automatic aggregation in LangSmith

📊 KEY INSIGHTS:

- A2A Pattern: Enables complex multi-step reasoning
- LangGraph: Provides explicit control flow vs. simple chains
- Custom Evaluators: Essential for domain-specific quality
- LangSmith Integration: Comprehensive tracking and analysis

🎯 NEXT STEPS:

1. Review detailed results in LangSmith UI
2. Analyze per-question performance breakdowns
3. Identify patterns in low-scoring categories
4. Iterate on agent prompts and retrieval strategy
5. Establish production quality thresholds
6. Implement continuous evaluation pipeline

💡 ADVANCED FEATURES USED:

- State graphs with conditional edges
- Tool-calling with dynamic invocation
- Memory management with checkpointing
- Multi-dimensional evaluation framework
- Custom scoring functions
- Metadata-rich experiment tracking

This advanced implementation provides enterprise-grade RAG evaluation
with sophisticated agent architecture and comprehensive quality assessment!
""")