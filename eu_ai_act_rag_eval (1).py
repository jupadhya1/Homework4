# -*- coding: utf-8 -*-
"""week_04_hw_complete.ipynb

EU AI Act RAG Evaluation - Complete Implementation
"""

# Install required dependencies
# %pip install -U --quiet langsmith langchain langchain-openai langchain-community python-dotenv openai tiktoken pypdf requests

# Import necessary libraries
import os
import requests
from typing_extensions import Annotated, TypedDict
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langsmith import Client, traceable
from langsmith.evaluation import evaluate
from langsmith.schemas import Run, Example

# Load environment variables
load_dotenv()

# Set up LangSmith environment variables
os.environ['LANGSMITH_TRACING'] = 'true'
os.environ['LANGSMITH_API_KEY'] = 'your_api_key'
os.environ['LANGSMITH_PROJECT'] = 'eu-ai-act-rag-evaluation'
os.environ['OPENAI_API_KEY'] = 'sk-your-api-key'

# Initialize LangSmith client
client = Client()
print("✓ Environment configured successfully!")

# =============================================================================
# PART 2: EU AI Act Document Processing
# =============================================================================

def load_eu_ai_act_pdf():
    """Load the EU AI Act PDF document."""
    pdf_path = "eu_ai_act.pdf"

    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        return None

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"✓ Loaded EU AI Act PDF with {len(documents)} pages")
    return documents

def create_knowledge_base(documents):
    """Create a knowledge base from the EU AI Act PDF documents."""
    if not documents:
        print("Error: No documents provided")
        return None

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200
    )

    doc_splits = text_splitter.split_documents(documents)
    print(f"✓ Created {len(doc_splits)} document chunks")

    vectorstore = InMemoryVectorStore.from_documents(
        documents=doc_splits,
        embedding=OpenAIEmbeddings()
    )

    return vectorstore

# Load and process documents
documents = load_eu_ai_act_pdf()
if documents:
    vectorstore = create_knowledge_base(documents)
    retriever = vectorstore.as_retriever(k=4)
    print("✓ Knowledge base created successfully!")

# =============================================================================
# PART 3: RAG Implementation
# =============================================================================

llm = ChatOpenAI(model="gpt-4o", temperature=0.1)

@traceable
def rag_search(query: str) -> str:
    """Search the EU AI Act document for relevant information."""
    try:
        docs = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        return f"Relevant information from EU AI Act:\n{context}"
    except Exception as e:
        return f"Error searching documents: {str(e)}"

rag_tool = Tool(
    name="eu_ai_act_search",
    description="Search the EU AI Act document for information about AI regulations, compliance requirements, and legal provisions",
    func=rag_search
)

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

agent_executor = create_react_agent(
    model=llm, 
    tools=[rag_tool], 
    checkpointer=InMemorySaver()
)

print("✓ ReAct agent created successfully!")

# =============================================================================
# PART 4: Evaluation Dataset
# =============================================================================

evaluation_examples = [
    {
        "inputs": {"question": "What are the prohibited AI practices under the EU AI Act?"},
        "outputs": {"answer": "The EU AI Act prohibits certain AI practices that pose unacceptable risks to fundamental rights and safety, including AI systems that deploy subliminal techniques to manipulate behavior, exploit vulnerabilities of specific groups, enable social scoring by public authorities, perform real-time remote biometric identification in public spaces for law enforcement (with limited exceptions), and use biometric categorization systems based on sensitive characteristics."},
        "metadata": {"category": "prohibited_practices"}
    },
    {
        "inputs": {"question": "What is a high-risk AI system according to the EU AI Act?"},
        "outputs": {"answer": "High-risk AI systems are those that pose significant risks to health, safety, or fundamental rights. They include AI systems used in critical infrastructure, educational or vocational training, employment, essential private and public services, law enforcement, migration and border control, and administration of justice. These systems are subject to strict requirements including risk management, data governance, technical documentation, and human oversight."},
        "metadata": {"category": "definitions"}
    },
    {
        "inputs": {"question": "What are the requirements for providers of high-risk AI systems?"},
        "outputs": {"answer": "Providers of high-risk AI systems must establish a risk management system, ensure appropriate data governance and management practices, maintain technical documentation, implement logging capabilities for traceability, provide transparency and information to deployers, ensure human oversight measures, achieve appropriate levels of accuracy, robustness and cybersecurity, and establish a quality management system. They must also register their systems in the EU database."},
        "metadata": {"category": "compliance"}
    },
    {
        "inputs": {"question": "What is the conformity assessment procedure for high-risk AI systems?"},
        "outputs": {"answer": "The conformity assessment procedure involves either internal control (for most high-risk AI systems) or third-party assessment by a notified body (for specific high-risk systems like biometric identification). Providers must demonstrate compliance with requirements through technical documentation, quality management systems, and post-market monitoring. Upon successful assessment, they affix the CE marking and draw up an EU declaration of conformity."},
        "metadata": {"category": "procedures"}
    },
    {
        "inputs": {"question": "What are the transparency requirements for AI systems?"},
        "outputs": {"answer": "The EU AI Act requires transparency for AI systems that interact with humans, emotion recognition systems, and biometric categorization systems. Providers must ensure users are informed they are interacting with AI. For AI-generated or manipulated content (deepfakes), there must be clear disclosure that the content was artificially generated or manipulated. General-purpose AI models must also provide technical documentation and comply with transparency obligations."},
        "metadata": {"category": "transparency"}
    },
    {
        "inputs": {"question": "What are the obligations of deployers of high-risk AI systems?"},
        "outputs": {"answer": "Deployers must use high-risk AI systems according to instructions, ensure human oversight, monitor system operation for risks, report serious incidents to providers and authorities, keep logs automatically generated by the system, and conduct a data protection impact assessment when required. They must also ensure input data is relevant and representative for the system's intended purpose."},
        "metadata": {"category": "compliance"}
    },
    {
        "inputs": {"question": "What penalties can be imposed for non-compliance with the EU AI Act?"},
        "outputs": {"answer": "The EU AI Act establishes a tiered system of administrative fines. For prohibited AI practices, fines can reach up to 35 million EUR or 7% of total worldwide annual turnover. For violations of obligations for high-risk systems, fines can be up to 15 million EUR or 3% of turnover. For supplying incorrect information, fines can reach 7.5 million EUR or 1% of turnover, whichever is higher."},
        "metadata": {"category": "enforcement"}
    },
    {
        "inputs": {"question": "What is the role of the AI Office established by the EU AI Act?"},
        "outputs": {"answer": "The AI Office, established within the European Commission, coordinates implementation and enforcement of the regulation at EU level. It supervises general-purpose AI models, maintains the EU database of high-risk AI systems, supports national authorities, promotes AI literacy and public awareness, and facilitates the development of standards and technical specifications for AI systems."},
        "metadata": {"category": "governance"}
    }
]

print(f"✓ Created {len(evaluation_examples)} evaluation examples")

# Create dataset
dataset_name = "eu-ai-act-rag-evaluation"

try:
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="EU AI Act RAG evaluation dataset with compliance questions and reference answers"
    )
    client.create_examples(dataset_id=dataset.id, examples=evaluation_examples)
    print(f"✓ Dataset '{dataset_name}' created successfully!")
except Exception as e:
    if "already exists" in str(e):
        dataset = client.read_dataset(dataset_name=dataset_name)
        print(f"✓ Using existing dataset: {dataset.id}")
    else:
        print(f"Error creating dataset: {e}")

# =============================================================================
# PART 5: EVALUATORS IMPLEMENTATION
# =============================================================================

# 1. CORRECTNESS EVALUATOR
class CorrectnessGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    correct: Annotated[bool, ..., "True if the answer is correct, False otherwise"]

correctness_instructions = """You are an expert legal analyst evaluating the accuracy of EU AI Act compliance responses.

You will be given a QUESTION, the GROUND TRUTH (correct) ANSWER, and the STUDENT ANSWER.

Evaluation criteria:
1. Factual accuracy relative to the ground truth answer about EU AI Act provisions
2. No conflicting statements with the reference legal interpretation
3. Additional accurate legal information is acceptable if it doesn't contradict
4. Legal terminology and concepts must be correctly applied

Correctness:
- True: Student answer meets all criteria and is factually accurate regarding EU AI Act
- False: Student answer contains legal errors or conflicts with ground truth

Provide step-by-step reasoning before your final assessment."""

grader_llm = ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(
    CorrectnessGrade, method="json_schema", strict=True
)

def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    """Evaluator for RAG answer accuracy against reference answer."""
    evaluation_content = f"""QUESTION: {inputs['question']}
GROUND TRUTH ANSWER: {reference_outputs['answer']}
STUDENT ANSWER: {outputs['answer']}"""

    grade = grader_llm.invoke([
        {"role": "system", "content": correctness_instructions},
        {"role": "user", "content": evaluation_content}
    ])

    return grade["correct"]

print("✓ Correctness evaluator created!")

# 2. RELEVANCE EVALUATOR
class RelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[bool, ..., "True if the answer addresses the question"]

relevance_instructions = """You are an expert on evaluating the relevance of EU AI Act compliance responses.

You will be given a QUESTION and a STUDENT ANSWER.

Evaluation criteria:
1. Answer directly addresses the EU AI Act question asked
2. Response is helpful and informative for compliance purposes
3. The student answer is concise and focused
4. The student answer helps to answer the legal/compliance question

Relevance:
- True: Answer meets all criteria and directly addresses the EU AI Act question
- False: Answer is off-topic, unhelpful, or doesn't address the question

Provide step-by-step reasoning before your final assessment."""

relevance_llm = ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(
    RelevanceGrade, method="json_schema", strict=True
)

def relevance(inputs: dict, outputs: dict) -> bool:
    """Evaluator for RAG answer relevance to the input question."""
    evaluation_content = f"""QUESTION: {inputs['question']}
STUDENT ANSWER: {outputs['answer']}"""

    grade = relevance_llm.invoke([
        {"role": "system", "content": relevance_instructions},
        {"role": "user", "content": evaluation_content}
    ])

    return grade["relevant"]

print("✓ Relevance evaluator created!")

# 3. GROUNDEDNESS EVALUATOR
class GroundedGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    grounded: Annotated[bool, ..., "True if the answer is grounded in the facts"]

grounded_instructions = """You are an expert legal auditor evaluating the factual grounding of EU AI Act responses.

You will be given FACTS (retrieved EU AI Act documents) and a STUDENT ANSWER.

Evaluation criteria:
1. Answer is supported by the provided EU AI Act provisions
2. No information contradicts the legal facts
3. No hallucinated legal information outside the scope of facts
4. Answer draws reasonable legal conclusions from the facts

Grounded:
- True: Answer is fully supported by the EU AI Act facts and doesn't contain hallucinations
- False: Answer contains unsupported legal claims or contradicts the facts

Provide step-by-step reasoning before your final assessment."""

grounded_llm = ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(
    GroundedGrade, method="json_schema", strict=True
)

def groundedness(inputs: dict, outputs: dict) -> bool:
    """Evaluator for RAG answer groundedness in retrieved documents."""
    doc_string = "\n\n".join(doc.page_content for doc in outputs["documents"])

    evaluation_content = f"""FACTS: {doc_string}
STUDENT ANSWER: {outputs['answer']}"""

    grade = grounded_llm.invoke([
        {"role": "system", "content": grounded_instructions},
        {"role": "user", "content": evaluation_content}
    ])

    return grade["grounded"]

print("✓ Groundedness evaluator created!")

# 4. RETRIEVAL RELEVANCE EVALUATOR
class RetrievalRelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[bool, ..., "True if retrieved documents are relevant to the question"]

retrieval_relevance_instructions = """You are an expert legal information retrieval specialist evaluating EU AI Act document relevance.

You will be given a QUESTION and a set of FACTS (retrieved EU AI Act documents).

Evaluation criteria:
1. Documents contain legal provisions or information related to the question
2. Documents provide EU AI Act context that could help answer the question
3. Some irrelevant information is ACCEPTABLE if overall relevance exists
4. Focus on whether documents can contribute to answering the compliance question

Relevance:
- True: Documents contain relevant EU AI Act information that could help answer the question
- False: Documents are completely unrelated to the question

Provide step-by-step reasoning before your final assessment."""

retrieval_relevance_llm = ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(
    RetrievalRelevanceGrade, method="json_schema", strict=True
)

def retrieval_relevance(inputs: dict, outputs: dict) -> bool:
    """Evaluator for document relevance to the input question."""
    doc_string = "\n\n".join(doc.page_content for doc in outputs["documents"])

    evaluation_content = f"""QUESTION: {inputs['question']}
FACTS: {doc_string}"""

    grade = retrieval_relevance_llm.invoke([
        {"role": "system", "content": retrieval_relevance_instructions},
        {"role": "user", "content": evaluation_content}
    ])

    return grade["relevant"]

print("✓ Retrieval relevance evaluator created!")

# =============================================================================
# PART 6: TARGET FUNCTION AND EVALUATION
# =============================================================================

@traceable
def target_function(inputs: dict) -> dict:
    """
    Target function for EU AI Act RAG evaluation.
    Takes dataset inputs and returns answer with retrieved documents.
    """
    question = inputs["question"]
    
    # Use the agent to get the answer
    config = {"configurable": {"thread_id": f"eval_{hash(question)}"}}
    response = agent_executor.invoke({"messages": question}, config=config)
    answer = response["messages"][-1].content
    
    # Get the retrieved documents for evaluation
    docs = retriever.invoke(question)
    
    return {
        "answer": answer,
        "documents": docs
    }

print("✓ Target function created!")

# Test individual evaluators
print("\n" + "="*80)
print("TESTING INDIVIDUAL EVALUATORS")
print("="*80)

test_question = "What are the requirements for providers of high-risk AI systems?"
test_result = target_function({"question": test_question})

test_inputs = {"question": test_question}
test_outputs = {
    "answer": test_result["answer"], 
    "documents": test_result["documents"]
}
test_reference = {
    "answer": "Providers of high-risk AI systems must establish a risk management system, ensure appropriate data governance and management practices, maintain technical documentation, implement logging capabilities for traceability, provide transparency and information to deployers, ensure human oversight measures, achieve appropriate levels of accuracy, robustness and cybersecurity, and establish a quality management system."
}

print(f"\nTest Question: {test_question}")
print(f"\nRAG Answer: {test_result['answer'][:200]}...")

print("\n" + "-"*80)
print("Evaluator Test Results:")
print("-"*80)

try:
    correctness_score = correctness(test_inputs, test_outputs, test_reference)
    print(f"✓ Correctness: {correctness_score}")
except Exception as e:
    print(f"✗ Correctness error: {e}")

try:
    relevance_score = relevance(test_inputs, test_outputs)
    print(f"✓ Relevance: {relevance_score}")
except Exception as e:
    print(f"✗ Relevance error: {e}")

try:
    groundedness_score = groundedness(test_inputs, test_outputs)
    print(f"✓ Groundedness: {groundedness_score}")
except Exception as e:
    print(f"✗ Groundedness error: {e}")

try:
    retrieval_score = retrieval_relevance(test_inputs, test_outputs)
    print(f"✓ Retrieval Relevance: {retrieval_score}")
except Exception as e:
    print(f"✗ Retrieval relevance error: {e}")

# =============================================================================
# FULL EVALUATION
# =============================================================================

print("\n" + "="*80)
print("RUNNING COMPREHENSIVE RAG EVALUATION")
print("="*80)
print("This may take a few minutes...")

try:
    experiment_results = evaluate(
        target_function,
        data=dataset_name,
        evaluators=[correctness, relevance, groundedness, retrieval_relevance],
        experiment_prefix="eu-ai-act-rag-eval",
        metadata={
            "version": "gpt-4o",
            "model_type": "react_agent_rag",
            "domain": "eu_ai_act_compliance"
        },
        max_concurrency=2
    )
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print(f"✓ View detailed results in LangSmith UI")
    print(f"✓ Project: eu-ai-act-rag-evaluation")
    print(f"✓ Experiment prefix: eu-ai-act-rag-eval")
    
except Exception as e:
    print(f"Error running evaluation: {e}")

# =============================================================================
# PART 7: RESULTS ANALYSIS
# =============================================================================

print("\n" + "="*80)
print("KEY TAKEAWAYS")
print("="*80)

print("""
✓ Implemented comprehensive RAG evaluation for EU AI Act compliance
✓ Created 4 key evaluators:
  - Correctness: Measures factual accuracy against reference answers
  - Relevance: Ensures responses address compliance questions effectively
  - Groundedness: Prevents hallucinations by checking document support
  - Retrieval Relevance: Validates quality of document retrieval

✓ Best Practices Applied:
  - Multi-dimensional assessment for comprehensive insights
  - LLM-as-judge methodology for sophisticated evaluation
  - Domain-specific prompts tailored to EU AI Act compliance
  - Systematic testing before full evaluation

✓ Next Steps:
  1. Review evaluation results in LangSmith UI
  2. Analyze patterns in failed evaluations
  3. Identify areas for RAG system improvement
  4. Iterate on retrieval and generation strategies
  5. Establish quality thresholds for production deployment

Remember: RAG evaluation is continuous - use these insights to enhance
your EU AI Act compliance assistant over time!
""")