#------------------------------------------#
# This module defines the vector db and the retriever function
# It runs the semantic search between user query and vector db and returns k most relevant documents
#-----------------------------------------#

import os
import sys
import json
import time
from typing import TypedDict, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from retriever import retrieve          # the module created right before this

# load the api keys
load_dotenv()

# define the LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def print_chunks(docs: list, label: str):
    """
    prints a list of chunks with their metadata and content.
    Used for debugging retrieval and grading in the console.
    """
    print(f"\n  {'─'*55}")
    print(f"  {label} ({len(docs)} chunk{'s' if len(docs) != 1 else ''})")
    print(f"  {'─'*55}")
    if not docs:
        print("  (none)")
    for i, doc in enumerate(docs, 1):
        author = doc.metadata.get("author", "unknown")
        book   = doc.metadata.get("book",   "unknown")
        domain = doc.metadata.get("domain", "unknown")
        print(f"\n  [{i}] {author.title()} — {book.replace('_', ' ').title()}")
        print(f"      domain : {domain}")
        print(f"      content: {doc.page_content[:300].strip()}{'...' if len(doc.page_content) > 300 else ''}")
    print(f"\n  {'─'*55}")

# define the state that flows through all the nodes
class RAGState(TypedDict):
    question: str
    domain: str | None
    author: str | None
    documents: List[Document]
    filtered_documents: List[Document]
    answer: str
    grounded: bool
    hallucination_score: float
    hallucination_detail: dict

def retrieve_node(state: RAGState) -> RAGState:
    """
    Call the retrieve function created in the previous module
    and return the most relevant k chunks
    """
    print("\n[Node: retrieve]")
    
    docs = retrieve(
        query=state["question"],
        domain=state.get("domain"),
        author=state.get("author"),
        k=4,
        )

    print_chunks(docs, "Retrieved chunks")
    #  update the state with the retrieved docs list
    return {**state, "documents": docs}

def grade_documents_node(state: RAGState) -> RAGState:
    """
    1.Pulls the question and retrieved documents out of the RAG state
    2.For each document chunk, asks the LLM "is this chunk relevant to the question?"
    3.Keeps only the chunks the LLM says YES to
    4.Returns the filtered list back into state
    """

    print("\n[Node: grade_documents]")
    question = state["question"]
    documents = state["documents"]

    grading_prompt = """You are grading whether a document chunk is relevant to a question.
                    Answer only YES or NO. No explanation.

                    Question: {question}
                    Document: {document}

                    Is this document relevant to the question? (YES/NO):"""

    filtered = []

    # loop through the retrieved chunks, and evaluate the relevance to the questions
    for doc in documents:
        response = llm.invoke([
            HumanMessage(content=grading_prompt.format(
                question=question,
                document=doc.page_content[:500]
            ))
        ])

        verdict = "YES" if "YES" in response.content.upper() else "NO"
        print(f"  chunk {documents.index(doc)+1}: {verdict}  — {doc.metadata.get('author','?').title()} / {doc.metadata.get('book','?')[:40]}")
        if "YES" in verdict:
            filtered.append(doc)

    print_chunks(filtered, "Kept after grading")
    # update the state with the list of filtered documents
    return {**state, "filtered_documents": filtered}

def generate_node(state: RAGState) -> RAGState:
    """
    Sections the relevant chunks by author-book to create a context 
    Generates an answer to the asked question using this context
    """
    print("\n[Node: generate]")
    question = state["question"]
    docs = state["filtered_documents"]

    # handle the case where no relevant chunks could be found
    if not docs:
        return {**state, "answer": "I could not find relevant passages in the corpus to answer this question."}

    # create a context by sectioning the relevant chunks by the author and the book name
    context = "\n\n---\n\n".join([
        f"[{doc.metadata.get('author', 'unknown').title()} — {doc.metadata.get('book', 'unknown')}]\n{doc.page_content}"
        for doc in docs
        ])

    system_prompt = """You are a thoughtful guide with deep knowledge of the authors in this corpus.
    Answer the user's question using ONLY the provided passages.
    Always cite which author and book you are drawing from.
    If the passages do not contain enough information, say so honestly."""

    # provide the instructions, context, and question to the llm 
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Passages:\n{context}\n\nQuestion: {question}")
    ])

    # print the generated answer
    # update the state with the answer
    print(f"  Generated answer ({len(response.content)} chars)")
    return {**state, "answer": response.content}

def check_hallucination_node(state: RAGState) -> RAGState:
    print("\n[Node: check_hallucination]")
    answer = state["answer"]
    docs = state["filtered_documents"]

    if not docs:
        return {**state, "grounded": False, "hallucination_score": -1.0}

    context = "\n\n".join([
        f"[{doc.metadata.get('author')}]: {doc.page_content[:400]}"
        for doc in docs
    ])

    check_prompt = """You are evaluating whether an answer is faithfully grounded in source passages.

    For each claim in the answer, check if it is directly supported by the source passages.
    Return a JSON object with this exact structure:
    {{
    "score": <float between 0.0 and 1.0>,
    "supported_claims": <number of claims supported by sources>,
    "total_claims": <total number of claims in the answer>,
    "reasoning": "<one sentence explanation>"
    }}

    Score guide:
    1.0 = every claim is directly supported
    0.7 = most claims supported, minor extrapolation
    0.4 = some claims supported, significant extrapolation
    0.0 = answer is not grounded in the sources at all

    Source passages:
    {context}

    Answer to evaluate:
    {answer}

    Return only the JSON object, nothing else."""

    response = llm.invoke([
        HumanMessage(content=check_prompt.format(
            context=context,
            answer=answer
        ))
    ])

    try:
        result = json.loads(response.content.strip())
        score = float(result.get("score", 0.0))
        print(f"  Hallucination score : {score}")
        print(f"  Claims supported    : {result.get('supported_claims')}/{result.get('total_claims')}")
        print(f"  Reasoning           : {result.get('reasoning')}")

    except json.JSONDecodeError:
        print("  Could not parse scoring response — defaulting to 0.0")
        score = 0.0
        result = {}

    return {
        **state,
        "grounded": score >= 0.7,
        "hallucination_score": score,
        "hallucination_detail": result
    }

# define the graph, add all the nodes and edges, compile it
def build_graph():
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade_documents", grade_documents_node)
    graph.add_node("generate", generate_node)
    graph.add_node("check_hallucination", check_hallucination_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade_documents")
    graph.add_edge("grade_documents", "generate")
    graph.add_edge("generate", "check_hallucination")
    graph.add_edge("check_hallucination", END)

    return graph.compile()

if __name__ == "__main__":

    # create an instance of the graph
    rag = build_graph()

    # define initial state parameters
    questions = [
        {
            "question": "How does suffering give life meaning?",
            "author": "jung",
            "domain": "psychology"
        },
        {
            "question": "What is the relationship between man and god?",
            "author": None,
            "domain": "psychology"
        }
    ]

    # loop through the questions
    for q in questions:
        print(f"\n{'='*60}")
        print(f"Question: {q['question']}")
        print('='*60)

        # invoke the graph by providing it with question, domain, and author
        result = rag.invoke({
            "question": q["question"],
            "domain": q["domain"],
            "author": q["author"],
            "documents": [],
            "filtered_documents": [],
            "answer": "",
            "grounded": False
        })

        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nHallucination score : {result.get('hallucination_score', 'n/a')}")
        print(f"\nGrounded: {result['grounded']}")