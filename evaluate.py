import json
import os
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from graph import build_graph

load_dotenv()

EVAL_DATASET_PATH = "eval_dataset.json"
RESULTS_PATH = "eval_results.json"

def run_rag_on_question(rag, question: str) -> dict:
    result = rag.invoke({
        "question": question,
        "domain": None,
        "author": None,
        "documents": [],
        "filtered_documents": [],
        "answer": "",
        "grounded": False,
        "hallucination_score": 0.0,
        "hallucination_detail": {}
    })
    
    contexts = [doc.page_content for doc in result.get("filtered_documents", [])]
    answer = result.get("answer", "")
    
    return {"answer": answer, "contexts": contexts}

def evaluate_pipeline(label: str = "baseline"):
    print(f"\n=== Running RAGAS evaluation: {label} ===\n")
    
    with open(EVAL_DATASET_PATH) as f:
        eval_data = json.load(f)
    
    rag = build_graph()
    
    questions = []
    answers = []
    contexts = []
    references = []
    
    for i, item in enumerate(eval_data):
        print(f"Running question {i+1}/{len(eval_data)}: {item['question'][:60]}...")
        result = run_rag_on_question(rag, item["question"])
        
        questions.append(item["question"])
        answers.append(result["answer"])
        contexts.append(result["contexts"])
        references.append(item["reference"])
    
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "reference": references
    })
    
    langchain_llm = ChatOpenAI(model="gpt-4o-mini")
    langchain_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    ragas_llm = LangchainLLMWrapper(langchain_llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(langchain_embeddings)
    
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    for metric in metrics:
        metric.llm = ragas_llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = ragas_embeddings
    
    results = evaluate(dataset, metrics=metrics)
    scores = results.to_pandas()[
        ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    ].mean().to_dict()
    
    print(f"\n--- Results: {label} ---")
    for metric, score in scores.items():
        print(f"  {metric:<25} {score:.4f}")
    
    existing = []
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            existing = json.load(f)
    
    existing.append({"label": label, "scores": scores})
    
    with open(RESULTS_PATH, "w") as f:
        json.dump(existing, f, indent=2)
    
    print(f"\nSaved to {RESULTS_PATH}")
    return scores

def compare_results():
    if not os.path.exists(RESULTS_PATH):
        print("No results file found. Run evaluate_pipeline() first.")
        return
    
    with open(RESULTS_PATH) as f:
        results = json.load(f)
    
    print(f"\n{'Label':<30} {'Faithfulness':<16} {'Ans Relevancy':<16} {'Ctx Precision':<16} {'Ctx Recall'}")
    print("-" * 90)
    
    for r in results:
        s = r["scores"]
        print(
            f"{r['label']:<30} "
            f"{s.get('faithfulness', 0):<16.4f} "
            f"{s.get('answer_relevancy', 0):<16.4f} "
            f"{s.get('context_precision', 0):<16.4f} "
            f"{s.get('context_recall', 0):.4f}"
        )

if __name__ == "__main__":
    evaluate_pipeline(label="baseline_no_query_rewriting")