import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from graph import build_graph

load_dotenv()

def format_history(history: list) -> str:
    """
    Creates a history string that has the last 
    4 messages between the user and the AI
    """
    if not history:
        return ""
    formatted = []
    for msg in history[-4:]:
        if isinstance(msg, HumanMessage):
            formatted.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            formatted.append(f"Assistant: {msg.content}")
    return "\n".join(formatted)

def chat():
    print('chat started')
    rag = build_graph()
    history = []

    print("\n=== Author Council RAG ===")
    print("Authors loaded: Jung, Frankl")
    print("Type 'quit' to exit, 'clear' to reset conversation\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:  # the case where the user type nothing
            continue
        if user_input.lower() == "quit":    # when user wants to end the chat
            print("Goodbye.")
            break
        if user_input.lower() == "clear":   # when user wants to clear the chat
            history = []
            print("Conversation cleared.\n")
            continue

        question_with_context = user_input  
        if history:                         # if history exists, add the current question to it
            question_with_context = f"""Previous conversation:
                {format_history(history)}

                Current question: {user_input}"""

        result = rag.invoke({
            "question": question_with_context,
            "domain": "psychology",
            "author": None,
            "documents": [],
            "filtered_documents": [],
            "answer": "",
            "grounded": False,
            "hallucination_score": 0.0,
            "hallucination_detail": {}
        })

        answer = result["answer"]
        score = result.get("hallucination_score", 0.0)

        history.append(HumanMessage(content=user_input))
        history.append(AIMessage(content=answer))

        print(f"\nCouncil: {answer}")
        print(f"[grounding score: {score:.2f}]\n")

if __name__ == "__main__":
    chat()