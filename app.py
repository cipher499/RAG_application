import gradio as gr
from dotenv import load_dotenv
from graph import build_graph

load_dotenv()

rag = build_graph()

def format_history_for_rag(history: list) -> str:
    if not history:
        return ""
    formatted = []
    for msg in history[-4:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        formatted.append(f"{role}: {msg['content']}")
    return "\n".join(formatted)

def chat(message: str, history: list) -> tuple:
    if not message.strip():
        return history

    question_with_context = message
    if history:
        question_with_context = f"""Previous conversation:
{format_history_for_rag(history)}

Current question: {message}"""

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
    grounding_indicator = "🟢" if score >= 0.7 else "🟡" if score >= 0.4 else "🔴"
    full_answer = f"{answer}\n\n{grounding_indicator} Grounding score: {score:.2f}"

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": full_answer})

    return history

with gr.Blocks(title="Author Council") as demo:
    gr.Markdown("""
    # Author Council
    ### A RAG system built on the works of Carl Jung & Viktor Frankl
    Ask questions about psychology, meaning, the unconscious, and the human condition.
    """)

    chatbot = gr.Chatbot(
        label="Conversation",
        height=500,
        buttons=["copy_all"],
    )

    with gr.Row():
        msg = gr.Textbox(
            label="Your question",
            placeholder="e.g. What does Jung say about the shadow self?",
            scale=4,
            autofocus=True
        )
        submit = gr.Button("Ask", variant="primary", scale=1)

    clear = gr.Button("Clear conversation", variant="secondary")

    gr.Examples(
        examples=[
            "What is the shadow according to Jung?",
            "How does suffering give life meaning?",
            "What is individuation and why does it matter?",
            "How do Jung and Frankl think about the unconscious differently?",
            "What role does love play in finding meaning?"
        ],
        inputs=msg,
        label="Try these questions"
    )

    gr.Markdown("""
    ---
    *🟢 ≥ 0.7 well grounded · 🟡 0.4–0.7 partial · 🔴 < 0.4 treat with caution*
    """)

    submit.click(
        fn=chat,
        inputs=[msg, chatbot],
        outputs=[chatbot]
    ).then(lambda: "", outputs=msg)

    msg.submit(
        fn=chat,
        inputs=[msg, chatbot],
        outputs=[chatbot]
    ).then(lambda: "", outputs=msg)

    clear.click(lambda: [], outputs=chatbot)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())