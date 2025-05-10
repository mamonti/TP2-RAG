import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from main import imagen_a_texto, create_or_load_vectorstore, get_llm, generate_qa_chain


def evaluar_samples_desempeño(qa_chain, vectorstore, carpeta="samples"):
    samples = sorted([
        f for f in os.listdir(carpeta)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    preguntas = []
    respuestas = []
    contextos = []

    for nombre in samples:
        path = os.path.join(carpeta, nombre)
        print(f"\nProcesando {nombre}...")

        # OCR
        pregunta = imagen_a_texto(path)
        if not pregunta:
            print("No se pudo extraer texto.")
            continue
        print(f"Texto extraído: {pregunta}")

        # Contexto
        docs = vectorstore.similarity_search(pregunta, k=2)
        contexto = [doc.page_content for doc in docs]

        # Respuesta del modelo
        resultado = qa_chain.invoke({"query": pregunta})
        respuesta = resultado["result"]
        print(f"Respuesta: {respuesta}")

        # Guardar en listas
        preguntas.append(pregunta)
        respuestas.append(respuesta)
        contextos.append(contexto)

    # Crear Dataset
    data = {
        "question": preguntas,
        "answer": respuestas,
        "contexts": contextos,
    }
    dataset = Dataset.from_dict(data)

    # Evaluar
    print("\nEvaluando con métricas RAGAS...")
    score = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        raise_exceptions=False
    )
    print(score.to_pandas())

if __name__ == "__main__":
    vectorstore = create_or_load_vectorstore()
    llm = get_llm()
    qa_chain = generate_qa_chain(vectorstore, llm)

    evaluar_samples_desempeño(qa_chain, vectorstore)
