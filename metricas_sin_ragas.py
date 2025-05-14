import os
from datasets import Dataset
from main import imagen_a_texto, create_or_load_vectorstore, get_llm, generate_qa_chain
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

def cosine_sim(a, b):
    return cosine_similarity([a], [b])[0][0]

def evaluar_metricas_custom(dataset, embeddings_model):
    resultados = []
    for row in dataset:
        q = row["question"]
        a = row["answer"]
        contexts = row["contexts"]

        joined_context = " ".join(contexts)

        q_emb = embeddings_model.encode(q)
        a_emb = embeddings_model.encode(a)
        ctx_emb = embeddings_model.encode(joined_context)

        # Métricas simples con similitud coseno
        faithfulness = cosine_sim(a_emb, ctx_emb)
        answer_relevancy = cosine_sim(a_emb, q_emb)
        context_precision = cosine_sim(ctx_emb, a_emb)
        context_recall = cosine_sim(ctx_emb, q_emb)

        resultados.append({
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
        })

    # Promedios
    promedios = {
        k: np.mean([r[k] for r in resultados])
        for k in resultados[0]
    }

    return promedios

def evaluar_samples_desempeno_doble(qa_chain, vectorstore, carpeta="samples"):
    samples = sorted([
        f for f in os.listdir(carpeta)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    preguntas = []
    teoricas = []
    resoluciones = []
    contextos = []

    for nombre in samples:
        path = os.path.join(carpeta, nombre)
        print(f"\nProcesando {nombre}...")

        pregunta = imagen_a_texto(path)
        if not pregunta:
            print("No se pudo extraer texto.")
            continue
        print(f"Texto extraído: {pregunta}")

        docs = vectorstore.similarity_search(pregunta, k=2)
        contexto = [doc.page_content for doc in docs]

        prompt_teorico = (
            f"Dado el siguiente ejercicio, ¿qué propiedades, definiciones o teoremas se podrían usar para resolverlo? "
            f"No des la resolución. Ejercicio: {pregunta}. Solo responde en español."
        )
        r_teorica = qa_chain.invoke({"query": prompt_teorico})["result"]
        print(f"Teoría sugerida: {r_teorica}")

        prompt_completo = f"{pregunta}. Solo responde en español."
        r_completo = qa_chain.invoke({"query": prompt_completo})["result"]
        print(f"Resolución: {r_completo}")

        preguntas.append(pregunta)
        teoricas.append(r_teorica)
        resoluciones.append(r_completo)
        contextos.append(contexto)

    dataset_teorico = Dataset.from_dict({
        "question": preguntas,
        "answer": teoricas,
        "contexts": contextos,
    })

    dataset_completo = Dataset.from_dict({
        "question": preguntas,
        "answer": resoluciones,
        "contexts": contextos,
    })

    print("\nEvaluando sugerencias teóricas...")
    emb_model = SentenceTransformer("all-MiniLM-L6-v2")
    score_teorico = evaluar_metricas_custom(dataset_teorico, emb_model)
    print(score_teorico)

    print("\nEvaluando resoluciones completas...")
    score_completo = evaluar_metricas_custom(dataset_completo, emb_model)
    print(score_completo)

if __name__ == "__main__":
    vectorstore = create_or_load_vectorstore()
    llm = get_llm()
    qa_chain = generate_qa_chain(vectorstore, llm)

    evaluar_samples_desempeno_doble(qa_chain, vectorstore)
    del llm
