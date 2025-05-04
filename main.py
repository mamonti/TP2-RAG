import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.chains import RetrievalQA

def load_documents():
    loader = DirectoryLoader("docs", glob="*.txt", loader_cls=TextLoader)
    return loader.load()

def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200, add_start_index=True)
    return splitter.split_documents(docs)

def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}  # o "cuda" si tenés GPU
    )
    return Chroma.from_documents(documents=chunks, embedding=embeddings)

def get_llm():
    return HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        task="text-generation",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        max_new_tokens=512,
        top_k=30,
        temperature=0.3,
        repetition_penalty=1.2
    )

def generate_qa_chain(vectorstore, llm):
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2}),
        verbose=False
    )

def main():
    load_dotenv()

    print("📚 Cargando documentos...")
    docs = load_documents()
    chunks = split_documents(docs)
    vectorstore = create_vectorstore(chunks)
    llm = get_llm()
    qa_chain = generate_qa_chain(vectorstore, llm)

    print("\n🤖 Asistente: ¡Hola! Escribí un enunciado de lógica o escribí 'salir' para terminar.")
    while True:
        user_input = input("Ejercicio de lógica (o 'salir'): ")
        if user_input.lower() == "salir":
            break
        respuesta = qa_chain.invoke({"query": user_input})
        print("📘 Asistente:", respuesta["result"])

if __name__ == "__main__":
    main()
