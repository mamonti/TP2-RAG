import os
from dotenv import load_dotenv
import cv2
import pytesseract
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.chains import RetrievalQA

# OCR: convertir imagen a texto
def imagen_a_texto(ruta_imagen):
    img = cv2.imread(ruta_imagen)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    # Asegura que Tesseract y la carpeta del idioma están bien seteados
    pytesseract.pytesseract.tesseract_cmd = "/usr/local/bin/tesseract"  # <-- Reemplazá si usás otro path
    os.environ["TESSDATA_PREFIX"] = "/usr/local/share/tessdata"
    texto = pytesseract.image_to_string(thresh, lang='spa')
    return texto.strip()

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

    print("\n🤖 Asistente: Escribí un enunciado o pasá una imagen con un ejercicio (escribí 'salir' para terminar).")
    while True:
        user_input = input("Texto o ruta de imagen (o 'salir'): ")
        if user_input.lower() == "salir":
            break
        elif user_input.lower().endswith((".png", ".jpg", ".jpeg")) and os.path.exists(user_input):
            print("🖼️ Procesando imagen con OCR...")
            user_input = imagen_a_texto(user_input)
            print("📝 Texto extraído:", user_input)

        respuesta = qa_chain.invoke({"query": user_input})
        print("📘 Asistente:", respuesta["result"])

if __name__ == "__main__":
    main()
