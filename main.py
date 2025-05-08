import os
from dotenv import load_dotenv
import cv2
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.chains import RetrievalQA

VECTORSTORE_DIR = "chroma_db"

# OCR: convertir imagen a texto
def imagen_a_texto(ruta_imagen):
    img = cv2.imread(ruta_imagen)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"
    os.environ["TESSDATA_PREFIX"] = "/opt/homebrew/share/tessdata"
    texto = pytesseract.image_to_string(thresh, lang='spa')
    return texto.strip()

# OCR masivo + extracción de PDFs
def ocr_desde_carpeta(carpeta="images"):
    documentos = []
    for archivo in os.listdir(carpeta):
        ruta = os.path.join(carpeta, archivo)

        if archivo.lower().endswith((".png", ".jpg", ".jpeg")):
            imagen = Image.open(ruta)
            pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"
            os.environ["TESSDATA_PREFIX"] = "/opt/homebrew/share/tessdata"
            texto = pytesseract.image_to_string(imagen, lang="spa").strip()
            if texto:
                documentos.append(Document(page_content=texto, metadata={"source": archivo}))

        elif archivo.lower().endswith(".pdf"):
            try:
                doc = fitz.open(ruta)
                texto_pdf = ""
                for page in doc:
                    texto_pdf += page.get_text()

                if texto_pdf.strip():
                    print(f"Texto embebido extraído de {archivo}")
                    documentos.append(Document(page_content=texto_pdf.strip(), metadata={"source": archivo}))
                else:
                    print(f"PDF escaneado detectado: {archivo}. Procesando con OCR...")
                    paginas = convert_from_path(ruta, dpi=300)
                    for i, pagina in enumerate(paginas):
                        texto = pytesseract.image_to_string(pagina, lang="spa").strip()
                        if texto:
                            documentos.append(Document(
                                page_content=texto,
                                metadata={"source": f"{archivo} - página {i+1}"}
                            ))
            except Exception as e:
                print(f"Error procesando {archivo}: {e}")

    return documentos

def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200, add_start_index=True)
    return splitter.split_documents(docs)

def create_or_load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}  # o "cuda" si tenés GPU
    )

    if os.path.exists(VECTORSTORE_DIR) and os.listdir(VECTORSTORE_DIR):
        print("Cargando vectorstore persistido...")
        return Chroma(persist_directory=VECTORSTORE_DIR, embedding_function=embeddings)
    else:
        print("Generando nuevo vectorstore...")
        docs = ocr_desde_carpeta("images")
        chunks = split_documents(docs)
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=VECTORSTORE_DIR
        )
        return vectorstore

def get_llm():
    return HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        task="text-generation",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        max_new_tokens=512,
        top_k=30,
        temperature=0.3,
        repetition_penalty=1.2,
        provider="auto"
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
    vectorstore = create_or_load_vectorstore()
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
