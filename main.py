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
from langchain_community.llms import LlamaCpp

VECTORSTORE_DIR = "chroma_db"

# OCR: convertir imagen a texto
def imagen_a_texto(ruta_imagen):
    img = cv2.imread(ruta_imagen)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"
    os.environ["TESSDATA_PREFIX"] = "/opt/homebrew/share/tessdata"
    texto = pytesseract.image_to_string(thresh, lang='spa')
    return texto.replace("\n", " ").strip()

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

        elif archivo.lower().endswith(".txt"):
            try:
                with open(ruta, "r", encoding="utf-8") as f:
                    texto_txt = f.read().strip()
                    if texto_txt:
                        print(f"Texto extraído de {archivo}")
                        documentos.append(Document(page_content=texto_txt, metadata={"source": archivo}))
            except Exception as e:
                print(f"Error leyendo {archivo}: {e}")

    return documentos

def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, add_start_index=True)
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
    return LlamaCpp(
    model_path="/Users/florenciamonti/Documents/GitHub/models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",  # AJUSTÁ ESTA RUTA
    temperature=0.7,
    max_tokens=256,
    n_ctx=2048,
    verbose=False,
    n_gpu_layers=0,
)

def generate_qa_chain(vectorstore, llm):
    return RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2})
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
        user_input = user_input.strip().strip('"').strip("'")
        if user_input.lower() == "salir":
            break
        elif user_input.lower().endswith((".png", ".jpg", ".jpeg")) and os.path.exists(user_input):
            print("🖼️ Procesando imagen con OCR...")
            user_input = imagen_a_texto(user_input)
            print("📝 Texto extraído:", user_input)

        # Paso 1: obtener sugerencias teóricas
        sugerencia_query = (
            f"Dado el siguiente ejercicio, ¿qué propiedades, definiciones o teoremas se podrían usar para resolverlo? "
            f"No des la resolución. Ejercicio: {user_input}. Solo responde en español."
        )
        sugerencias = qa_chain.invoke({"query": sugerencia_query})["result"]

        print("📘 Asistente - Sugerencias teóricas:", sugerencias)

        # Paso 2: ofrecer resolución
        desea_resolver = input("¿Querés que intente resolver el ejercicio? (sí/no): ").strip().lower()
        if desea_resolver in ["sí", "si", "s"]:
            respuesta = qa_chain.invoke({"query": user_input + " Solo responde en español."})["result"]
            print("📘 Asistente - Resolución:", respuesta)

if __name__ == "__main__":
    main()
