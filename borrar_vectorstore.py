import shutil
import os

VECTORSTORE_DIR = "chroma_db"

def borrar_vectorstore():
    if os.path.exists(VECTORSTORE_DIR):
        confirmacion = input(f"Esto eliminará completamente '{VECTORSTORE_DIR}'. ¿Seguro? (y/n): ")
        if confirmacion.lower() == 'y':
            shutil.rmtree(VECTORSTORE_DIR)
            print("Vectorstore eliminado.")
        else:
            print("Operación cancelada.")
    else:
        print("No existe ningún vectorstore a eliminar.")

if __name__ == "__main__":
    borrar_vectorstore()
