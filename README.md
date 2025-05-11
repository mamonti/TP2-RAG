
# Asistente Lógico - Proyecto RAG

Este asistente permite resolver ejercicios escritos a mano de la materia *Lógica* del ITBA. 
Dado un enunciado extraído de una imagen, el sistema devuelve sugerencias, teoremas útiles, 
y eventualmente una imagen con la resolución hecha por un alumno.

---

## ▶️ Cómo usar

1. **Crear entorno virtual (opcional):**

```bash
python3 -m venv venv
source venv/bin/activate  # o venv\Scripts\activate.bat en Windows
```

2. **Instalar dependencias:**

```bash
pip install -r requirements.txt
```

3. **Colocar el modelo LLM local:**

Descargá un modelo compatible en formato .gguf, por ejemplo:
Mistral 7B Instruct (Q4_K_M)

Ubicalo en una ruta como:

```
/Users/tu_usuario/Documents/GitHub/models/mistral-7b-instruct-v0.1.Q4_K_M.gguf
```

4. **Configurar el uso del modelo local:**

Verificá que en main.py esté configurado algo como:

```
from langchain.llms import LlamaCpp

llm = LlamaCpp(
    model_path="ruta/a/tu/modelo.gguf",
    temperature=0.7,
    max_tokens=512,
    n_ctx=4096,
    verbose=False,
)

```

5. **Ejecutar el asistente:**

```bash
python main.py
```

El asistente abrirá un prompt donde podés escribir un enunciado o duda, y recibís una respuesta generada a partir de tu base de conocimientos personalizada.

---

## 📝 Ejemplo

**Con texto:**

```
Tú: ¿Cómo pruebo que ¬(p ∧ q) ↔ (¬p ∨ ¬q)?
Asistente: Podés usar la ley de De Morgan y luego demostrar la doble implicación con tablas de verdad...
```

**Con path a una imagen:**

```
Tú: path/a/tu/imagen.png
Asistente: Podés usar la ley de De Morgan y luego demostrar la doble implicación con tablas de verdad...
```

---


