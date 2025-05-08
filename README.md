
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

3. **Agregar tus archivos de conocimiento:**

Agregá tus archivos `.txt` con ejercicios, teoremas o resoluciones en la carpeta `docs/`.

4. **Agregar archivo `.env`:**

Debe contener tu clave de HuggingFace Hub:

```
HUGGINGFACEHUB_API_TOKEN=tu_token_aqui
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


