
import os, re, json, openai
from PyPDF2 import PdfReader
from dotenv import load_dotenv

from openai import OpenAI
import os

load_dotenv()  # sólo en desarrollo local
openai.api_key = os.getenv("OPENAI_API_KEY")

def pdf_to_text(file) -> str:
    reader = PdfReader(file)
    return "\n".join([p.extract_text() or "" for p in reader.pages])

def chunk_text(text, max_chars=1200, overlap=300):
    chunks, start = [], 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append({"content": text[start:end]})
        start += max_chars - overlap
    return chunks

def find_relevant_chunks(question, docs_chunks, max_chunks=5):
    q_keys = set(re.findall(r'\w+', question.lower()))
    scored = []
    for c in docs_chunks:
        c_keys = set(re.findall(r'\w+', c["content"].lower()))
        scored.append((len(q_keys & c_keys), c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:max_chunks]]

def detect_district_from_text(text: str) -> str | None:
    """
    Busca patrones como "Municipalidad Distrital de X" y extrae "X".
    """
    m = re.search(
        r"Municipalidad\s+Distrital\s+de\s+([A-Za-zÁÉÍÓÚÑáéíóúñ\s]+)",
        text,
        re.IGNORECASE,
    )
    if m:
        # Normalizamos espacios y mayúsculas
        return m.group(1).strip()
    return None

def ask_openai(system_prompt: str, question: str, context_chunks: str) -> str:
    """
    Envía los mensajes al endpoint de chat de OpenAI v1+ y devuelve la respuesta.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": context_chunks + "\n\nPregunta: " + question},
    ]

    resp = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.2,
    )
    return resp.choices[0].message.content
