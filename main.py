import os
import json
import tempfile
import streamlit as st
import shutil 
import glob

from utils import (
    pdf_to_text,
    chunk_text,
    find_relevant_chunks,
    detect_district_from_text,
    ask_openai,
)
from tools import convert_pdf_to_json, convert_pdf_to_images, convert_files_to_text, create_json_per_folder

HERE = os.path.dirname(__file__)
JSON_DIR = os.path.join(HERE, "json") 

# ── Limpiar todos los JSON al iniciar el programa ──
if os.path.isdir(JSON_DIR):
    for json_file in glob.glob(os.path.join(JSON_DIR, "*.json")):
        try:
            os.remove(json_file)
        except Exception as e:
            print(f"No pude borrar {json_file}: {e}")
else:
    os.makedirs(JSON_DIR, exist_ok=True)
# ───────────────────────────────────────────────────

st.set_page_config(page_title="Chat con Contratos de Obras Públicas", layout="wide")
st.sidebar.header("🔧 ¿Cómo funciona?")
st.sidebar.markdown(
    "Sube uno o varios PDFs de contratos de obra pública de **cualquier** distrito de Perú para empezar a chatear con"
    "el chatbot."
)

def process_pdf_file(uploaded_file):
    """Guarda el UploadedFile en disco y devuelve su path."""
    tmp_dir = tempfile.mkdtemp()
    path = os.path.join(tmp_dir, uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.read())
    return path

def main():
    st.title("💬 Chatea con un contrato público")

    # 1) Subida de PDFs
    uploaded = st.file_uploader("Sube PDFs", type="pdf", accept_multiple_files=True)
    if not uploaded:
        st.info("Primero sube al menos un PDF para empezar.")
        return

    all_text = []
    chunks   = []

    for pdf in uploaded:
        pdf_path = process_pdf_file(pdf)
        # 2) Intentamos extraer texto
        raw_text = pdf_to_text(pdf_path)
        if len(raw_text.strip()) < 50:
            #  → PDF probablemente escaneado: convertir a JSON con OCR
            st.warning(f"🖨️ El archivo **{pdf.name}** parece escaneado. Aplicando OCR...")
            status = st.empty()

            # Paso 1: convertir a imágenes
            status.text("1/3: Convirtiendo PDF a imágenes…")
            output_img_base = "img/"
            convert_pdf_to_images(pdf_path, output_img_base)

            # Paso 2: OCR de las imágenes
            status.text("2/3: Aplicando OCR a imágenes…")
            img_folder = os.path.join(
                output_img_base,
                os.path.splitext(os.path.basename(pdf_path))[0]
            )
            output_txt_base = "txt/"
            convert_files_to_text(img_folder, output_txt_base, lang="spa")

            # Paso 3: generar JSON
            status.text("3/3: Creando JSON…")
            output_json_base = "json/"
            create_json_per_folder(output_txt_base, output_json_base, chunk_size=1600)

            # Éxito
            status.success("¡Proceso OCR completado!")

            stem = os.path.splitext(pdf.name)[0]
            this_json = os.path.join(JSON_DIR, f"{stem}.json")

            # <-- Aquí añadimos el st.write para indicar qué JSON leemos -->
            st.write(f"🔄 Leyendo chunks desde el JSON correspondiente a **{pdf.name}**:")
            st.write(f"    {os.path.basename(this_json)}")

            with open(this_json, "r", encoding="utf-8") as jf:
                ocr_chunks = json.load(jf)
            chunks.extend(ocr_chunks)
            all_text.append("")

            # json_path = os.path.join(
            #     output_json_base,
            #     os.path.splitext(os.path.basename(pdf_path))[0] + ".json"
            # )
            # with open(json_path, "r", encoding="utf-8") as jf:
            #     ocr_chunks = json.load(jf)
            # chunks.extend(ocr_chunks)
            # all_text.append("")

        else:
            #  → PDF con texto: chunkear directamente
            all_text.append(raw_text)
            chunks.extend(chunk_text(raw_text))

        os.makedirs(JSON_DIR, exist_ok=True)
        all_chunks = []
        for fname in os.listdir(JSON_DIR):
            if fname.lower().endswith(".json"):
                json_path = os.path.join(JSON_DIR, fname)
                with open(json_path, "r", encoding="utf-8") as jf:
                    all_chunks.extend(json.load(jf))

        # Unimos los chunks de texto puro + los de OCR
        chunks.extend(all_chunks)

        # Guardamos en sesión
        st.session_state["chunks"] = chunks

        # 3) Detectar distrito a partir de todo el texto concatenado
        full_text = "\n\n".join(filter(None, all_text))
        detected = detect_district_from_text(full_text)
        st.session_state["detected_district"] = detected or "el distrito correspondiente"

        st.success(f"📄 Generados {len(chunks)} chunks.")
        st.info(f"🔍 Distrito detectado: **{st.session_state['detected_district']}**")

    # 4) UI de conversación
    if "chunks" in st.session_state:
        pregunta = st.text_input("Tu pregunta:")
        if st.button("Enviar") and pregunta:
            distrito = st.session_state["detected_district"]
            system_prompt = (
                f"Eres un experto en obra pública en Perú. "
                f"Estás analizando un contrato de la municipalidad distrital de {distrito}. "
                "Responde preguntas sobre requisitos, costos, precios unitarios y otros datos relevantes."
            )

            relevantes = find_relevant_chunks(pregunta, st.session_state["chunks"])
            contexto = "\n\n".join([c["content"] for c in relevantes])

            with st.spinner("Generando respuesta…"):
                respuesta = ask_openai(system_prompt, pregunta, contexto)

            st.markdown("**Respuesta:**")
            st.write(respuesta)
    else:
        st.info("Primero sube al menos un PDF para empezar.")

if __name__ == "__main__":
    main()
