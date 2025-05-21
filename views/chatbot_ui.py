import streamlit as st
import openai
from dotenv import load_dotenv
from utils import pdf_to_text, chunk_text, find_relevant_chunks
import os

# Carga de .env en desarrollo local
load_dotenv()

# Configura tu API key
openai.api_key = st.secrets.get("openai", {}).get("api_key") or os.getenv("OPENAI_API_KEY")

def main():
    st.header("🗂️ Chat con tu contrato de obra pública")

    distrito = st.text_input("Nombre del distrito:", "Jesús María")
    uploaded_files = st.file_uploader("Sube PDFs", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        # Procesa todos los PDFs
        chunks = []
        for pdf in uploaded_files:
            raw = pdf_to_text(pdf)
            chunks.extend(chunk_text(raw, max_chars=1200, overlap=300))
        st.session_state["chunks"] = chunks

    if "chunks" in st.session_state:
        pregunta = st.text_input("Tu pregunta:")
        if st.button("Enviar") and pregunta:
            system_prompt = (
                f"Eres un experto en obra pública en Perú. "
                f"Has cargado info de un contrato de {distrito}."
            )
            relevantes = find_relevant_chunks(pregunta, st.session_state["chunks"])
            contexto = "\n\n".join([c["content"] for c in relevantes])
            resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role":"system","content":system_prompt},
                    {"role":"user","content":contexto + "\n\nPregunta: " + pregunta}
                ]
            )
            st.markdown("**Respuesta:**")
            st.write(resp.choices[0].message.content)

if __name__ == "__main__":
    main()
