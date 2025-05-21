import os
import json
import streamlit as st
import openai
from dotenv import load_dotenv
import re  
from openai import OpenAI

# # Streamlit configuration
# with st.sidebar:
#     st.markdown(
#         """
#         <h1 style="color: white;">Interactive Chatbot</h1>
#         <h2 style="color: white;">About this chatbot</h2>
#         <p style="color: white;">
#         This is a chatbot that allows you to ask questions regarding a contract of a public work with the Jesus Maria district in Peru.
#         </p>
#         """,
#         unsafe_allow_html=True
#     )

load_dotenv()

# Define load_chunks_from_json function
def load_chunks_from_json(input_file='data/docs_chunks.json'):
    with open(input_file, 'r', encoding='utf-8') as f:
        docs_chunks = json.load(f)
    return docs_chunks

# Load the chunks from the JSON file
docs_chunks = load_chunks_from_json('data/docs_chunks.json') # Load the chunks from the JSON file

def main():
    st.header("Chat with Database 💬")

# Define the system_prompt
system_prompt = """
You are an expert in public works in Peru. You have uploaded information about a contract with the district municipality 
of Jesus Maria, in Lima, Peru. You are going to answer questions regarding the requirements of the public work, input costs,
unit prices and other relevant information about the work.  """

if "messages" not in st.session_state:
    st.session_state["messages"] = []

prompt = st.text_input("Your inquiry:", "")


openai.api_key = st.secrets['openai_key']

def find_relevant_chunks(question, docs_chunks, max_chunks=5):
    # Tokeniza la pregunta para extraer palabras clave significativas
    question_keywords = set(re.findall(r'\w+', question.lower()))
    relevance_scores = []

    # Calcula un puntaje de relevancia para cada chunk (puede ser el conteo de palabras clave coincidentes)
    for chunk in docs_chunks:
        chunk_text = chunk["content"].lower()
        chunk_keywords = set(re.findall(r'\w+', chunk_text))
        common_keywords = question_keywords.intersection(chunk_keywords)
        relevance_scores.append((len(common_keywords), chunk))

    # Ordena los chunks por su puntaje de relevancia, de mayor a menor
    relevant_chunks = [chunk for _, chunk in sorted(relevance_scores, key=lambda x: x[0], reverse=True)]

    # Retorna los top N chunks más relevantes
    return relevant_chunks[:max_chunks]

def send_question_to_openai(question, docs_chunks):
    # Encuentra los chunks más relevantes para la pregunta
    relevant_chunks = find_relevant_chunks(question, docs_chunks)
    
    # Construye el prompt completo con el system_prompt y los chunks de texto relevantes
    prompt_text = system_prompt + "\n\n" + "\n\n".join([chunk["content"] for chunk in relevant_chunks]) + "\n\nQuestion: " + question

    # Llama a la API de OpenAI con el prompt para chat
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
    )
    
    # Return the message content directly
    return response.choices[0].message.content

if st.button("Send"):
    if prompt:  # Check if the prompt is not empty
        user_message = {"role": "user", "content": prompt}
        st.session_state.messages.append(user_message)

        with st.spinner("Generating answer..."):
            response_text = send_question_to_openai(prompt, docs_chunks)
            if response_text:  # Check if the response_text is not None or empty
                assistant_message = {"role": "assistant", "content": response_text}
                st.session_state.messages.append(assistant_message)
            else:
                st.error("Failed to get a response.")  # Display an error if no response was received

# Display only the latest messages
if st.session_state.messages:
    # Get the last user message
    latest_user_message = None
    latest_assistant_message = None

    for message in reversed(st.session_state.messages):
        if message["role"] == "user" and latest_user_message is None:
            latest_user_message = message
        elif message["role"] == "assistant" and latest_assistant_message is None:
            latest_assistant_message = message
        
        if latest_user_message and latest_assistant_message:
            break
    
    # Display the last user question
    if latest_user_message:
        st.text_area("Question", value=latest_user_message["content"], height=75, disabled=True)
    
    # Display the last assistant answer
    if latest_assistant_message:
        st.text_area("Answer", value=latest_assistant_message["content"], height=100, disabled=True)

if __name__ == "__main__":
    main()