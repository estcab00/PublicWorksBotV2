# Libraries

from PIL import Image as PilImage
from io import BytesIO
import pytesseract
# Specify the path where Tesseract-OCR was installed
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pytesseract import Output
import re
import glob
import os
import PIL.Image
from PIL import Image
from pdf2image import convert_from_path
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError
import os
import json
import getpass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langchain.tools import tool
from langchain_openai import ChatOpenAI
from IPython.display import Image, display
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.messages import HumanMessage, SystemMessage

import openai
from openai import OpenAI
import re

from langchain_core.tools import tool
import streamlit as st


###########################
# PDF to JSON Functions
###########################

def convert_pdf_to_images(pdf_path, export_path, page_ranges=None):
    """
    Objective:
    - Converts specified pages from a PDF file into images and saves them in a specified directory.
    If no page_ranges are provided, it converts the entire PDF.

    Input:
    - pdf_path (str)    : Path to the PDF file.
    - page_ranges (str) : Pages to be converted. Can be a single page (e.g., '5'),
                          a range of pages (e.g., '5-10'), or multiple ranges 
                          (e.g., '5-10, 15-18'). Pages must be specified in an 
                          ascending numerical order. If None, converts all pages.
    - export_path (str) : Base directory path where the image folder will be created.

    Output:
    - Saves the extracted pages as images in a subfolder named after the PDF file 
      within the given export path. Each image is named according to its page number 
      (e.g., 'page_5.jpg').
    """
    print("Converting PDF to images...")
    
    # Create a directory for the output images
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_dir = os.path.join(export_path, base_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Determine the total number of pages in the PDF
    pdf_reader = PdfReader(pdf_path)
    total_pages = len(pdf_reader.pages)

    # Parse the page ranges
    pages_to_convert = []
    if page_ranges:  # Specific ranges provided
        for part in page_ranges.split(','):
            if '-' in part:
                start, end = part.split('-')
                pages_to_convert.extend(range(int(start), int(end) + 1))
            else:
                pages_to_convert.append(int(part))
    else:  # If no range is provided, convert all pages
        pages_to_convert = range(1, total_pages + 1)

    # Convert the specified pages
    images = convert_from_path(pdf_path, first_page=min(pages_to_convert), last_page=max(pages_to_convert))

    # Save the images
    for i, page in enumerate(pages_to_convert, start=1):
        if i <= len(images):
            images[i - 1].save(os.path.join(output_dir, f'page_{page}.jpg'), 'JPEG')

def convert_images_to_text( input_path, export_path, lang = 'spa' ):
    """
    Objective:
    - Converts images in a given folder to text files using pytesseract and saves them in a new subfolder 
    within the export directory. The subfolder is named after the last directory in the input path.

    Input:
    - input_path (str) :  Base folder path where the images are located. 
                          We assume that all files are valid image files
                          ( .jpg, .png or .jpeg )
    - export_path (str) : Base folder path where the text files will be saved.
    - lang (str)        : Language for pytesseract to use (default is English - 'eng').

    Output:
    - Creates a new subfolder in the export directory named after the last directory of the input path.
      For each image in the input folder, a corresponding text file is created in this subfolder.
    """
    print("Converting images to text...")

    # Get the last directory name from input_path
    last_dir_name = os.path.basename( os.path.normpath( input_path ) )

    # Create a new subdirectory in the export directory
    new_export_path = os.path.join( export_path, last_dir_name )
    if not os.path.exists( new_export_path ):
        os.makedirs( new_export_path )

    # Process each image in the input directory
    for filename in os.listdir( input_path ):
            
        # Read the image and extract text
        img_path = os.path.join( input_path, filename )
        img      = PilImage.open( img_path )
        text     = pytesseract.image_to_string( img, lang = lang )

        # Save the extracted text to a .txt file in the new subdirectory
        text_file_path = os.path.join( new_export_path, os.path.splitext( filename )[ 0 ] + '.txt' )
        with open( text_file_path, 'w', encoding = 'utf-8' ) as file:
            file.write( text )

def create_json_per_folder(input_base_path, output_base_path, chunk_size=1600):
    """
    Objective:
    - For each subfolder in the input base path, consolidate all `.txt` files into a single JSON file.
    - Save the JSON file in the output base path with the name of the folder.

    Input:
    - input_base_path (str): Path to the base folder containing subfolders with `.txt` files.
    - output_base_path (str): Path to the folder where JSON files will be saved.
    - chunk_size (int): Maximum number of characters in each chunk.

    Output:
    - Creates a JSON file for each subfolder, named `<subfolder_name>.json`.
    """
    print("Creating JSON files from text...")

    if not os.path.exists(output_base_path):
        os.makedirs(output_base_path)
    
    # Iterate through all subfolders in the input base path
    for folder_name in os.listdir(input_base_path):
        folder_path = os.path.join(input_base_path, folder_name)
        
        # Check if the item is a folder
        if os.path.isdir(folder_path):
            consolidated_chunks = []
            
            # Iterate through all .txt files in the subfolder
            for file_name in os.listdir(folder_path):
                if file_name.lower().endswith('.txt'):
                    file_path = os.path.join(folder_path, file_name)
                    
                    # Read the text file
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        text = infile.read()
                    
                    # Split the text into chunks
                    for i in range(0, len(text), chunk_size):
                        chunk = text[i:i+chunk_size]
                        consolidated_chunks.append({"content": chunk})
            
            # Save the consolidated chunks to a JSON file
            json_file_name = f"{folder_name}.json"
            output_json_path = os.path.join(output_base_path, json_file_name)
            with open(output_json_path, 'w', encoding='utf-8') as jsonfile:
                json.dump(consolidated_chunks, jsonfile, ensure_ascii=False, indent=4)

def convert_files_to_text(input_path, export_path, lang='spa'):
    """
    Objective:
    - Converts images (.jpg, .png, .jpeg) and Word files (.docx, optionally .doc) 
      in a given folder to text files, saving them to a new subfolder in 'export_path'.

    Parameters:
    - input_path (str):  Path to the folder containing images and/or Word files.
    - export_path (str): Base folder path where the text files will be saved.
    - lang (str):        Language for pytesseract to use (default is 'spa' for Spanish).

    Output:
    - Creates a new subfolder in 'export_path', named after the last directory in 'input_path'.
      For each supported file (.jpg, .png, .jpeg, .docx, .doc), a corresponding .txt file is created 
      in that subfolder with the same base name.
    """
    print("Converting files to text...")

    # Get the last directory name from input_path
    last_dir_name = os.path.basename(os.path.normpath(input_path))

    # Create a new subdirectory in the export directory
    new_export_path = os.path.join(export_path, last_dir_name)
    if not os.path.exists(new_export_path):
        os.makedirs(new_export_path)

    # Supported file extensions
    image_exts = {'.jpg', '.jpeg', '.png'}
    doc_exts   = {'.docx', '.doc'}

    # Process each file in the input directory
    for filename in os.listdir(input_path):
        file_path = os.path.join(input_path, filename)
        # Skip directories
        if os.path.isdir(file_path):
            continue

        # Identify file extension
        ext = os.path.splitext(filename)[1].lower()

        # Initialize text variable
        text = ""

        try:
            # Check if it's an image
            if ext in image_exts:
                with PilImage.open(file_path) as img:
                    text = pytesseract.image_to_string(img, lang=lang)

            # Check if it's a Word doc (.docx or .doc)
            elif ext in doc_exts:
                # Attempt to process Word files using docx2txt
                text = docx2txt.process(file_path)
                if not text:
                    text = "No text could be extracted from this file."

            # If it's not supported, skip
            else:
                continue

        except Exception as e:
            # If any error occurs, store the error message instead of text
            text = f"An error occurred while processing {filename}:\n{str(e)}"

        # Define the output .txt file path
        text_file_path = os.path.join(
            new_export_path,
            os.path.splitext(filename)[0] + '.txt'
        )
        # Write the text content to the .txt file
        with open(text_file_path, 'w', encoding='utf-8') as file:
            file.write(text)

@tool
def convert_pdf_to_json(pdf_path: str) -> str:
    """
    Convierte un PDF descargado en un archivo JSON con chunks de texto.
    Devuelve la ruta al JSON generado.
    """
    output_img_path = 'img/'
    output_txt_path = 'txt/'
    output_json_path = 'json/'

    print("1. PDF to images.")
    convert_pdf_to_images(pdf_path, output_img_path)
    
    print("2. Images to text.")
    for folder in glob.glob(f'{output_img_path}*'):
        convert_images_to_text(folder, output_txt_path, lang='spa')

    print("3. JSON files from text.")
    create_json_per_folder(output_txt_path, output_json_path, chunk_size=1600)
    
    json_filename = os.path.splitext(os.path.basename(pdf_path))[0] + ".json"
    return os.path.join(output_json_path, json_filename)


###########################
# Relevant Chunk Search Functions
###########################           


def find_relevant_chunks_tfidf_2step(question, directory_path, max_files=3, max_chunks=5):
    """
    A 2-step TF-IDF approach:
      1) Compare the question with the "summary" of each JSON file,
         select the top 'max_files' relevant files.
      2) For each selected file, compare the question with the text of each chunk
         in 'contentList', and pick the top 'max_chunks'.

    Returns:
        A dict with:
          - "relevant_files": list of file-level matches (sorted by similarity desc)
          - "relevant_chunks": list of chunk-level matches across those files
    """

    # --------------------------------------------------------------------
    # STEP 1: File-level search by comparing question vs. "summary"
    # --------------------------------------------------------------------
    files_data = []  # Will store (filename, summary_text, content_list)
    for file_name in os.listdir(directory_path):
        if not file_name.endswith('.json'):
            continue

        file_path = os.path.join(directory_path, file_name)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Each file has:
                # {
                #   "summary": "...",
                #   "contentList": [{ "content": ... }, ...]
                # }
                summary_text = data.get("summary", "").strip()
                content_list = data.get("contentList", [])
                files_data.append((file_name, summary_text, content_list))
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")

    # If no files or all empty, return empty results
    if not files_data:
        return {
            "relevant_files": [],
            "relevant_chunks": []
        }

    # Build TF-IDF for the question + all summaries
    summaries = [fd[1] for fd in files_data]  # all summary texts
    documents = [question] + summaries        # index 0 is the question
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    question_vector = tfidf_matrix[0]    # the question
    summary_vectors = tfidf_matrix[1:]   # the file summaries
    similarities = cosine_similarity(question_vector, summary_vectors).flatten()

    # Pair each file with its similarity
    file_scores = list(zip(similarities, files_data))
    # Sort by similarity descending
    file_scores.sort(key=lambda x: x[0], reverse=True)

    # Pick top N files
    top_files = file_scores[:max_files]

    # Format them for returning
    relevant_files = [
        {
            "file_name": file_info[0],
            "summary": file_info[1],
            "similarity": sim
        }
        for (sim, file_info) in top_files
    ]

    # --------------------------------------------------------------------
    # STEP 2: Within each top file, compare question vs. each chunk
    # --------------------------------------------------------------------
    all_relevant_chunks = []
    for sim_file, (file_name, summary_text, content_list) in top_files:
        # Build a new doc set: [question] + all chunk texts
        chunk_texts = [chunk.get("content", "") for chunk in content_list if chunk.get("content")]
        if not chunk_texts:
            continue  # no chunks in this file

        documents = [question] + chunk_texts
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(documents)

        question_vector = tfidf_matrix[0]
        chunk_vectors = tfidf_matrix[1:]
        chunk_sims = cosine_similarity(question_vector, chunk_vectors).flatten()

        # Pair each chunk with its similarity
        chunk_scores = list(zip(chunk_sims, content_list))
        chunk_scores.sort(key=lambda x: x[0], reverse=True)
        top_chunks = chunk_scores[:max_chunks]

        # Format chunk-level results
        for chunk_sim, chunk_data in top_chunks:
            all_relevant_chunks.append({
                "file_name": file_name,
                "similarity": chunk_sim,
                "chunk": chunk_data
            })

    # Return both file-level results and chunk-level results
    return {
        "relevant_files": relevant_files,
        "relevant_chunks": all_relevant_chunks
    }

@tool
def search_relevant_chunks_tool_new(question: str, directory_path: str, max_files: int = 3, max_chunks: int = 5) -> dict:

    """
    Searches for the most relevant JSON files in a directory based on their 'summary',
    then for each chosen file, selects the most relevant chunks from 'contentList'.

    Args:
    - question (str): The question or query to compare against each JSON file's summary and chunks.
    - directory_path (str): Path to the directory containing JSON files with structure:
            {
                "summary": "...",
                "contentList": [
                    { "content": "..." },
                    ...
                ]
            }
    - max_files (int): How many of the top relevant files to retrieve by summary-level matching.
    - max_chunks (int): How many top relevant chunks to retrieve from each file.

    Returns:
    - dict: A dictionary with two keys. For "relevant_files", the values are lists of dictionaries that contain the file name, summary, and similarity. For "relevant_chunks", the values are lists of dictionaries that contain the file name, similarity, and chunk content.
    """
    return find_relevant_chunks_tfidf_2step(question, directory_path, max_files, max_chunks)
