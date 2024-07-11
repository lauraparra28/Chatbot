from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import os
import sys
import glob
import torch
import pandas as pd
from tqdm import tqdm
 
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent_dir)
 
import functions as fn
 
def get_embeddings(chunk_size, chunk_overlap, model_name, input_path='docs/*.txt', output_path='embeddings/embeddings.xlsx'):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
 
    all_splitted_text = []
    file_names = []
 
    for file in glob.glob(input_path):
        text = fn.load_text(file)
        splitted_text = text_splitter.create_documents([text])
        all_splitted_text.extend(splitted_text)
        file_names.extend([os.path.basename(file)] * len(splitted_text))
 
    model = SentenceTransformer(model_name)
 
    embeddings_list = []
    content_list = []
    file_name_list = []
    model_name_list = []
 
    for segment, file_name in tqdm(zip(all_splitted_text, file_names), desc="Procesando segmentos"):
        embeddings = model.encode(segment.page_content)
        embeddings_list.append(embeddings)
        content_list.append(segment.page_content)
        file_name_list.append(file_name)
        model_name_list.append(model_name)
 
    embeddings_df = pd.DataFrame(embeddings_list)
    embeddings_df['segment_content'] = content_list
    embeddings_df['file_name'] = file_name_list
    embeddings_df['model_name'] = model_name_list
 
    embeddings_df.to_excel(output_path, index=False)
 
if __name__ == "__main__":
    current_dir = os.getcwd()
    get_embeddings(chunk_size=512, chunk_overlap=100, model_name='intfloat/multilingual-e5-large')
