import chardet
import torch
from langchain_openai import ChatOpenAI, OpenAI
from langchain_core.prompts import PromptTemplate
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
import os
import pandas as pd
import json

current_dir = os.getcwd()
api_key_file = "api_key.json"

def load_api_key():
    if not 'OPENAI_API_KEY' in os.environ:
        with open(api_key_file, 'r') as file:
            api_key = json.load(file)['api_key']
    else:
        api_key = os.environ['OPENAI_API_KEY']
    
    return api_key
    
def load_dictionary(json_path):
    with open(json_path, 'r', encoding='utf-8') as file:
        return json.load(file)
    
def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        raw_data = file.read()
    result = chardet.detect(raw_data)
    return result['encoding']

def load_text(file_path):
    encoding = detect_encoding(file_path)
    with open(file_path, 'r', encoding=encoding) as file:
        return file.read()
    
def search_query(query, embeddings_tensor, model, segment_contents, file_names, k=5):
    query_embedding = torch.tensor(model.encode(query)).unsqueeze(0)
    similarities = torch.mm(query_embedding, embeddings_tensor.t()).squeeze(0)
    topk_similarities, topk_indices = torch.topk(similarities, k)

    top_segments = [segment_contents[idx] for idx in topk_indices]
    top_file_names = [file_names[idx] for idx in topk_indices]
    top_similarities = topk_similarities.tolist()
    
    return top_segments, top_file_names, top_similarities

def load_embeddings(file_path="embeddings/embeddings.xlsx"):
    embeddings_df = pd.read_excel(os.path.join(current_dir, file_path))
    embeddings = embeddings_df.iloc[:, :-3].values
    segment_contents = embeddings_df['segment_content'].values
    num_segment_contents = len(segment_contents)
    num_documents = embeddings_df['file_name'].nunique()
    file_names = embeddings_df['file_name'].values
    model_name = embeddings_df['model_name'].values[0]

    return {
        "embeddings": embeddings,
        "segment_contents": segment_contents,
        "num_documents": num_documents, 
        "num_segment_contents": num_segment_contents,         
        "file_names": file_names,
        "model_name": model_name,
        }

def generate_answer_with_references(query, data, api_key):
    embeddings = data["embeddings"]
    segment_contents = data["segment_contents"]
    model_name = data["model_name"]
    file_names = data["file_names"]
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
    model = SentenceTransformer(model_name)
    dictionary_path = os.path.join(current_dir, 'documents_names.json')
    file_name_dict = load_dictionary(dictionary_path)
    file_names = [file_name_dict.get(name, name) for name in file_names]

    top_segments, top_file_names, top_similarities = search_query(query, embeddings_tensor, model, segment_contents, file_names, k=5)
    context = "\n----\n".join(top_segments)
    prompt_template = """
        Você é um assistente de inteligência artificial que responde a perguntas baseadas nos documentos de forma detalhada na forma culta da língua portuguesa.
        Não é possível gerar informações ou fornecer informações que não estejam contidas nos documentos recuperados. 
        Se a informação não se encontra nos documentos, responda com: Não foi possível encontrar a informação requerida nos documentos.
        
        Contexto:
        
        {context}

        Pergunta: {query}

        Resposta:""".format(context=context, query=query)
        
    qa_prompt = PromptTemplate.from_template(prompt_template)
        
    llm = ChatOpenAI(api_key=api_key, model="gpt-3.5-turbo")
    response = llm.invoke(qa_prompt.template)
    resposta = response.content
    total_tokens  = response.response_metadata['token_usage']['total_tokens']
    prompt_tokens = response.response_metadata['token_usage']['prompt_tokens']
        
    return resposta, total_tokens, prompt_tokens, top_segments, top_file_names, top_similarities, prompt_template

def rag_response(query, data, detailed_response):
    api_key = load_api_key()
    
    resposta, total_tokens, prompt_tokens, top_segments, top_file_names, top_similarities, prompt_template = generate_answer_with_references(query, data, api_key)
    file_names = [x[0] for x in top_file_names]
    file_links = {x[0]: x[1] for x in top_file_names}

    if detailed_response==True:
        references_detail = "\n\n".join([
        f"* Segmento: {segment}\nArquivo: <a href='{file_links[file_name]}' target='_blank'>{file_name}</a>\nSimilaridade: {similarity:.4f}"
        for segment, file_name, similarity in zip(top_segments, file_names, top_similarities)])
 
        formatted_detailed_response = f"Resposta:\n\n{resposta}\n\nPrompt:\n{prompt_template}\n\nPrompt Tokens: {prompt_tokens}\nTotal Tokens: {total_tokens}\n\n{references_detail}"

        return formatted_detailed_response 
    else: 
        file_set = set(file_name for file_name in file_names)
        references = "\n".join("<a href='{}' target='_blank'>{}</a>".format(file_links[file_name], file_name) for file_name in file_set)
        formatted_response = f"{resposta}\n\n----\n{references}"
        return formatted_response 
