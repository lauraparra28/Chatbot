import gradio as gr
import functions as fn
import json
import os

data = fn.load_embeddings()
num_documents = data['num_documents']
num_segment_contents = data['num_segment_contents']

with open("gradio.json", encoding='utf-8') as f:
    config = json.load(f)
    config['description'] = config['description'].format(num_documents=num_documents, num_segment_contents=num_segment_contents)

def on_submit(query):
    response = fn.rag_response(query, data=data, detailed_response=False)
    return response.replace("\n", "<br>")

def load_evaluations():
    if os.path.exists(evaluations_file):
        with open(evaluations_file, encoding='utf-8') as file:
           return json.load(file)
    return []
    
def save_evaluation(evaluations):
    with open(evaluations_file, "w", encoding='utf-8') as f:
        json.dump(evaluations, f, ensure_ascii=False, indent=4)


def evaluate_answer(question, answer, rating, feedback):
    evaluation = {
        "question": question,
        "answer": answer,
        "rating": rating,
        "feedback": feedback
    }
    evaluations.append(evaluation)
    save_evaluation(evaluations)
    return f"A avalição numérica dada é {rating} e foi dado o seguinte feedback: {feedback}"

evaluations_file = "evaluations.json"
evaluations = load_evaluations()

# # # # # # # # # # # # 
##   Chat Interface  ##
# # # # # # # # # # # # 

with gr.Blocks() as demo:
    gr.Markdown(f"## {config.get('title')}")
    gr.Markdown(config["description"])
    query_input = gr.Textbox(label="Escreva a sua pergunta")

    with gr.Row():
        submit_btn = gr.Button("Enviar")
        clear_btn = gr.ClearButton(components=[query_input], value="Limpar")
        
    gr.Examples(examples=config["examples"], inputs=query_input)
    answer_output = gr.HTML(label="Resposta")
      
    submit_btn.click(fn=on_submit, inputs=query_input, outputs=answer_output)

    with gr.Column() as eval_section:
        gr.Markdown("### Avaliação a resposta")
        
        rating = gr.Slider(1, 5, step=0.5, label="Pontuação Avaliação")
        feedback = gr.Textbox(lines=3, label="Feedback", placeholder="Se você tem alguma sugestão ou comentario, por favor, escreva aqui.")
        
        eval_btn = gr.Button("Guardar Avaliação")

        evaluation_output = gr.Textbox(label="Resumo da avaliação")

        eval_btn.click(
            fn=evaluate_answer, 
            inputs=[query_input, answer_output, rating, feedback], 
            outputs=evaluation_output
        )
   

demo.launch()

