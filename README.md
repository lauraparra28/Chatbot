# Chatbot 
Este projeto implementa um assistente de inteligência artificial que responde perguntas baseado em documentos recuperados. O sistema utiliza recuperação de informação via similaridade de embeddings para encontrar os segmentos de texto mais relevantes e gera respostas detalhadas em português culto.

## Arquivos

- **embeddings.py**: Este arquivo lê os documentos fornecidos, divide-os em segmentos e os vetoriza.
- **functions.py**: Este arquivo contém funções utilitárias para detecção de encoding, leitura de texto, busca de segmentos usando embeddings e geração de respostas baseadas em modelos generativos.
- **app.py**: Este arquivo define a interface do usuário utilizando Gradio para interagir com o assistente de IA.
- **documents_names.json**: Arquivo que armazena os nomes com os quais serão reportados os documentos utilizados.
- **evaluations.json**: Arquivo que armazena as perguntas, respostas do chatbot, e avaliações do usuário.
- **docs/**: Pasta onde se adicionam os documentos que serão utilizados para o processo de RAG. Os documentos devem estar em formato .txt.
- **embeddings/**: Pasta onde se armazenam os embeddings resultantes da execução do arquivo embeddings.py.

## Uso

1. Criar e ativar um novo ambiente utilizando a versão de Python 3.11.9.
```bash
conda create --name chatbot python=3.11.9
conda activate chatbot

```
2. Instalar os requisitos com `pip install -r requirements.txt`.
3. Executar o arquivo `embeddings.py` para gerar os embeddings e vetorizar os documentos.
4. Adicionar os nomes com os quais serão reportados os documentos finais no arquivo `documents_names.json`. Os valores "key" correspondem ao nome do documento existente em formato .txt, e os valores "value" representam o nome com o qual se deseja reportar esses documentos.
5. Executar o arquivo `app.py` para lançar a interface do chatbot.
