import argparse
from dataclasses import dataclass
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.chat_models import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import os
import shutil
import utils
import time 
import newscores
from langchain_community.callbacks import get_openai_callback
from langchain_openai import OpenAI
import shutil 

experiment = {
    "name": "(Trial #2)",
    "llms": [
        {"model_name": "gpt-4", "deployment_name": "fs-gpt-4"}
    ],
    "embeddings": ["text-embedding-ada-002"],
    "k_params": [3],
    "similarity_lim": [0.3],
    "chunk_params": [
        {"size": 2000, "overlap": 1000},
    ]
}

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""
API_KEY = ********************
os.environ["AZURE_OPENAI_API_KEY"] = API_KEY
os.environ["AZURE_OPENAI_ENDPOINT"] = ********************
os.environ["AZUREOPENAI_API_TYPE"] = ********************
os.environ["OPENAI_API_VERSION"] = ********************

def generate_db(embeddings_model, chunk_size, chunk_overlap):
    CHROMA_PATH = utils.get_chroma_folder_name(embeddings_model, chunk_size, chunk_overlap)
    documents = load_documents()
    chunks = split_text(documents, chunk_size, chunk_overlap)
    save_to_chroma(chunks, CHROMA_PATH, embeddings_model)


def query_db(llm_model, llm_deployment, embeddings_model, size, overlap, k_val, cs):

    # Set smart-string names
    if (embeddings_model == "text-embedding-ada-002"):
        embedding = "ada"
    elif (embeddings_model == "text-embedding-3-large"):
        embedding = "lg"
    else:
        embedding = "sm"

    CHROMA_PATH = utils.get_chroma_folder_name(embeddings_model, size, overlap)
    start_time = time.time()

    # Prepare the DB.
    embedding_function = AzureOpenAIEmbeddings(
        openai_api_key=embeddings_model,
        api_key=API_KEY)   
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # counter = query #
    counter = 0

    # callback() -> Track Token Usage
    with get_openai_callback() as cb:
        # Parse "prompts.txt" file for input prompts
        with open("prompts.txt", "r") as file:
            prompts = file.read().splitlines()        
            for question in prompts:
                counter += 1
                # Search the DB.
                if len(question) == 0:
                    continue
                results = db.similarity_search_with_relevance_scores(question, k=k_val)
                
                # Number of documents found 
                num_docs = len(results)

                context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
                prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
                prompt = prompt_template.format(context=context_text, question=question)
                print(prompt)

                model = AzureChatOpenAI(
                    model=llm_model,
                    azure_deployment=llm_deployment
                )
                modelNum=llm_model
                response_text = model.predict(prompt)

                sources = [doc.metadata.get("source", None) for doc, _score in results]
                formatted_response = f"Response: {response_text}\nSources: {sources}"
                if len(results) == 0 or results[0][1] < cs:
                    print(f"Unable to find matching results.")
                else:
                    print(formatted_response)

                # Calculate Latency
                end_time = time.time()
                elapsed_time = end_time - start_time

                # Tokens usage 
                prompt_tokens = cb.prompt_tokens
                completion_tokens = cb.completion_tokens

                # Save output to a file path
                output_directory = "completions"

                k_directory = f"k-{k_val}.cs{cs}"
                output_path = os.path.join(output_directory, k_directory)
                os.makedirs(output_path, exist_ok=True)

                batch = f"llm-{llm_model}.te-{embedding}.cs-{size}.co-{overlap}.prompt-{counter}"
                output_file = os.path.join(output_path, batch)
                with open(output_file, "a") as f:
                    final = f"<<<Query>>>\n{question}\n<<<Prompt>>>\n{prompt}\n<<<Response>>>\n{formatted_response}\n\n<<<Time Taken>>>{elapsed_time}\n<<<>>>\n\n"
                    f.write(final)
                    # Make call to gpt-4 that returns array containing (Accuracy and Quality)
                    quality = newscores.get_scores(db, final, size, overlap)
                    values = quality.split(",")
                    accuracy = values[0]
                    quality = values[1]
                    write_table(counter, llm_model, embeddings_model, size, overlap, k_val, cs, batch, accuracy, quality, elapsed_time, num_docs, prompt_tokens, completion_tokens)

def write_table(queries, llm_model, embeddings_model, size, overlap, k_val, cs, batch, accuracy, quality, latency, num_docs, prompt_tokens, completion_tokens):
    with open("table.csv", "a") as f:
        print(f"{queries},{llm_model},{embeddings_model},{size},{overlap},{k_val},{cs},{batch},{accuracy},{quality},{latency},{num_docs},{prompt_tokens},{completion_tokens}",file=f)

# def load_documents():
#     loader = DirectoryLoader("mes-wiki/data", glob="*.md")
#     documents = loader.load()
#     return documents

def load_documents():  
    # "*.md", "*.txt", "*.html", "*.pdf", "*.docx", 
    file_extensions = ["*.md"]   
    documents = [] 
    for ext in file_extensions:  
        loader = DirectoryLoader("mes-wiki/data", glob=ext)  
        documents.extend(loader.load())  
    return documents  

def split_text(documents: list[Document], size, overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks

def save_to_chroma(chunks: list[Document], CHROMA_PATH, EMBEDDINGS_MODEL_NAME):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    embeddings_model = AzureOpenAIEmbeddings(
        model=EMBEDDINGS_MODEL_NAME,
        azure_deployment=EMBEDDINGS_MODEL_NAME,
        api_key=API_KEY
    )
    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks,
        embeddings_model,
        persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")    

def main():
    if os.path.exists("completions"):  
        print("Completion folder already exists. Program will now exit.")
        return
    with open("table.csv", "w") as f:      
        f.write("no_query,llm,embedding_model,chunk_size,chunk_overlap,k,cosine_similarity,batch,accuracy,quality,latency_s,no_documents_found,prompt_tokens,completion_tokens\n")
        f.flush()
        for embedding_name in experiment["embeddings"]:
            for chunk_dict in experiment["chunk_params"]:
                chunk_size = chunk_dict["size"]
                chunk_overlap = chunk_dict["overlap"]
                generate_db(embedding_name, chunk_size, chunk_overlap)
                for k in experiment["k_params"]:
                    for cs in experiment["similarity_lim"]:
                        for llm_dict in experiment["llms"]:
                            llm_name = llm_dict["model_name"]
                            llm_deployment = llm_dict["deployment_name"]
                            query_db(llm_name, llm_deployment, embedding_name, chunk_size, chunk_overlap, k, cs)

if __name__ == "__main__":
    main()
