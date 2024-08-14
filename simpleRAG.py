
import os
from dotenv import load_dotenv

#For chunking
from langchain.text_splitter import RecursiveCharacterTextSplitter

#For embeddings
from langchain_openai import AzureOpenAIEmbeddings
from uuid import uuid4

#FAISS vector store and retriever
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

#Chat
from openai import AzureOpenAI

#Gradio for UI
import gradio as gr

#Extract text from document
def extract_text():
    with open('hiphop.txt',"r") as file:
        p = file.read()
    return p    

#Split the text into chunks
def split_text():
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300,chunk_overlap=10)
    documents = text_splitter.create_documents([doc])
    return documents

#Combine the chunks
def chain_answer(results):
    r = ""
    for res in results:
        r += res.page_content
    return r

#Generate the search results from vector store
def generate_response(question):
    #Query the results from vector store
    results = vector_store.similarity_search(
        question,
        k=2,
    )

    #Combine the chunks
    answer = chain_answer(results)

    #Get the response from the chat client by passing the question and answer to get a meaningful result
    response = client.chat.completions.create(
        model="gpt4",
        messages=[{"role":"user","content": question},
                {"role":"system","content": "You are a helpful assistant and answer the following questions. "},
                {"role":"system","content": "Answer the question according to the context provided only. If you dont know the answer, just say I dont know. "},
                {"role":"system", "content": "Answer in two sentences and make it concise. "},
                {"role":"user", "content": answer}]
    )

    return response.choices[0].message.content, answer

#Render UI with gradio
def render_UI():
    with gr.Blocks() as demo:
        question = gr.Textbox(label="Enter your question")
        answer = gr.Textbox(label="Answer")
        chunks = gr.Textbox(label="Chunks")
        submit_btn = gr.Button("Submit")
        submit_btn.click(fn=generate_response, inputs=question, outputs=[answer, chunks], api_name="")

        demo.launch()

#Main
if __name__ == '__main__':

    load_dotenv()

    #Create embedding model
    embeddings = AzureOpenAIEmbeddings(api_key=os.getenv("AZR_KEY"),
                                    api_version=os.getenv("AZR_VERSION"),
                                    azure_deployment=os.getenv("AZR_EMBEDDING_MODEL"),
                                    azure_endpoint=os.getenv("AZR_ENDPOINT"),
                                    model=os.getenv("AZR_EMBEDDING_MODEL"))

    #ceating index for storing in vector store
    index = faiss.IndexFlatL2(len(embeddings.embed_query("first poc")))

    #Create vectorstore FAISS
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )

    #Openai chat client
    client = AzureOpenAI(
        api_key=os.getenv("AZR_KEY"),
        api_version=os.getenv("AZR_VERSION"),
        azure_endpoint=os.getenv("AZR_ENDPOINT")
    )

    #Read and create vector chunks as documents
    doc = extract_text()
    docs = split_text()

    #Create uuids to store in vector store
    uuids = [str(uuid4()) for _ in range(len(docs))]

    #Add the documents in vector store
    vector_store.add_documents(documents=docs, ids=uuids)

    #Lanch the UI
    render_UI()
