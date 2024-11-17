
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import gradio as gr

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') 
DOC_PATH = "./data/resume.pdf"
CHROMA_PATH = "resume_db" 

loader = PyPDFLoader(DOC_PATH)
pages = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(pages)

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

db_chroma = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)

PROMPT_TEMPLATE = """
    Answer the question based only on the following context:
    {context}
    Answer the question based on the above context: {question}.
    Provide a detailed answer.
    Don’t justify your answers.
    Don’t give information not mentioned in the CONTEXT INFORMATION.
    Do not say "according to the context" or "mentioned in the context" or similar.
    """
prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
model = ChatOpenAI()


def run_query(query):
    docs_chroma = db_chroma.similarity_search_with_score(query, k=5)

    context_text = "\n\n".join([doc.page_content for doc, _score in docs_chroma])
   
    prompt = prompt_template.format(context=context_text, question=query)

    response = model.invoke(prompt).content
    return response + '\n'

def process_response(message, history):
    return run_query(message)

gr.ChatInterface(process_response, type="messages").launch()
