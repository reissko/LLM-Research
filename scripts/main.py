# Langchain dependencies
from langchain_community.document_loaders import PyPDFDirectoryLoader # Importing PDF loader from Langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter # Importing text splitter from Langchain
from langchain_community.embeddings import OpenAIEmbeddings # Importing OpenAI embeddings from Langchain
from langchain.schema import Document # Importing Document schema from Langchain
from langchain_community.vectorstores import Chroma # Importing Chroma vector store from Langchain
from dotenv import load_dotenv # Importing dotenv to get API key from .env file
from langchain_community.chat_models import ChatOpenAI # Import OpenAI LLM
import os # Importing os module for operating system functionalities
import shutil # Importing shutil module for high-level file operations

# path to data
DATA_PATH = '../materials'
CHROMA_PATH = '../chroma'

def load_documents():
  # Initialize PDF loader with specified directory
  document_loader = PyPDFDirectoryLoader(DATA_PATH) 
  # Load PDF documents and return them as a list of Document objects
  return document_loader.load() 

documents = load_documents() # Call the function
# Inspect the contents of the first document as well as metadata
print(documents[0])