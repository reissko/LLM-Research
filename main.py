# Langchain dependencies
from langchain_community.document_loaders import PyPDFDirectoryLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama



DATA_PATH = 'materials'
CHROMA_PATH = 'chroma'

query_text = "What are some in real life applications of stacks and queues?"

PROMPT_TEMPLATE = """
You are a tutor for a Data Structures class. Answer the user's question as concisely as possible;
Preferably in 3-5 sentences.
You are given the following context to use as sources of data:
{context}

Answer the following question based on the provided context:
"{question}"
"""


def load_documents():
  docs = PyPDFDirectoryLoader(DATA_PATH).load()
  return docs

def split_documents(docs):
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    length_function=len
  )
  return text_splitter.split_documents(docs)

def create_vectorstore(chunks):
  embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
  # if the chroma database exists already then delete it before creating a new one
  """ 
  if os.path.exists(CHROMA_PATH):
    shutil.rmtree(CHROMA_PATH)
  """
  vectorstore = Chroma.from_documents(
    collection_name="cs305",
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_PATH
  )
  return vectorstore

def query_vectorstore(vectorstore, query_text):
  query_vector = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2").embed_query(query_text)
  docs = vectorstore.similarity_search_by_vector(query_vector, k=4)
  # return the first (most similar) chunk
  return docs
  
try:
  # load the textbook PDF and chunk it
  chunks = split_documents(load_documents())
  # initialize the chromadb vectorstore with the embedded chunks
  vector_store = create_vectorstore(chunks)
  # context is an array of chunks (documents) to be used as context for the LLM
  context = query_vectorstore(vector_store, query_text)
  # create the prompt template
  prompt_template = PROMPT_TEMPLATE.format(context=context, question=query_text)
  
  # create the LLM
  llm = ChatOllama(
    model="llama3.2",
    temperature=0,
    prompt_template=prompt_template,
  )

  # pass the question and context to the LLM
  response = llm.invoke(input=prompt_template)
  print(response.content)
except Exception as e:
  print(e)
