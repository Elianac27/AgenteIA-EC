import os
from pathlib import Path
from typing import Dict
from langchain_groq import ChatGroq
from langchain_cohere import CohereEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

load_dotenv()

# 1. CONFIGURACIÓN DEL MODELO
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    api_key = GROQ_API_KEY,
    model_name = 'llama3-70b-8192',
    temperature = 0
)

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
embeddings_model = CohereEmbeddings(
        cohere_api_key=COHERE_API_KEY,
        model="embed-multilingual-v3.0"
    )


# 2. FUNCIÓN PARA CARGAR LOS PDF
def cargar_documentos(ruta_documentos: str):
    """
    Carga todos los archivos PDF encontrados en la carpeta.
    """

    docs = []
    cantidad_archivos = 0
    
    ruta = Path(ruta_documentos)
    for documento in ruta.glob("*.pdf"):
        try:
            loader = PyMuPDFLoader(str(documento))
            documentos_cargados = loader.load()
            docs.extend(documentos_cargados)
            print(f"Archivo cargado: {documento.name}")
            cantidad_archivos += 1
        except Exception as e:
            print(
                f"Error al cargar el archivo "
                f"{documento.name}: {e}"
            )

    print(f"\nTotal de archivos cargados: {cantidad_archivos}")

    return docs

# 3. DIVIDIR DOCUMENTOS EN FRAGMENTOS
def dividir_documentos(docs):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)
    print(f"Total de chunks creados: {len(chunks)}")

    return chunks


# 4. CREAR VECTORSTORE
def crear_vectorstore(chunks):

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model
    )

    return vectorstore


# 5. CREAR RETRIEVER
def crear_retriever(vectorstore):

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 4
        }
    )

    return retriever


# 6. PREPARAR CONTEXTO
def preparar_contexto(documentos):

    contexto = []

    for doc in documentos:

        fuente = Path(doc.metadata.get(
                "source", "Documento desconocido")).name

        pagina = (doc.metadata.get(
                "page",0) + 1)

        contexto.append(f"""DOCUMENTO: {fuente} PÁGINA: {pagina} CONTENIDO: {doc.page_content}""")

    return "\n\n".join(contexto)

