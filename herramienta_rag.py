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
from dotenv import load_dotenv

load_dotenv()

# 1. CONFIGURACIÓN DEL MODELO
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile",
    temperature=0
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
            print(f"Error al cargar el archivo {documento.name}: {e}")

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
        search_kwargs={"k": 4}
    )
    return retriever


# 6. PREPARAR CONTEXTO
def preparar_contexto(documentos):
    contexto = []
    for doc in documentos:
        fuente = Path(doc.metadata.get("source", "Documento desconocido")).name
        pagina = doc.metadata.get("page", 0) + 1
        contexto.append(f"DOCUMENTO: {fuente} PÁGINA: {pagina} CONTENIDO: {doc.page_content}")
    return "\n\n".join(contexto)


# 7. CREAR HERRAMIENTA RAG
def crear_herramienta_rag(retriever, llm):
    # Recibe 'retriever' (ya armado desde el pipeline: cargar -> dividir ->
    # vectorstore -> retriever) y 'llm', para que la herramienta no dependa
    # de variables globales.

    @tool
    def herramienta_respuestas_RAG(pregunta: str) -> Dict:
        """
        Utiliza esta herramienta siempre que el usuario haga preguntas sobre la
        información contenida en los documentos de la base de conocimiento.
        """
        documentos_relacionados = retriever.invoke(pregunta)

        if not documentos_relacionados:
            return {
                "respuesta": "No poseo información oficial sobre esto en mi base de conocimiento actual",
                "citaciones": [],
                "documentos_encontrados": False
            }

        contexto = preparar_contexto(documentos_relacionados)

        prompt_rag = ChatPromptTemplate.from_messages([
            ("system", """
            Eres el Asistente de Conocimiento Interno y Copiloto Técnico de Santo Pegasus Soluciones.
            Tu objetivo es brindar respuestas precisas, rápidas y estructuradas a los colaboradores.

            Debes responder única y exclusivamente utilizando la información encontrada en los documentos recuperados.

            REGLAS:
            1. No inventes información.
            2. Si la información no aparece, responde: "No poseo información oficial sobre esto..."
            3. Indica siempre Nombre del documento, Página y Sección.
            4. No expongas credenciales o secretos.
            5. Responde en el idioma del usuario.
            6. Sé profesional, claro y usa listas o viñetas.

            CONTEXTO RECUPERADO:
            {context}
            """),
            ("human", "Pregunta del empleado: {input}")
        ])

        document_chain = prompt_rag | llm | StrOutputParser()

        answer = document_chain.invoke({
            "input": pregunta,
            "context": contexto
        })

        citaciones = []
        for doc in documentos_relacionados:
            citaciones.append({
                "documento": Path(doc.metadata.get("source", "Documento desconocido")).name,
                "pagina": doc.metadata.get("page", 0) + 1
            })

        return {
            "respuesta": answer,
            "citaciones": citaciones,
            "documentos_encontrados": True
        }

    return herramienta_respuestas_RAG