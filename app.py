import os
import tempfile

import streamlit as st
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from herramienta_rag import (
    cargar_documentos,
    dividir_documentos,
    crear_vectorstore,
    crear_retriever,
    crear_herramienta_rag,
)

# 1. CONFIGURACIÓN DE PÁGINA
st.set_page_config(page_title="Asistente de Conocimiento Interno con IA", layout="centered")
st.title("🤖 Asistente de Conocimiento Interno con IA")

with st.chat_message("assistant", avatar="🤖"):
    st.markdown("**¡Hola! Soy Sara, tu asistente.** 👋 ¿En qué puedo ayudarte?")
    st.markdown(
        "Uso un agente creado con LangChain y LangGraph para consultar la "
        "información de tus documentos PDF. Sube uno o más archivos y luego "
        "hazme preguntas."
    )

# 2. CONFIGURACIÓN DE LA SESIÓN
if "agente" not in st.session_state:
    st.session_state["agente"] = None
if "agente_listo" not in st.session_state:
    st.session_state["agente_listo"] = False


# 3. CARGA DE PDF
st.markdown("### 📁 Realiza la carga de tus archivos PDF")
archivos_cargados = st.file_uploader(
    "Selecciona uno o más archivos PDF",
    type="pdf",
    accept_multiple_files=True,
    label_visibility="collapsed"
)

if archivos_cargados and not st.session_state["agente_listo"]:
    with st.spinner("Procesando los documentos y preparando al agente..."):

        carpeta_temporal = tempfile.mkdtemp()
        for archivo in archivos_cargados:
            ruta_pdf = os.path.join(carpeta_temporal, archivo.name)
            with open(ruta_pdf, "wb") as f:
                f.write(archivo.getvalue())

        documentos = cargar_documentos(carpeta_temporal)
        chunks = dividir_documentos(documentos)
        vectorstore = crear_vectorstore(chunks)
        retriever = crear_retriever(vectorstore)

        GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model_name="llama-3.3-70b-versatile",
            temperature=0
        )

        herramienta_respuestas_RAG = crear_herramienta_rag(retriever, llm)

        memory = MemorySaver()
        system_prompt = """
        Eres Sara, el Asistente de Conocimiento Interno y Copiloto Técnico de Santo Pegasus Soluciones.
        Tu función es responder preguntas basándote exclusivamente en la información oficial contenida
        en los documentos PDF disponibles en la base de conocimiento.
        Tienes disponible la siguiente herramienta:

        - herramienta_respuestas_RAG:
        Consulta la base de conocimiento de documentos PDF y recupera información relevante para responder preguntas.

        REGLAS:
        1. Siempre que el usuario pregunte sobre información relacionada con la empresa, sus procesos, onboarding,
           ingeniería, arquitectura, microservicios, operaciones SRE o cualquier contenido que pueda estar en los
           documentos PDF, utiliza la herramienta herramienta_respuestas_RAG.
        2. No inventes información que no esté respaldada por los documentos recuperados.
        3. Si la herramienta indica que no existe información oficial disponible, responde:
           "No poseo información oficial sobre esto en mi base de conocimiento actual".
        4. Responde en el mismo idioma utilizado por el usuario.
        5. Utiliza un lenguaje profesional, claro y estructurado.
        6. Cuando la información esté disponible, menciona las fuentes recuperadas indicando el nombre
           del documento y la página cuando sea posible.
        7. No solicites ni expongas credenciales, contraseñas, tokens o secretos.
        """

        agente = create_react_agent(
            model=llm,
            tools=[herramienta_respuestas_RAG],
            prompt=system_prompt,
            checkpointer=memory
        )

        st.session_state["agente"] = agente
        st.session_state["agente_listo"] = True

    st.success(f"¡{len(archivos_cargados)} archivo(s) cargado(s) y agente listo!")



# 4. ZONA DE PREGUNTAS
st.markdown("---")
st.markdown("## ⚡ Pregúntale al agente")

if st.session_state["agente_listo"]:
    pregunta = st.text_input(
        "Escribe tu pregunta",
        placeholder="¿Qué beneficios ofrece la empresa a sus colaboradores?"
    )

    if st.button("🚀 Preguntar"):
        if not pregunta.strip():
            st.warning("Escribe una pregunta.")
        else:
            with st.spinner("El agente está buscando la información..."):
                result = st.session_state["agente"].invoke(
                    {"messages": [{"role": "user", "content": pregunta}]},
                    config={"configurable": {"thread_id": "usuario_streamlit"}}
                )

                respuesta_final = result["messages"][-1]

                st.markdown("### 🤖 Respuesta")
                st.markdown(respuesta_final.content)
else:
    st.caption("Sube un PDF arriba para activar al agente.")