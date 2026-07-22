import os
import tempfile

import streamlit as st
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent  # OJO: viene de langgraph, no de langchain.agents

from herramienta_rag import (
    cargar_documentos,
    dividir_documentos,
    crear_vectorstore,
    crear_retriever,
    crear_herramienta_rag,
)

