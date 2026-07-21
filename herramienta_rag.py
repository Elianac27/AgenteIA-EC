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

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
