from llama_index.llms.ollama import Ollama
import re
import json
import pandas as pd
import sys
import os
import numpy as np
import spacy
from heapq import nlargest
from llama_index.core import PromptTemplate
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompts.prompt_templates import *
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Union
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from utils.RAG_process import refine_supporting_facts
from utils.iterative_reasoner import process_question_with_kg_awareness
from utils.KG_builder import build_rdflib_knowledge_graph, process_contexts_in_batches
import traceback
from datetime import datetime
import ast
import argparse

# Import the new KG-Guided RAG system
from utils.kg_guided_rag import KGGuidedRAG, kg_guided_rag_generate

# Add argument parsing for different RAG approaches
parser = argparse.ArgumentParser(description='Multi-Hop Question Answering with KG')
parser.add_argument('--rag_method', type=str, default='kg_guided', 
                    choices=['standard', 'kg_guided'],
                    help='RAG method to use (standard or kg_guided)')
parser.add_argument('--model', type=str, default='llama3.3',
                    help='Model name for Ollama')
parser.add_argument('--sample_size', type=int, default=400,
                    help='Number of samples to use from dataset')
parser.add_argument('--output_dir', type=str, default='result',
                    help='Directory to save results')
args = parser.parse_args()