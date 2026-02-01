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
from utils.RAG_process import rag_generate
from utils.RAG_process import refine_supporting_facts
import ast
from utils.iterative_reasoner import process_question_with_kg_awareness
from utils.KG_builder import build_rdflib_knowledge_graph,process_contexts_in_batches
import traceback
from datetime import datetime


llm = Ollama(model="llama3.3", request_timeout= 600.0,keep_alive=-1)
response = llm.complete("What are the recurring archetypes in Bulgarian political myths?")

Hotpot_train = json.load(open('dataset/hotpot/hotpot_train_v1.1.json'))

dataset = Hotpot_train[:400]

RAG_context = rag_generate(dataset,llm)
dataset_df = pd.DataFrame(dataset)
refine_dataset = dataset_df[dataset_df['question'].isin(RAG_context['Question'])].reset_index(drop=True)
refined_df = refine_supporting_facts(RAG_context, RAG_context, refine_dataset)

def compare_facts(refined_df):
    """
    Compares if all Supporting Facts are in Retrieval Facts.
    Sets coverage_all to 1 if all are present, else 0.
    """
    df = refined_df.copy()
    
    # Function to check if all supporting facts are in retrieval facts
    def check_coverage(row):
        supporting = row['Supporting Facts']
        retrieval = row['Retrieval Facts']
        
        # Convert strings to lists if needed
        if isinstance(supporting, str):
            supporting = ast.literal_eval(supporting)
        if isinstance(retrieval, str):
            retrieval = ast.literal_eval(retrieval)
        
        # Convert lists to strings for simple comparison
        supporting_str = [str(item) for item in supporting]
        retrieval_str = [str(item) for item in retrieval]
        
        # Check if all supporting facts are in retrieval facts
        all_present = all(item in retrieval_str for item in supporting_str)
        return 1 if all_present else 0
    
    # Apply the function to each row
    df['coverage_all'] = df.apply(check_coverage, axis=1)
    
    return df

refined_df2 = compare_facts(refined_df)

RAG_context_1 = refined_df2[refined_df2['coverage_all'] == 1].reset_index(drop=True)

# Create result directory if it doesn't exist
os.makedirs('result', exist_ok=True)

# Set the model name for the file prefix - change this for different models
MODEL_PREFIX = "llama3.3_70b"  # Change to "llama_7b" or any other model name

# Initialize an empty list to store results
result_set = []

# Process each item in RAG_context_1
for i in range(len(RAG_context_1)):
    start_time = datetime.now()
    question = RAG_context_1['Question'][i]
    
    print(f"Processing question {i+1}/{len(RAG_context_1)}: {question[:50]}...")
    
    try:
        all_facts = RAG_context_1['Retrieval Result'][i]
        context_data = RAG_context_1['Retrieval Result'][i]
        
        # Process knowledge graph
        knowledge_graph = process_contexts_in_batches(
            context_items=all_facts,
            llm=llm,
            prompt_template=prompt_template_context,
            batch_size=2  # Process 2 contexts at a time
        )
        
        kg = build_rdflib_knowledge_graph(knowledge_graph)
        
        # Process the question with KG-aware exploration
        result = process_question_with_kg_awareness(
            kg=kg,
            context_data=context_data,
            question=question,
            llm=llm,
            max_iterations=3
        )
        
        # Add the result directly to result_set
        result_set.append(result)
        
    except Exception as e:
        # If error occurs, create a result dict with None values except for question
        error_result = {
            "question": question,
            "answer": None,
            "queries_tried": None,
            "kg_exploration": None,
            "evidence_source": None,
            "confidence": None,
            "full_response_for_final": None
        }
        result_set.append(error_result)
        
        # Print error information for debugging
        print(f"Error processing question {i}: {str(e)}")
        print(traceback.format_exc())
    
    # Save intermediate results periodically
    if i % 5 == 0 or i == len(RAG_context_1) - 1:
        temp_df = pd.DataFrame(result_set)
        checkpoint_path = f'result/{MODEL_PREFIX}_checkpoint_{i}.csv'
        temp_df.to_csv(checkpoint_path, index=False)
        print(f"Saved checkpoint at {checkpoint_path}")

# Create final DataFrame from all results
results_df = pd.DataFrame(result_set)

# Save to CSV with model prefix
final_path = f'result/{MODEL_PREFIX}_results_final.csv'
results_df.to_csv(final_path, index=False)

# Save a JSON version for easier programmatic access
json_path = f'result/{MODEL_PREFIX}_results_final.json'
results_df.to_json(json_path, orient='records', lines=True)

# Create a summary file
summary = {
    'model': MODEL_PREFIX,
    'total_questions': len(results_df),
    'successfully_answered': results_df['answer'].notnull().sum(),
    'confidence_avg': results_df['confidence'].mean(),
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

summary_df = pd.DataFrame([summary])
summary_path = f'result/{MODEL_PREFIX}_summary.csv'
summary_df.to_csv(summary_path, index=False)

print(f"Processing complete. Results saved to:")
print(f"  - CSV: {final_path}")
print(f"  - JSON: {json_path}")
print(f"  - Summary: {summary_path}")
print(f"Total questions processed: {len(results_df)}")
print(f"Successfully answered: {results_df['answer'].notnull().sum()}")

# Display first few rows of the results
results_df.head()
