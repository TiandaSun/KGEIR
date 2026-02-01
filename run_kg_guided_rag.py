#!/usr/bin/env python3
"""
Script to run the KG-Guided RAG model on the HotpotQA dataset.
Simple usage: python run_kg_guided_rag.py
"""

from llama_index.llms.ollama import Ollama
import json
import pandas as pd
import os
import argparse
from tqdm import tqdm
from datetime import datetime
import sys
import traceback

# Import the KG-Guided RAG system
from utils.kg_guided_rag import KGGuidedRAG, kg_guided_rag_generate
# Import prompt templates
sys.path.append(os.path.abspath("MHQA-ontology"))
from prompts.prompt_templates import prompt_template_context
from utils.RAG_process import refine_supporting_facts
from utils.iterative_reasoner import process_question_with_kg_awareness
from utils.KG_builder import build_rdflib_knowledge_graph, process_contexts_in_batches

def compare_facts(refined_df):
    """
    Compares if all Supporting Facts are in Retrieval Facts.
    Sets coverage_all to 1 if all are present, else 0.
    """
    import ast
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

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run KG-Guided RAG model')
    parser.add_argument('--model', type=str, default='llama3.3',
                        help='Model name for Ollama (default: llama3.3)')
    parser.add_argument('--sample_size', type=int, default=100,
                        help='Number of samples to process (default: 100)')
    parser.add_argument('--output_dir', type=str, default='result',
                        help='Directory to save results (default: result)')
    parser.add_argument('--max_iterations', type=int, default=3,
                        help='Maximum iterations for KG-guided RAG (default: 3)')
    parser.add_argument('--dataset_path', type=str, 
                        default='dataset/hotpot/hotpot_train_v1.1.json',
                        help='Path to HotpotQA dataset')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize model
    print(f"Initializing {args.model} model...")
    llm = Ollama(model=args.model, request_timeout=600.0, keep_alive=-1)
    
    # Load dataset
    print(f"Loading dataset from {args.dataset_path}...")
    try:
        hotpot_data = json.load(open(args.dataset_path))
        dataset = hotpot_data[:args.sample_size]
        print(f"Loaded {len(dataset)} examples")
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return
    
    # Generate RAG context using KG-guided approach
    print("Running KG-Guided RAG process...")
    kg_rag = KGGuidedRAG(llm, max_iterations=args.max_iterations)
    RAG_context = kg_guided_rag_generate(dataset, llm, prompt_template_context)
    
    # Prepare dataset for refinement
    print("Refining supporting facts...")
    dataset_df = pd.DataFrame(dataset)
    refine_dataset = dataset_df[dataset_df['question'].isin(RAG_context['Question'])].reset_index(drop=True)
    refined_df = refine_supporting_facts(RAG_context, RAG_context, refine_dataset)
    
    # Compare facts to get coverage
    refined_df2 = compare_facts(refined_df)
    
    # Filter to examples with full coverage of supporting facts
    RAG_context_1 = refined_df2[refined_df2['coverage_all'] == 1].reset_index(drop=True)
    
    # Print coverage statistics
    coverage_count = sum(1 for x in refined_df2['coverage_all'] if x == 1)
    print(f"Coverage of all supporting facts: {coverage_count}/{len(refined_df2)} ({coverage_count/len(refined_df2)*100:.2f}%)")
    
    # Set model prefix for file naming
    model_prefix = f"{args.model}_kg_guided"
    
    # Initialize an empty list to store results
    result_set = []
    
    # Process each item with full coverage for answer generation
    print("Generating answers for questions with full supporting facts coverage...")
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
            checkpoint_path = f'{args.output_dir}/{model_prefix}_checkpoint_{i}.csv'
            temp_df.to_csv(checkpoint_path, index=False)
            print(f"Saved checkpoint at {checkpoint_path}")
    
    # Create final DataFrame from all results
    results_df = pd.DataFrame(result_set)
    
    # Save to CSV with model prefix
    final_path = f'{args.output_dir}/{model_prefix}_results_final.csv'
    results_df.to_csv(final_path, index=False)
    
    # Save a JSON version for easier programmatic access
    json_path = f'{args.output_dir}/{model_prefix}_results_final.json'
    results_df.to_json(json_path, orient='records', lines=True)
    
    # Create a summary file
    summary = {
        'model': args.model,
        'rag_method': 'kg_guided',
        'max_iterations': args.max_iterations,
        'total_questions': len(results_df),
        'successfully_answered': results_df['answer'].notnull().sum(),
        'confidence_avg': results_df['confidence'].mean(),
        'supporting_facts_coverage': coverage_count/len(refined_df2),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    summary_df = pd.DataFrame([summary])
    summary_path = f'{args.output_dir}/{model_prefix}_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    
    print(f"\nProcessing complete. Results saved to:")
    print(f"  - CSV: {final_path}")
    print(f"  - JSON: {json_path}")
    print(f"  - Summary: {summary_path}")
    print(f"Total questions processed: {len(results_df)}")
    print(f"Successfully answered: {results_df['answer'].notnull().sum()}")
    print(f"Supporting facts coverage: {coverage_count}/{len(refined_df2)} ({coverage_count/len(refined_df2)*100:.2f}%)")

if __name__ == "__main__":
    main()


    # python run_kg_guided_rag.py --model gemma3:12b --sample_size 20 --output_dir results --max_iterations 4