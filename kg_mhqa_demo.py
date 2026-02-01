import argparse
import json
import logging
import pandas as pd
import os
from typing import Dict, List, Any, Optional

# Import the knowledge graph reasoner
from kg_reasoner import KnowledgeGraphReasoner, process_question_with_kg
from custom_prompt import QueryRefinementPromptGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("kg_mhqa.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Optional: Import LLM if available
try:
    from llama_index.llms.ollama import Ollama
    HAS_LLM = True
except ImportError:
    HAS_LLM = False
    logger.warning("LLM library not found. Running in basic mode without LLM.")

def load_llm(model_name: str = "llama3.1:70b") -> Optional[Any]:
    """
    Load the language model if available.
    
    Args:
        model_name: Name of the model to load
        
    Returns:
        Language model instance or None if not available
    """
    if not HAS_LLM:
        return None
        
    try:
        logger.info(f"Loading LLM model: {model_name}")
        llm = Ollama(model=model_name, request_timeout=240.0)
        return llm
    except Exception as e:
        logger.error(f"Error loading LLM: {str(e)}")
        return None

def load_data(data_file: str) -> Dict:
    """
    Load test data from a file.
    
    Args:
        data_file: Path to the data file
        
    Returns:
        Dictionary with loaded data
    """
    logger.info(f"Loading data from: {data_file}")
    
    try:
        # Determine file type based on extension
        if data_file.endswith('.json'):
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
        elif data_file.endswith('.txt'):
            with open(data_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # Try to parse as JSON or Python literal
                try:
                    data = json.loads(content)
                except json.JSONDecodeError:
                    data = eval(content)  # Be careful with eval!
                    
        else:
            logger.error(f"Unsupported file format: {data_file}")
            raise ValueError(f"Unsupported file format: {data_file}")
            
        return data
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def process_single_example(kg_file: str, context_file: str, question: str, llm=None) -> Dict:
    """
    Process a single example with the knowledge graph reasoner.
    
    Args:
        kg_file: Path to the knowledge graph file
        context_file: Path to the context file
        question: The question to answer
        llm: Optional language model
        
    Returns:
        Dictionary with results
    """
    logger.info(f"Processing example: {question}")
    
    try:
        # Load knowledge graph
        with open(kg_file, 'r', encoding='utf-8') as f:
            kg_data = f.read()
            
        # Load context
        context_data = load_data(context_file)
        
        # Process with the reasoner
        result = process_question_with_kg(question, kg_data, context_data, llm)
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing example: {str(e)}")
        return {
            "question": question,
            "answer": "Error processing example",
            "error": str(e)
        }

def run_interactive_demo(kg_file: str, context_file: str, llm=None) -> None:
    """
    Run an interactive demo where the user can input questions.
    
    Args:
        kg_file: Path to the knowledge graph file
        context_file: Path to the context file
        llm: Optional language model
    """
    logger.info("Starting interactive demo")
    
    try:
        # Load knowledge graph and context once
        with open(kg_file, 'r', encoding='utf-8') as f:
            kg_data = f.read()
            
        context_data = load_data(context_file)
        
        # Create a reasoner instance
        reasoner = KnowledgeGraphReasoner(llm=llm)
        reasoner.load_kg_from_string(kg_data)
        reasoner.load_context(context_data)
        
        print("\n=== Knowledge Graph MHQA Interactive Demo ===")
        print(f"Loaded KG from: {kg_file}")
        print(f"Loaded context from: {context_file}")
        print("Type 'quit' or 'exit' to end the demo")
        
        while True:
            print("\n" + "-" * 50)
            question = input("Enter your question: ")
            
            if question.lower() in ['quit', 'exit']:
                break
                
            if not question.strip():
                continue
                
            try:
                # Process the question
                result = reasoner.answer_question(question)
                
                # Display results
                print("\nAnswer:", result['answer'])
                print("Confidence:", result['confidence'])
                print("Evidence source:", result['evidence_source'])
                
                # Ask if the user wants to see the queries
                show_queries = input("\nShow SPARQL queries? (y/n): ")
                if show_queries.lower() == 'y':
                    for i, query in enumerate(result['queries_tried']):
                        print(f"\nQuery {i+1}:")
                        print(query)
                
            except Exception as e:
                print(f"Error: {str(e)}")
                
        print("\nDemo ended. Thanks for using the Knowledge Graph MHQA system!")
        
    except Exception as e:
        logger.error(f"Error in interactive demo: {str(e)}")
        print(f"Error: {str(e)}")

def process_dataset(dataset_file: str, output_file: str, llm=None) -> None:
    """
    Process a dataset of questions and save the results.
    
    Args:
        dataset_file: Path to the dataset file
        output_file: Path to save the results
        llm: Optional language model
    """
    logger.info(f"Processing dataset: {dataset_file}")
    
    try:
        # Load the dataset
        dataset = pd.read_json(dataset_file)
        
        results = []
        total = len(dataset)
        
        # Process each example
        for i, row in dataset.iterrows():
            logger.info(f"Processing example {i+1}/{total}")
            
            question = row['question']
            kg_data = row['kg'] if 'kg' in row else row.get('kg_path', None)
            context_data = row['context'] if 'context' in row else row.get('context_path', None)
            
            # Load KG data if it's a file path
            if isinstance(kg_data, str) and os.path.isfile(kg_data):
                with open(kg_data, 'r', encoding='utf-8') as f:
                    kg_data = f.read()
                    
            # Load context data if it's a file path
            if isinstance(context_data, str) and os.path.isfile(context_data):
                context_data = load_data(context_data)
                
            # Process the example
            try:
                result = process_question_with_kg(question, kg_data, context_data, llm)
                results.append(result)
                
                # Print progress
                print(f"Example {i+1}/{total}: {question}")
                print(f"Answer: {result['answer']}")
                print("-" * 30)
                
            except Exception as e:
                logger.error(f"Error processing example {i+1}: {str(e)}")
                results.append({
                    "question": question,
                    "answer": "Error processing example",
                    "error": str(e)
                })
                
        # Save the results
        results_df = pd.DataFrame(results)
        results_df.to_json(output_file, orient='records', indent=2)
        
        logger.info(f"Saved results to: {output_file}")
        print(f"Processed {total} examples. Results saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Error processing dataset: {str(e)}")
        print(f"Error: {str(e)}")

def main():
    """Main function to parse arguments and run the demo."""
    parser = argparse.ArgumentParser(description="Knowledge Graph MHQA Demo")
    
    # Define command-line arguments
    parser.add_argument("--mode", choices=["interactive", "single", "dataset"], default="interactive",
                        help="Demo mode: interactive, single example, or dataset processing")
                        
    parser.add_argument("--kg", default="kg.ttl",
                        help="Path to the knowledge graph file")
                        
    parser.add_argument("--context", default="datarow.txt",
                        help="Path to the context file")
                        
    parser.add_argument("--question", 
                        help="Question to answer (for single mode)")
                        
    parser.add_argument("--dataset", 
                        help="Path to dataset file (for dataset mode)")
                        
    parser.add_argument("--output", default="results.json",
                        help="Path to save results (for dataset mode)")
                        
    parser.add_argument("--model", default="llama3.1:70b",
                        help="LLM model to use")
                        
    parser.add_argument("--no-llm", action="store_true",
                        help="Run without the language model")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Load LLM if needed and available
    llm = None
    if not args.no_llm and HAS_LLM:
        llm = load_llm(args.model)
    
    # Run the appropriate mode
    if args.mode == "interactive":
        run_interactive_demo(args.kg, args.context, llm)
        
    elif args.mode == "single":
        if not args.question:
            print("Error: --question is required for single mode")
            return
            
        result = process_single_example(args.kg, args.context, args.question, llm)
        
        # Print the result
        print("\n=== Result ===")
        print(f"Question: {result['question']}")
        print(f"Answer: {result['answer']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Evidence source: {result['evidence_source']}")
        
        print("\n=== Queries Tried ===")
        for i, query in enumerate(result['queries_tried']):
            print(f"\nQuery {i+1}:")
            print(query)
            
    elif args.mode == "dataset":
        if not args.dataset:
            print("Error: --dataset is required for dataset mode")
            return
            
        process_dataset(args.dataset, args.output, llm)

if __name__ == "__main__":
    main()