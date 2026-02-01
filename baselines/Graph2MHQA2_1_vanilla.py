from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage, MessageRole

# Initialize the model
llm = Ollama(model="gemma3:27b", request_timeout=240.0, temperature=0.0)

llm.complete('hi')

import json
import re
import ast

def retrieve_relevant_paragraphs_direct(llm, context, question, k=5):
    """Retrieve the k most relevant paragraphs from the context based on the question directly."""
    # Make sure context is properly parsed if it's a string
    if isinstance(context, str):
        try:
            context = ast.literal_eval(context)
        except (SyntaxError, ValueError):
            print("Error: Could not parse context string")
            return [{"title": "ERROR", "paragraph_text": "Could not parse context"}]
    
    
    prompt = f"""
    # Paragraph Retrieval for Question Answering

    ## Question
    {question}

    ## Task Definition
    Your task is to select the {k} most relevant paragraphs that would help answer the given question.

    ## Document Corpus
    {context}

    ## Selection Criteria
    1. Does the paragraph contain information directly related to the question?
    2. Does the paragraph contain key facts, dates, numbers, or names mentioned in the question?
    3. Does the paragraph provide context or background information needed to understand the topic of the question?
    4. Could the paragraph be part of a chain of information needed to answer the question?

    ## Return Format
    Return your selections as a JSON array:
    ```json
    [
        {{
            "title": "document_title",
            "paragraph_index": 0,
            "paragraph_text": "the full text of the paragraph",
            "relevance": "Brief explanation of why this paragraph is relevant"
        }},
        ...
    ]
    ```
    """
    
    response = llm.complete(prompt)
    response_text = str(response)
    # Extract JSON from the response
    json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    match = re.search(json_pattern, response_text)
    
    if match:
        try:
            paragraphs = json.loads(match.group(1))
            if isinstance(paragraphs, list):
                return paragraphs
        except json.JSONDecodeError:
            pass
    
    # Fallback to manual extraction
    return [{"title": "ERROR", "paragraph_text": "Could not extract paragraphs from response"}]

def simple_answer_question(llm, question, paragraphs):
    """Answer the question using the retrieved paragraphs without complex reasoning chains."""
    # Format paragraphs for the prompt
    formatted_paragraphs = ""
    for i, para in enumerate(paragraphs):
        title = para.get("title", "Unknown")
        text = para.get("paragraph_text", "")
        formatted_paragraphs += f"[{i+1}] {title}: {text}\n\n"
    
    prompt = f"""
    # Question Answering Task
    
    ## Question
    {question}
    
    ## Available Information
    {formatted_paragraphs}
    
    ## Instructions
    Please answer the question directly based on the information provided above.
    Provide only the answer without explanations or reasoning steps.
    If you cannot determine the answer from the given information, respond with "null". and tell me why.
    
    ## Answer format:
    Return your answer as json
    ```json
    {{
        "answer": "your_answer_here",
        "reason": "your_reasoning_here"
    }}
    ```
    """
    
    response = llm.complete(prompt)
    response_text = str(response)
    return response_text

def vanilla_baseline(example):
    """Process a single example through a simple retrieval-based pipeline without entity extraction."""
    llm = Ollama(model="gemma3:27b", request_timeout=240.0, temperature=0.0)
    
    question = example.get('question', '')
    context = example.get('context', [])
    gold_answer = example.get('answer', '')
    gold_supporting_fact_content = example.get('supporting_fact_content', [])

    print(f"Question: {question}")
    print(f"Gold answer: {gold_answer}")
    print("-" * 50)
    
    # Step 1: Directly retrieve relevant paragraphs using the question (no entity extraction)
    print("Retrieving relevant paragraphs...")
    relevant_paragraphs = retrieve_relevant_paragraphs_direct(llm, context, question)

    print(f"Retrieved {len(relevant_paragraphs)} paragraphs")
    print("-" * 50)
    
    # Step 2: Directly answer the question based on retrieved paragraphs
    print("Generating direct answer...")
    result = simple_answer_question(llm, question, relevant_paragraphs)

    print("\nFinal Answer:",result)


def evaluate_result_correctness(llm, llm_reply, question, gold_answer, retrieved_paragraphs, gold_supporting_fact_content=""):
    """
    Ask the LLM to evaluate if the answer and reasoning are correct based on direct output.
    
    Args:
        llm: The language model to use
        llm_reply: Raw string output from the LLM
        question: The original question
        gold_answer: The known correct answer
        retrieved_paragraphs: List of paragraphs that were retrieved
        gold_supporting_fact_content: Optional supporting facts content
        
    Returns:
        Dictionary with evaluation results (answer_correct, supporting_fact_correct, reason)
    """
    # Extract answer and reasoning from LLM reply using regex or JSON parsing
    extracted_answer = ""
    extracted_reasoning = ""
    
    # Try to extract as JSON first
    json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    match = re.search(json_pattern, llm_reply)
    
    if match:
        try:
            json_str = match.group(1).strip()
            # Clean up the JSON string
            json_str = re.sub(r'\s+(?=["{\[:])', '', json_str)
            json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
            
            parsed_reply = json.loads(json_str)
            extracted_answer = parsed_reply.get("answer", "")
            extracted_reasoning = parsed_reply.get("reason", "")
        except json.JSONDecodeError:
            # If JSON parsing fails, try regex extraction
            answer_pattern = r'"answer"\s*:\s*"([^"]*)"'
            reason_pattern = r'"reason"\s*:\s*"([^"]*)"'
            
            answer_match = re.search(answer_pattern, llm_reply)
            reason_match = re.search(reason_pattern, llm_reply)
            
            extracted_answer = answer_match.group(1) if answer_match else "Failed to extract answer"
            extracted_reasoning = reason_match.group(1) if reason_match else "Failed to extract reasoning"
    else:
        # If no JSON block found, treat the whole response as the answer
        extracted_answer = llm_reply[:200]  # Take first 200 chars as answer
        extracted_reasoning = llm_reply  # Use full reply as reasoning
    
    # Format retrieved paragraphs for the prompt
    formatted_paragraphs = ""
    for i, para in enumerate(retrieved_paragraphs):
        title = para.get("title", "Unknown")
        text = para.get("paragraph_text", "")
        formatted_paragraphs += f"[{i+1}] {title}: {text}\n"
    
    # Evaluation prompt
    eval_prompt = f"""
    # Evaluation of Question Answering Results
    
    ## Question
    {question}
    
    ## Predicted Answer
    "{extracted_answer}"
    
    ## Gold Answer
    "{gold_answer}"
    
    ## System's Retrieved Paragraphs
    {formatted_paragraphs}
    
    ## System's Reasoning
    {extracted_reasoning}
    
    ## Gold Supporting Facts Content
    "{gold_supporting_fact_content}"
    
    ## Evaluation Instructions
    
    1. Compare the predicted answer with the gold answer and determine if they are semantically equivalent.
       - Consider variations in wording, capitalization, etc.
       - Focus on the core meaning, not exact string matching.
    
    2. Determine if the system's retrieved paragraphs contain the key information needed to answer the question.
       - The paragraphs should contain the essential information present in the gold supporting facts.
       - The reasoning should correctly connect information to reach the answer.
    
    ## Required Output Format
    
    You must respond ONLY with a JSON object using this exact format:
    
    ```json
    {{
        "answer_correct": true/false,
        "supporting_fact_correct": true/false,
        "reason": "Your detailed explanation of both evaluations"
    }}
    ```
    
    Use boolean true/false values (not strings) for the evaluation results.
    """
    
    # Get evaluation from LLM
    response = llm.complete(eval_prompt)
    response_text = str(response)
    
    # Extract JSON from the response
    json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    match = re.search(json_pattern, response_text)
    
    if match:
        try:
            json_text = match.group(1)
            # Clean up the JSON text
            json_text = re.sub(r',\s*}', '}', json_text)
            evaluation = json.loads(json_text)
            
            # Ensure keys are present
            if 'answer_correct' in evaluation and 'supporting_fact_correct' in evaluation and 'reason' in evaluation:
                return evaluation
        except Exception as e:
            print(f"JSON parsing error: {e}")
    
    # Fallback extraction methods
    try:
        # Try to find JSON without code block markers
        json_pattern = r'\{\s*"answer_correct":\s*(true|false),\s*"supporting_fact_correct":\s*(true|false),\s*"reason":\s*"([^"]*)"'
        match = re.search(json_pattern, response_text, re.IGNORECASE)
        
        if match:
            answer_correct = match.group(1).lower() == "true"
            supporting_correct = match.group(2).lower() == "true"
            reason = match.group(3)
            
            return {
                "answer_correct": answer_correct,
                "supporting_fact_correct": supporting_correct,
                "reason": reason
            }
    except Exception as e:
        print(f"Fallback JSON extraction error: {e}")
    
    # Last resort: text analysis
    print("Using text analysis fallback for evaluation")
    lower_text = response_text.lower()
    
    answer_correct = (
        "answer is correct" in lower_text or 
        "answer_correct: true" in lower_text or
        "answer_correct\": true" in lower_text
    )
    
    supporting_correct = (
        "supporting facts are correct" in lower_text or
        "supporting_fact_correct: true" in lower_text or
        "supporting_fact_correct\": true" in lower_text
    )
    
    # Extract some explanation text
    reason = "Could not extract structured evaluation. Review the evaluation text directly."
    
    return {
        "answer_correct": answer_correct,
        "supporting_fact_correct": supporting_correct,
        "reason": reason
    }


import pandas as pd
import argparse


parser = argparse.ArgumentParser(description='Process HotpotQA dataset')
parser.add_argument('--path', type=str, default='dataset/hotpot/hotpot_train_v1.1_400.csv',
                    help='Path to the HotpotQA CSV file')

parser.add_argument('--result', type=str, default='result_v2/evaluate_result.csv',
                    help='Path to the HotpotQA CSV file')

args = parser.parse_args()




dataset = pd.read_csv(args.path)
evaluation_pd = pd.DataFrame(columns=['answer_correct', 'supporting_fact_correct', 'reason'])

for index, row in dataset.iterrows():
    try:
        # Get the question and gold answer
        question = row.get('question', '')
        gold_answer = row.get('answer', '')
        gold_supporting_fact_content = row.get('supporting_fact_content', '')
        
        # Get paragraphs and LLM reply
        relevant_paragraphs = retrieve_relevant_paragraphs_direct(llm, row.get('context', []), question)
        llm_reply = simple_answer_question(llm, question, relevant_paragraphs)
        
        # Evaluate the result
        evaluation = evaluate_result_correctness(
            llm, 
            llm_reply, 
            question,
            gold_answer, 
            relevant_paragraphs, 
            gold_supporting_fact_content
        )
        
        evaluation_pd = pd.concat([evaluation_pd, pd.DataFrame([evaluation])], ignore_index=True)
        
    except Exception as e:
        print(f"Error processing row {index}: {e}")
        continue
    
evaluation_pd.to_csv(args.result, index=False)
print(evaluation_pd['answer_correct'].value_counts(), evaluation_pd['supporting_fact_correct'].value_counts())