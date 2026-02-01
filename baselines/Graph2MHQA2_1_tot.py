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


def tree_of_thought_answer(llm, question, paragraphs):
    """Answer the question using Tree-of-Thought reasoning approach."""
    # Format paragraphs for the prompt
    formatted_paragraphs = ""
    for i, para in enumerate(paragraphs):
        title = para.get("title", "Unknown")
        text = para.get("paragraph_text", "")
        formatted_paragraphs += f"[{i+1}] {title}: {text}\n\n"
    
    prompt = f"""
    # Multi-Hop Question Answering with Tree-of-Thought Reasoning
    
    ## Question
    {question}
    
    ## Retrieved Information
    {formatted_paragraphs}
    
    ## Tree-of-Thought Reasoning Instructions
    You will solve this multi-hop question by exploring multiple reasoning paths in a tree-like structure.
    
    ### Step 1: Question Decomposition
    Break down the complex question into 2-3 simpler sub-questions or reasoning directions. Each forms an initial branch of your reasoning tree.
    
    ### Step 2: Branch Exploration
    For each branch (sub-question):
    1. Identify relevant information from the retrieved texts
    2. Generate 2-3 possible intermediate conclusions
    3. Evaluate the strength of evidence for each conclusion (strong, moderate, weak)
    4. Select the most promising conclusion with the strongest evidence
    
    ### Step 3: Branch Evaluation and Selection
    Compare conclusions from different branches and select the most promising path based on:
    - Strength of supporting evidence
    - Logical consistency with the question
    - Completeness of the reasoning chain
    
    ### Step 4: Final Reasoning and Answer
    Follow the selected path to its logical conclusion, integrating information across branches if necessary.
    
    ## Output Format
    Structure your reasoning as a tree and provide your final answer in JSON format:
    
    ```json
    {{
        "reasoning_tree": {{
            "branch1": {{
                "sub_question": "First sub-question",
                "relevant_info": ["Key information from paragraph [X]", "Key information from paragraph [Y]"],
                "possible_conclusions": [
                    {{ "conclusion": "Possible conclusion 1", "evidence_strength": "strong/moderate/weak" }},
                    {{ "conclusion": "Possible conclusion 2", "evidence_strength": "strong/moderate/weak" }}
                ],
                "best_conclusion": "Selected conclusion with strongest evidence"
            }},
            "branch2": {{
                "sub_question": "Second sub-question",
                "relevant_info": ["Key information from paragraph [Z]"],
                "possible_conclusions": [
                    {{ "conclusion": "Possible conclusion 1", "evidence_strength": "strong/moderate/weak" }},
                    {{ "conclusion": "Possible conclusion 2", "evidence_strength": "strong/moderate/weak" }}
                ],
                "best_conclusion": "Selected conclusion with strongest evidence"
            }}
        }},
        "selected_path": "Description of which branch(es) led to the final answer",
        "answer": "Your final answer to the original question",
        "confidence": "high/medium/low"
    }}
    ```
    
    If you cannot determine the answer from the given information, set "answer" to "null" and explain why in "selected_path".
    """
    
    response = llm.complete(prompt)
    response_text = str(response)
    return response_text

def evaluate_result_correctness_tot(llm, llm_reply, question, gold_answer, retrieved_paragraphs, gold_supporting_fact_content=""):
    """
    Ask the LLM to evaluate if the answer and reasoning are correct based on direct output.
    
    Args:
        llm: The language model to use
        llm_reply: Raw string output from the LLM (Tree-of-Thought reasoning)
        question: The original question
        gold_answer: The known correct answer
        retrieved_paragraphs: List of paragraphs that were retrieved
        gold_supporting_fact_content: Optional supporting facts content
        
    Returns:
        Dictionary with evaluation results (answer_correct, supporting_fact_correct, reason)
    """
    # Format retrieved paragraphs for the prompt
    formatted_paragraphs = ""
    for i, para in enumerate(retrieved_paragraphs):
        title = para.get("title", "Unknown")
        text = para.get("paragraph_text", "")
        formatted_paragraphs += f"[{i+1}] {title}: {text}\n"
    
    # Evaluation prompt - directly using the full LLM reply
    eval_prompt = f"""
    # Evaluation of Tree-of-Thought Question Answering Results
    
    ## Question
    {question}
    
    ## Gold Answer
    "{gold_answer}"
    
    ## System's Retrieved Paragraphs
    {formatted_paragraphs}
    
    ## System's Complete Tree-of-Thought Output
    {llm_reply}
    
    ## Gold Supporting Facts Content
    "{gold_supporting_fact_content}"
    
    ## Evaluation Instructions
    
    Review the system's Tree-of-Thought reasoning and determine:
    
    1. If the final answer in the reasoning is semantically equivalent to the gold answer. Consider variations in wording, capitalization, etc.
    
    2. If the system's retrieved paragraphs and reasoning path contains the key information needed to answer the question, similar to what's in the gold supporting facts.
    
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
    
    # Fallback extraction methods - still need these for parsing the evaluator's response
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
        llm_reply = tree_of_thought_answer(llm, question, relevant_paragraphs)
        
        # Evaluate the result using the simplified function
        evaluation = evaluate_result_correctness_tot(
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