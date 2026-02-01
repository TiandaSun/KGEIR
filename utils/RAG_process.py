from typing import List, Dict, Tuple, Union, Any, Optional
import pandas as pd
import logging
import numpy as np
import re
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def extract_json_content(text):
    """
    Extract JSON content from text, handling both triple-backtick-wrapped and plain JSON.
    Can extract JSON from markdown code blocks or plaintext that contains JSON-like structures.
    """
    # Try first with triple backticks
    json_pattern = r'```json\s*([\s\S]*?)\s*```'
    json_matches = re.finditer(json_pattern, text)
    all_json_objects = []
    
    for match in json_matches:
        try:
            json_str = match.group(1).strip()
            json_str = re.sub(r'//.*$', '', json_str, flags=re.MULTILINE)
            json_obj = json.loads(json_str)
            all_json_objects.append(json_obj)
        except json.JSONDecodeError:
            continue

    # If no JSON found in triple backticks, try to find JSON-like structures
    if not all_json_objects:
        # First, find all potential JSON blocks (text between { and })
        start_indices = [m.start() for m in re.finditer(r'\{', text)]
        
        for start_idx in start_indices:
            # Keep track of nested braces
            brace_count = 0
            potential_json = ""
            
            # Scan through the text from the starting brace
            for i in range(start_idx, len(text)):
                char = text[i]
                potential_json += char
                
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    
                # If we've found a matching closing brace
                if brace_count == 0:
                    try:
                        # Clean up the string
                        json_str = potential_json.strip()
                        # Remove any trailing commas before closing braces
                        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
                        # Parse the JSON
                        json_obj = json.loads(json_str)
                        all_json_objects.append(json_obj)
                    except json.JSONDecodeError:
                        pass
                    break

    if not all_json_objects:
        print("No valid JSON content found in the text")
        return None,None
    
    return all_json_objects[-1], all_json_objects

class PassageMatch:
    def __init__(self, title: str, sent_idx: int, text: str, score: float):
        self.title = title
        self.sent_idx = sent_idx
        self.text = text
        self.score = score
        
    def to_dict(self) -> Dict:
        return {
            'passage': [self.title, self.text],  # Modified to return [title, text]
            'score': self.score,
            'location': [self.title, self.sent_idx]
        }
        
    def __str__(self) -> str:
        return f"Score: {self.score:.3f} | {self.text} | Location: [{self.title}, {self.sent_idx}]"

class RelationAwareRetriever:
    def __init__(self, llm, model_name: str = 'all-MiniLM-L6-v2'):
        self.llm = llm
        self.encoder = SentenceTransformer(model_name)
        
    def extract_components(self, question: str) -> Dict:
        prompt = """
        First, convert this question into an incomplete SPARQL-like query pattern to identify the key entities and relations we need to find. Then extract those components.

        Question: "{question}"

        Step 1: Convert to SPARQL pattern
        Think about what information we need to find and express it as a SPARQL-like pattern with missing information marked as '?'.

        Example 1:
        Question: "What nationality was James Henry Miller's wife?"
        SPARQL pattern:
        SELECT ?nationality
        WHERE {{
            James_Henry_Miller has_spouse ?wife .
            ?wife has_nationality ?nationality .
        }}

        Step 2: Extract components based on the SPARQL pattern.
        Return in this format:
        ```json
        {{
            "sparql_pattern": "The SPARQL pattern from step 1",
            "entities": [
                {{
                    "name": "entity name from pattern",
                    "type": "subject/object/variable",
                    "aliases": ["alternative names", "variations"],
                    "role": "role in the query (e.g., main subject, target, intermediate)"
                }}
            ],
            "relations": [
                {{
                    "name": "relation from pattern",
                    "type": "type of relation",
                    "patterns": [
                        "exact phrases to look for",
                        "alternative expressions"
                    ],
                    "context_terms": [
                        "broader context words",
                        "related concepts"
                    ]
                }}
            ],
            "reasoning_chain": "step by step explanation based on SPARQL pattern"
        }}
        ```

        Example 1 Response:
        ```json
        {{
            "sparql_pattern": "SELECT ?nationality WHERE {{ James_Henry_Miller has_spouse ?wife . ?wife has_nationality ?nationality . }}",
            "entities": [
                {{
                    "name": "James Henry Miller",
                    "type": "subject",
                    "aliases": ["James Miller", "James H. Miller"],
                    "role": "main subject"
                }},
                {{
                    "name": "wife",
                    "type": "variable",
                    "aliases": ["spouse", "partner"],
                    "role": "intermediate target"
                }}
            ],
            "relations": [
                {{
                    "name": "has_spouse",
                    "type": "personal",
                    "patterns": [
                        "married to",
                        "wife was",
                        "spouse"
                    ],
                    "context_terms": [
                        "marriage",
                        "wedding",
                        "family"
                    ]
                }},
                {{
                    "name": "has_nationality",
                    "type": "biographical",
                    "patterns": [
                        "nationality",
                        "born in",
                        "citizen of"
                    ],
                    "context_terms": [
                        "origin",
                        "birth",
                        "country"
                    ]
                }}
            ],
            "reasoning_chain": "1. Find James Henry Miller, 2. Find his wife, 3. Determine wife's nationality"
        }}
        ```

        Now analyze this question: {question}
        """
        
        response = self.llm.complete(prompt.format(question=question))
        try:
            components, _ = extract_json_content(str(response))
            return components
        except (ValueError, json.JSONDecodeError) as e:
            print(f"Error extracting components: {e}")
            return None
        
    def score_passage_for_relation(self, text: str, relation: Dict) -> float:
        """Enhanced scoring for relation matching."""
        text_lower = text.lower()
        
        # Pattern Matching Score
        pattern_matches = sum(pattern.lower() in text_lower 
                            for pattern in relation['patterns'])
        pattern_score = pattern_matches / len(relation['patterns'])
        
        # Context Terms Score
        context_matches = sum(term.lower() in text_lower 
                            for term in relation['context_terms'])
        context_score = context_matches / len(relation['context_terms'])
        
        # Semantic Similarity Score
        relation_text = ' '.join(relation['patterns'] + relation['context_terms'])
        text_embedding = self.encoder.encode(text,show_progress_bar=False)
        relation_embedding = self.encoder.encode(relation_text,show_progress_bar=False)
        semantic_score = cosine_similarity(
            text_embedding.reshape(1, -1),
            relation_embedding.reshape(1, -1)
        )
        
        # Combine scores with weights
        final_score = (0.4 * pattern_score + 
                      0.3 * context_score + 
                      0.3 * semantic_score)
        
        return final_score

    def find_relevant_paragraphs(self, 
                               contexts: List[List[str]], 
                               components: Dict,
                               k: int = 3) -> Dict[str, List[Dict]]:
        """Find relevant paragraphs using enhanced scoring."""
        entity_matches = []
        relation_matches = {rel['name']: [] for rel in components['relations']}
        
        # Process each context
        for title, sentences in contexts:
            for sent_idx, sentence in enumerate(sentences):
                # Entity matching
                for entity in components['entities']:
                    if (entity['name'].lower() in sentence.lower() or
                        any(alias.lower() in sentence.lower() 
                            for alias in entity['aliases'])):
                        
                        entity_text = ' '.join([entity['name']] + entity['aliases'])
                        similarity = cosine_similarity(
                            self.encoder.encode(sentence,show_progress_bar=False).reshape(1, -1),
                            self.encoder.encode(entity_text,show_progress_bar=False).reshape(1, -1)
                        )[0][0]
                        
                        match = PassageMatch(title, sent_idx, sentence, similarity)
                        entity_matches.append(match)
                
                # Relation matching
                for relation in components['relations']:
                    score = self.score_passage_for_relation(sentence, relation)
                    if score > 0.01:  # Threshold
                        match = PassageMatch(title, sent_idx, sentence, score)
                        relation_matches[relation['name']].append(match)
        
        # Sort and get top k matches
        entity_matches.sort(key=lambda x: x.score, reverse=True)
        entity_matches = entity_matches[:k]
        
        for rel_name in relation_matches:
            relation_matches[rel_name].sort(key=lambda x: x.score, reverse=True)
            relation_matches[rel_name] = relation_matches[rel_name][:k]
        
        # Convert to dictionary format
        results = {
            'sparql_pattern': components['sparql_pattern'],
            'entity_matches': [m.to_dict() for m in entity_matches],
            'relation_matches': {
                rel: [m.to_dict() for m in matches]
                for rel, matches in relation_matches.items()
            }
        }
        
        return results
    
    def retrieve(self, question: str, contexts: List[List[str]], k: int = 5) -> Dict:
        """Main retrieval method."""
        components = self.extract_components(question)
        if not components:
            return None
        try:

            relevant_paragraphs = self.find_relevant_paragraphs(contexts, components, k)
        except Exception as e:
            print(f"Error retrieving paragraphs: {e}")
            return None
        
        return {
            'components': components,
            'relevant_paragraphs': relevant_paragraphs
        }

def extract_passages(json_data):
    """
    Extract and group passages by title.
    Returns a list of dictionaries where each dictionary contains a title and its associated passages.
    """
    # Initialize a dictionary to store passages by titlejupyter lab --no-browser --ip 0.0.0.0
    passages_by_title = {}
    
    # Process entity matches
    for match in json_data['relevant_paragraphs']['entity_matches']:
        title, text = match['passage']  # Now passage is [title, text]
        if title not in passages_by_title:
            passages_by_title[title] = set()  # Use set to avoid duplicates
        passages_by_title[title].add(text)
    
    # Process relation matches
    for relation in json_data['relevant_paragraphs']['relation_matches'].values():
        for match in relation:
            title, text = match['passage']  # Now passage is [title, text]
            if title not in passages_by_title:
                passages_by_title[title] = set()
            passages_by_title[title].add(text)
    
    # Convert sets to lists and create final structure
    result = [
        {
            'title': title,
            'paragraphs': sorted(list(paragraphs))  # Convert set to sorted list
        }
        for title, paragraphs in passages_by_title.items()
    ]
    
    return result

def refine_supporting_facts(result_df: pd.DataFrame, 
                          rag_context_df: pd.DataFrame, 
                          refine_dataset_df: pd.DataFrame,
                          verbose: bool = True) -> pd.DataFrame:
    """
    Refine and match supporting facts across different datasets.
    
    Args:
        result_df: DataFrame containing 'Retrieval Fact' column
        rag_context_df: DataFrame containing 'Retrieval Result' column
        refine_dataset_df: DataFrame containing 'context' column
        verbose: Whether to print processing information (default: True)
    
    Returns:
        DataFrame with refined Retrieval Fact
    """
    
    # Set up logging
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)
    logger = logging.getLogger(__name__)
    
    refined_df = result_df.copy()
    
    def is_valid_index(idx: Any) -> bool:
        """Check if index is a valid non-negative integer."""
        return isinstance(idx, int) and idx >= 0
    
    for idx in range(len(result_df)):
        try:
            # Get data for current row
            supporting_facts = result_df['Retrieval Facts'].iloc[idx]
            rag_context = rag_context_df['Retrieval Result'].iloc[idx]
            refined_context = refine_dataset_df['context'].iloc[idx]
            
            if not isinstance(supporting_facts, list):
                logger.warning(f"Row {idx}: Invalid supporting_facts format")
                continue
            
            # Create mapping from RAG context
            rag_mapping = {}
            for item in rag_context:
                if isinstance(item, dict) and 'title' in item and 'paragraphs' in item:
                    rag_mapping[item['title']] = item['paragraphs']
            
            # Create mapping from refined context
            refined_mapping = {}
            for item in refined_context:
                if isinstance(item, list) and len(item) == 2:
                    title, paragraphs = item
                    if isinstance(paragraphs, list):
                        refined_mapping[title] = paragraphs
            
            # Process each supporting fact
            refined_facts = []
            for fact in supporting_facts:
                try:
                    # Skip if fact is None or not a list
                    if not fact or not isinstance(fact, list) or len(fact) != 2:
                        logger.warning(f"Row {idx}: Skipping invalid fact format: {fact}")
                        continue
                    
                    title, para_idx = fact
                    
                    # Skip if title is None or para_idx is not valid
                    if title is None or not is_valid_index(para_idx):
                        logger.warning(f"Row {idx}: Skipping fact with None or invalid index: {fact}")
                        continue
                    
                    # Skip if title not found in either context
                    if title not in refined_mapping:
                        logger.warning(f"Row {idx}: Title not found in refined context: {title}")
                        continue
                    
                    if title not in rag_mapping:
                        logger.warning(f"Row {idx}: Title not found in RAG context: {title}")
                        continue
                    
                    # First check if the paragraph exists in refined context
                    if para_idx >= len(refined_mapping[title]):
                        logger.warning(f"Row {idx}: Paragraph index {para_idx} out of range in refined context for title {title}")
                        continue
                        
                    # If we have the paragraph in both contexts, try to match content
                    if para_idx < len(rag_mapping[title]):
                        rag_para = rag_mapping[title][para_idx].strip()
                        # Look for exact match first
                        found_match = False
                        for refined_idx, refined_para in enumerate(refined_mapping[title]):
                            if refined_para.strip() == rag_para:
                                refined_facts.append([title, refined_idx])
                                found_match = True
                                break
                        
                        if found_match:
                            continue

                    # If no exact match found or paragraph index only exists in refined context,
                    # use the original index if it's valid in refined context
                    if para_idx < len(refined_mapping[title]):
                        refined_facts.append([title, para_idx])
                    
                except Exception as e:
                    logger.warning(f"Row {idx}: Error processing fact {fact}: {str(e)}")
                    continue
            
            # Update the refined dataframe
            refined_df.at[idx, 'Retrieval Facts'] = refined_facts
            
            if verbose and idx % 100 == 0:
                logger.info(f"Processed {idx} rows...")
                
        except Exception as e:
            logger.error(f"Error processing row {idx}: {str(e)}")
            continue
    
    logger.info("Processing complete")
    return refined_df

def rag_generate(dataset,llm):

    coverage = pd.DataFrame(columns=['Question', 'gold_answer','Supporting Facts','SPARQL query','Retrieval Facts','Retrieval Result' ,'coverage_all', 'coverage_rate'])

    for data_row in dataset:
        question = data_row['question']
        contexts = data_row['context']

        retriever = RelationAwareRetriever(llm)
        result = retriever.retrieve(question, contexts)
        location_list = []

        # Check if result is None or doesn't have the expected structure
        if result is None or not isinstance(result, dict) or 'relevant_paragraphs' not in result:
            print(f"Skipping question due to invalid retriever result: {question[:50]}...")
            
            # Add row with empty retrieval
            new_row = pd.DataFrame({
                'Question': [question],
                'gold_answer': [data_row['answer']],
                'Supporting Facts': [data_row['supporting_facts']],
                'SPARQL query': [''],  # Empty SPARQL query
                'Retrieval Facts': [[]],  # Empty list for no retrieval
                'Retrieval Result': [[]],  # Empty list for no retrieval
                'coverage_all': ['0'],    # No coverage
                'coverage_rate': [0.0]    # Zero coverage rate
            })
            coverage = pd.concat([coverage, new_row], ignore_index=True)
            continue

        try:
            # Only proceed if entity_matches exists
            if 'entity_matches' in result['relevant_paragraphs']:
                for match in result['relevant_paragraphs']['entity_matches']:
                    location_list.append(tuple(match['location']))
            
            # Only proceed if relation_matches exists
            if 'relation_matches' in result['relevant_paragraphs']:
                for rel_name, matches in result['relevant_paragraphs']['relation_matches'].items():
                    for match in matches:
                        location_list.append(tuple(match['location']))

            passages_list = extract_passages(result)
        except Exception as e:
            print(f"Error processing retrieval results for question: {question[:50]}...")
            print(f"Error: {str(e)}")
            location_list.append(('No Match',))

        # Create unique_list using list comprehension to avoid duplicates
        unique_list = []
        [unique_list.append(list(x)) for x in set(location_list) if list(x) not in unique_list]
        
        count = sum(len(context[1]) for context in data_row['context'])


        new_row = pd.DataFrame({
            'Question': [question],
            'gold_answer': [data_row['answer']],
            'Supporting Facts': [data_row['supporting_facts']],
            'Retrieval Facts': [unique_list],
            'SPARQL query': [result['components']['sparql_pattern']],
            'Retrieval Result': [passages_list],
            'coverage_all': ['1' if all(sf in unique_list for sf in data_row['supporting_facts']) else '0'],
            'coverage_rate': [len(unique_list)/count]
        })
        
        coverage = pd.concat([coverage, new_row], ignore_index=True)

    return coverage



