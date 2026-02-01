import json
import re
import ast
from typing import List, Dict, Any, Optional
from rdflib import Graph, Namespace, RDF, URIRef, Literal
import copy



def extract_entities_from_question(llm, question):
    """
    Extract key entities from a question and decompose it into incomplete triples for guided reasoning.
    
    Args:
        llm: The language model
        question: The question to answer
        
    Returns:
        Dictionary with entities, relations, and incomplete triples
    """
    prompt = f"""
    # Question Decomposition for Multi-Hop QA
    
    ## Input Question
    "{question}"
    
    ## Task
    Decompose this question into incomplete knowledge triples that represent the reasoning steps needed to answer it.
    
    1. First, identify all named entities and key terms in the question
    2. Then, break down the reasoning process into sequential steps represented as incomplete triples
    3. Each triple should have the form (entity, relation, entity) where unknown components are marked with "?"
    
    ## Important
    - The decomposition should create a clear reasoning path from question entities to the answer
    - The triples should capture intermediate reasoning steps, not just the final answer
    - Relations should be intuitive and semantically meaningful (e.g., "hasWife", "hasNationality", "bornIn")
    - Make incomplete triples explicit, with "?" for unknown components
    
    ## Output Format
    Return a JSON object with the following structure:
    ```json
    {{
        "entities": ["entity1", "entity2", "entity3"],
        "relations": ["relation1", "relation2"],
        "incomplete_triples": [
            ["entity1", "relation1", "?"],
            ["?", "relation2", "entity2"]
        ],
        "reasoning_path": "Step by step explanation of how these triples form a reasoning path to answer the question"
    }}
    ```
    
    ## Examples
    Question: "What nationality was James Henry Miller's wife?"
    ```json
    {{
        "entities": ["James Henry Miller", "nationality"],
        "relations": ["hasWife", "hasNationality"],
        "incomplete_triples": [
            ["James Henry Miller", "hasWife", "?"],
            ["?", "hasNationality", "?"]
        ],
        "reasoning_path": "First find who is James Henry Miller's wife, then determine that person's nationality"
    }}
    ```
    
    Question: "What university did John Nash teach at before winning the Nobel Prize?"
    ```json
    {{
        "entities": ["John Nash", "Nobel Prize"],
        "relations": ["teachAt", "win", "before"],
        "incomplete_triples": [
            ["John Nash", "teachAt", "?"],
            ["John Nash", "win", "Nobel Prize"],
            ["?", "before", "?"]
        ],
        "reasoning_path": "First find where John Nash taught, then find when he won the Nobel Prize, then determine which teaching position came before the Nobel Prize"
    }}
    ```
    """
    
    response = llm.complete(prompt)
    response_text = str(response)
    
    # Extract JSON from the response
    json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    match = re.search(json_pattern, response_text)
    
    if match:
        try:
            decomposition = json.loads(match.group(1))
            return decomposition
        except json.JSONDecodeError:
            pass
    
    # Fallback: try to find anything that looks like a JSON object
    try:
        brace_pattern = r'\{[\s\S]*\}'
        match = re.search(brace_pattern, response_text)
        if match:
            json_str = match.group(0)
            # Clean up the JSON string
            json_str = re.sub(r',\s*}', '}', json_str)
            decomposition = json.loads(json_str)
            return decomposition
    except:
        pass
    
    # Simple extraction as last resort
    entities = []
    for word in question.split():
        if word[0].isupper() or word in ["nationality", "wife", "spouse"]:
            entities.append(word.strip(",.?!"))
    
    # Create basic incomplete triples based on question type
    incomplete_triples = []
    relations = []
    
    if "nationality" in question.lower():
        relations.append("hasNationality")
        if any("wife" in word.lower() for word in question.split()):
            relations.append("hasWife")
            for entity in entities:
                if entity not in ["nationality", "wife", "spouse"]:
                    incomplete_triples.append([entity, "hasWife", "?"])
            incomplete_triples.append(["?", "hasNationality", "?"])
    
    return {
        "entities": entities,
        "relations": relations,
        "incomplete_triples": incomplete_triples,
        "reasoning_path": "Find the relevant entities and their relationships"
    }


def retrieve_relevant_paragraphs(llm, formatted_context, question_decomposition, k=5):
    """
    Retrieve the k most relevant paragraphs based on incomplete triples.
    
    Args:
        llm: The language model
        context: The full context
        question_decomposition: Dictionary with entities, relations, and incomplete triples
        k: Number of paragraphs to retrieve
        
    Returns:
        List of retrieved paragraphs
    """

    entities = question_decomposition.get("entities", [])
    relations = question_decomposition.get("relations", [])
    incomplete_triples = question_decomposition.get("incomplete_triples", [])
    reasoning_path = question_decomposition.get("reasoning_path", "")
    
    # Format entities and relations for the prompt
    entities_str = ", ".join([f'"{entity}"' for entity in entities])
    relations_str = ", ".join([f'"{relation}"' for relation in relations])
    
    # Format incomplete triples
    triples_str = ""
    for triple in incomplete_triples:
        subject, predicate, obj = triple
        triples_str += f"({subject}, {predicate}, {obj})\n"
    
    prompt = f"""
    # Enhanced Paragraph Retrieval for Multi-Hop QA
    
    ## Question Decomposition
    Entities: [{entities_str}]
    Relations: [{relations_str}]
    
    ## Incomplete Knowledge Triples
    {triples_str}
    
    ## Reasoning Path
    {reasoning_path}
    
    ## Document Corpus
    {formatted_context}
    
    ## Task
    Your task is to retrieve paragraphs that contain information needed to complete these knowledge triples. 
    Follow these steps:
    
    1. For each incomplete triple, identify paragraphs that:
       - Contain the known entities in the triple
       - Express the relation mentioned in the triple
       - Could provide the missing components marked with "?"
       
    2. Prioritize paragraphs that:
       - Connect multiple triples in the reasoning path
       - Contain explicit information about relations between entities
       - Form bridges between different stages in the reasoning process
    
    ## Selection Criteria
    - Paragraphs containing information about relations between entities are highest priority
    - Paragraphs mentioning multiple entities from different triples are second priority
    - Paragraphs with detailed information about single entities are third priority
    
    ## Output Format
    Return a JSON array of the top {k} most relevant paragraphs:
    ```json
    [
        {{
            "title": "document_title",
            "paragraph_index": 0,
            "paragraph_text": "the full text of the paragraph",
            "explanation": "How this paragraph helps complete specific triples",
            "completes_triples": ["(entity1, relation1, ?)", "(?, relation2, entity2)"],
            "rank": 1
        }},
        ...
    ]
    ```
    
    Make sure each paragraph has a "rank" field indicating its relevance (1 = most relevant).
    """

    
    print("Sending prompt to retrieve relevant paragraphs...")
    response = llm.complete(prompt)
    response_text = str(response)
    response_text = response_text.replace("\n", " ").replace("\r", " ").replace("\t", " ").replace("\\", " ")
    
    # Extract JSON from the response - first try with code block
    json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    match = re.search(json_pattern, response_text)
    paragraphs = []
    
    if match:
        try:
            json_str = match.group(1).strip()
            # Fix common JSON issues
            json_str = simple_fix_json(json_str)
            paragraphs = json.loads(json_str)
            
            if isinstance(paragraphs, list):
                print(f"Successfully extracted {len(paragraphs)} paragraphs from LLM response")
            else:
                print("Warning: Extracted JSON is not a list")
                paragraphs = []
                
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Problem near: {json_str[max(0, e.pos-30):min(len(json_str), e.pos+30)]}") 
            
            # Try more aggressive cleanup
            try:
                # Fix quotes within values
                pattern = r':\s*"(.*?)"(?=,|\s*})'
                def escape_quotes(match):
                    value = match.group(1)
                    value = value.replace('"', '\\"')
                    return f': "{value}"'
                
                json_str = re.sub(pattern, escape_quotes, json_str)
                
                # Other common JSON issues
                json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas in arrays
                json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas in objects
                
                paragraphs = json.loads(json_str)
                print("Successfully parsed JSON after additional cleanup")
            except json.JSONDecodeError:
                print("Failed to parse JSON even after additional cleanup")
    
    # If first method fails, try other extraction methods
    if not paragraphs:
        print("Trying alternate JSON extraction methods...")
        
        # Method 2: Try to find a valid JSON array directly
        array_pattern = r'\[\s*\{\s*"title"[\s\S]*?\}\s*\]'
        array_match = re.search(array_pattern, response_text)
        
        if array_match:
            try:
                array_text = array_match.group(0)
                # Fix common JSON issues
                array_text = simple_fix_json(array_text)
                paragraphs = json.loads(array_text)
                print("Successfully extracted paragraphs using direct array pattern")
            except json.JSONDecodeError:
                print("Failed to parse array pattern")
        
        # Method 3: Use regex to extract individual paragraph objects
        if not paragraphs:
            try:
                # Find all JSON-like paragraph objects
                paragraph_pattern = r'\{\s*"title":\s*"([^"]*)"[\s\S]*?"paragraph_text":\s*"([^"]*)"[\s\S]*?\}'
                paragraph_matches = re.finditer(paragraph_pattern, response_text)
                
                extracted_paragraphs = []
                for i, match in enumerate(paragraph_matches):
                    try:
                        title = match.group(1)
                        text = match.group(2)
                        extracted_paragraphs.append({
                            "title": title,
                            "paragraph_text": text,
                            "paragraph_index": i,
                            "rank": i+1,
                            "explanation": "Extracted via regex"
                        })
                    except:
                        continue
                
                if extracted_paragraphs:
                    paragraphs = extracted_paragraphs
                    print(f"Extracted {len(paragraphs)} paragraphs using regex")
            except Exception as e:
                print(f"Regex extraction failed: {e}")
    
    # Final fallback if all else fails
    if not paragraphs:
        print("All extraction methods failed. Using emergency fallback.")
        print('debug oringinal response:', response_text)   
        # Try to at least extract some titles and text
        doc_pattern = r'Document\s+\d+:\s+([^"\n]+)'
        para_pattern = r'Paragraph\s+\d+:\s+([^"\n]+)'
        
        doc_matches = re.findall(doc_pattern, formatted_context)
        para_matches = re.findall(para_pattern, formatted_context)
        
        if doc_matches and para_matches:
            paragraphs = []
            for i in range(min(k, len(doc_matches), len(para_matches))):
                paragraphs.append({
                    "title": doc_matches[i % len(doc_matches)],
                    "paragraph_index": i,
                    "paragraph_text": para_matches[i % len(para_matches)],
                    "explanation": "Emergency fallback extraction",
                    "rank": i+1
                })
        else:
            # Absolute last resort
            paragraphs = [{"title": "EXTRACTION_FAILED", "paragraph_text": "Failed to extract paragraphs from LLM response", "paragraph_index": 0, "rank": 1}]
    
    # Sort paragraphs by rank if available
    if all("rank" in p for p in paragraphs):
        paragraphs.sort(key=lambda p: p.get("rank", float('inf')))
        print("Sorted paragraphs by rank")
    
    # Take only top k paragraphs
    top_k_paragraphs = paragraphs[:k]
    print(f"Selected top {len(top_k_paragraphs)} paragraphs out of {len(paragraphs)}")
    
    # Display selected paragraphs
    for i, para in enumerate(top_k_paragraphs):
        title = para.get("title", "Unknown")
        rank = para.get("rank", i+1)
        print(f"Paragraph {i+1} (Rank: {rank}): {title}")
    
    return top_k_paragraphs


def extract_kg_from_paragraphs(llm, paragraphs):
    """
    Extract knowledge graph from retrieved paragraphs.
    
    Args:
        llm: The language model
        paragraphs: List of retrieved paragraphs
        
    Returns:
        A knowledge graph representation
    """
    # Format paragraphs for the prompt
    formatted_paragraphs = ""
    for i, para in enumerate(paragraphs):
        title = para.get("title", "Unknown")
        text = para.get("paragraph_text", "")
        formatted_paragraphs += f"[{i+1}] {title}: {text}\n\n"
    
    # Prompt for KG extraction
    kg_extraction_prompt = f"""
    # Knowledge Graph Extraction from Text
    
    ## Task
    Extract entities and relationships from the following paragraphs to create a knowledge graph.
    
    ## Paragraphs
    {formatted_paragraphs}
    
    ## Instructions
    1. Identify all named entities in the text
    2. Extract explicit and implicit relationships between entities
    3. Include attributes and properties of entities
    4. Preserve temporal information where available
    
    ## Required Output Format
    Begin your response with ONLY a JSON object in this EXACT format, and nothing else before it:
    
    ```json
    {{
        "entities": [
            {{
                "id": "e1",
                "name": "Entity Name",
                "type": ["entity_type"],
                "mentions": ["mention1", "mention2"],
                "attributes": [
                    {{
                        "attribute": "attribute_name",
                        "value": "attribute_value"
                    }}
                ]
            }}
        ],
        "triples": [
            {{
                "subject": "e1",
                "predicate": "relationship_name",
                "object": "e2",
                "source": "paragraph_index"
            }}
        ]
    }}
    ```
    
    IMPORTANT: Your response MUST begin with this JSON object. Do not include ANY explanatory text before the JSON.
    """
    
    # Get KG extraction from LLM
    response = llm.complete(kg_extraction_prompt)

    print("Extracting KG from LLM response...") 
    response_text = str(response)
   # Clean up escaped quotes in the response
    response_text = response_text.replace("\\'", "'").replace('\n', '').replace('\\', '')

    # Method 1: Extract from code blocks
    json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    match = re.search(json_pattern, response_text)

    if match:
        try:
            json_str = match.group(1).strip()
            # Clean up JSON
            json_str = re.sub(r'//.*$', '', json_str, flags=re.MULTILINE)
            json_str = re.sub(r',\s*([}\]])', r'\1', json_str)

            kg_data = json.loads(json_str)
            print("Successfully extracted KG JSON")
            return kg_data
        except json.JSONDecodeError as e:
            pass

    # Method 2: Extract JSON object directly
    try:
        # Find balanced braces
        brace_count = 0
        start_pos = None
        
        for i, char in enumerate(response_text):
            if char == '{' and brace_count == 0:
                start_pos = i
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_pos is not None:
                    json_str = response_text[start_pos:i+1]
                    # Clean up JSON
                    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
                    kg_data = json.loads(json_str)
                    print("Successfully extracted KG JSON")
                    return kg_data
    except Exception:
        pass

    # Method 3: Pattern-based extraction
    entities = []
    triples = []

    # Extract entities
    entity_patterns = [
        r'"name":\s*"([^"]+)"',
        r'"entity":\s*"([^"]+)"',
        r'entity:\s*"([^"]+)"'
    ]

    for pattern in entity_patterns:
        matches = re.findall(pattern, response_text)
        for i, name in enumerate(matches):
            if not any(e['name'] == name for e in entities):  # Avoid duplicates
                entities.append({
                    "id": f"e{len(entities)+1}",
                    "name": name,
                    "type": ["entity"],
                    "mentions": [name]
                })

    # Extract triples
    triple_patterns = [
        r'"subject":\s*"([^"]+)",\s*"predicate":\s*"([^"]+)",\s*"object":\s*"([^"]+)"',
        r'subject:\s*"([^"]+)",\s*predicate:\s*"([^"]+)",\s*object:\s*"([^"]+)"',
        r'\(([^,]+),\s*([^,]+),\s*([^)]+)\)'  # Pattern for (subject, predicate, object)
    ]

    for pattern in triple_patterns:
        matches = re.findall(pattern, response_text)
        for subj, pred, obj in matches:
            # Clean up the extracted values
            subj = subj.strip().strip('"')
            pred = pred.strip().strip('"')
            obj = obj.strip().strip('"')
            
            triples.append({
                "subject": subj,
                "predicate": pred,
                "object": obj,
                "source": "extraction"
            })

    # Return the result

    if entities or triples:
        print(f"Extracted KG with {len(entities)} entities and {len(triples)} triples")
        print_knowledge_graph({"entities": entities, "triples": triples}, title="Extracted KG")
        return {"entities": entities, "triples": triples}
    else:
        # Return empty KG if nothing found
        print("No KG data found")
        return {"entities": [], "triples": []}

def simple_fix_json(json_str):
    """Simple but effective approach to fix common JSON issues."""
    # Fix quotes within values by finding all value patterns
    pattern = r':\s*"(.*?)"(?=,|\s*})'
    
    def escape_quotes(match):
        value = match.group(1)
        # Escape any unescaped quotes in the value
        value = value.replace('"', '\\"')
        return f': "{value}"'
    
    fixed_json = re.sub(pattern, escape_quotes, json_str)
    return fixed_json

def identify_missing_information(llm, question, kg, retrieved_paragraphs, reasoning_result, question_decomposition=None):
    """
    Identify what information is missing to answer the question, guided by triples.
    
    Args:
        llm: The language model
        question: The question to answer
        kg: The current knowledge graph
        retrieved_paragraphs: The paragraphs retrieved so far
        reasoning_result: Result from reason_over_kg including aliases and missing entities
        question_decomposition: Dictionary with entities, relations, and incomplete triples
        
    Returns:
        Dictionary with missing information details
    """
    # Extract aliases and missing entities from reasoning result
    found_aliases = reasoning_result.get("found_aliases", [])
    missing_entities = reasoning_result.get("missing_entities", [])
    reasoning_chain = reasoning_result.get("reason", "")
    
    # Extract question entities to focus on
    question_entities = []
    words = question.split()
    for i in range(len(words)):
        if words[i][0].isupper():
            # Check for multi-word entities
            entity = words[i]
            j = i + 1
            while j < len(words) and words[j][0].isupper():
                entity += " " + words[j]
                j += 1
            question_entities.append(entity)
    
    # Format KG for the prompt
    kg_entities = ""
    for entity in kg.get("entities", []):
        entity_id = entity.get("id", "")
        entity_name = entity.get("name", "")
        entity_type = ", ".join(entity.get("type", [])) if isinstance(entity.get("type", []), list) else entity.get("type", "")
        kg_entities += f"- {entity_id}: {entity_name} (Type: {entity_type})\n"
    
    kg_relations = ""
    for triple in kg.get("triples", []):
        subject = triple.get("subject", "")
        predicate = triple.get("predicate", "")
        obj = triple.get("object", "")
        kg_relations += f"- {subject} --{predicate}--> {obj}\n"
    
    # Format retrieved paragraphs - include titles and first few words for context
    retrieved_content = ""
    for i, para in enumerate(retrieved_paragraphs):
        title = para.get("title", "Unknown")
        text = para.get("paragraph_text", "")
        preview = text[:50] + "..." if len(text) > 50 else text
        retrieved_content += f"- {title}: {preview}\n"
    
    # Format alias information
    alias_info = ""
    if found_aliases:
        alias_info = "## Detected Aliases\n"
        for alias in found_aliases:
            q_entity = alias.get("question_entity", "")
            alias_found = alias.get("alias_found", "")
            evidence = alias.get("evidence", "")
            alias_info += f"- {q_entity} is also known as {alias_found}\n  Evidence: {evidence}\n"
    
    # Include triple information in the prompt if available
    triples_guidance = ""
    if question_decomposition:
        triples_guidance = "\n## Question Triple Decomposition\n"
        for triple in question_decomposition.get('incomplete_triples', []):
            subject, predicate, obj = triple
            triples_guidance += f"({subject}, {predicate}, {obj})\n"
        
        triples_guidance += f"\n## Reasoning Path\n{question_decomposition.get('reasoning_path', '')}\n"
    
    # Enhanced prompt for missing information identification with triples guidance
    missing_info_prompt = f"""
    # Critical Information Gap Analysis for Triple-Guided Question Answering

    ## Question
    "{question}"
    
    {triples_guidance}
    
    ## Current Knowledge Graph
    
    ### Entities:
    {kg_entities}
    
    ### Relations:
    {kg_relations}
    
    ## Previously Retrieved Content:
    {retrieved_content}
    
    {alias_info}

    ## Previous Reasoning Process:
    {reasoning_chain}

    ## Missing Entities from Previous Analysis
    {', '.join(missing_entities) if missing_entities else 'None explicitly identified'}
    
    ## Task
    You are a critical reasoning expert analyzing knowledge gaps in multi-hop question answering. Your job is to identify exactly what information is missing to answer this question through the triple-based reasoning path.
    
    ## Critical Analysis Steps
    1. For each triple in the reasoning path, determine if it can be completed with existing KG information
    2. Identify which specific triples cannot be completed and why
    3. Determine what additional entities or relations would be needed to complete these triples
    4. Suggest specific search terms to find the missing information
    
    ## Search Strategy
    For each missing piece of information:
    1. Create search terms that target exactly the missing entity or relation
    2. Include both original entity names AND any aliases in search terms
    3. For incomplete triples, create search terms that might reveal the missing components
    4. Consider family relationships, professional relationships, biographical details
    5. Specify exact attributes needed (like "nationality", "spouse", "birthplace")
    
    ## Output Format
    Return a JSON object with the following structure:
    ```json
    {{
        "missing_entities": ["entity1", "entity2"],
        "missing_relations": ["entity1 → relation → entity2", "entity3 → relation → ?"],
        "query_expansion": ["search term 1", "search term 2", "entity1 nationality", "entity2 spouse"],
        "reasoning_gaps": "Detailed description of exactly what information would bridge the reasoning gaps",
        "alias_search_terms": ["alias1", "alias2"]
    }}
    ```
    
    IMPORTANT: Include BOTH original entity names AND any aliases in your search terms. If we've found that "James Henry Miller" is also known as "Ewan MacColl", include BOTH names in search terms.
    """
    
    # Get missing information from LLM
    response = llm.complete(missing_info_prompt)
    response_text = str(response)
    
    # Extract JSON from the response
    json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    match = re.search(json_pattern, response_text)
    
    if match:
        try:
            missing_info = json.loads(match.group(1))
            # Make sure alias_search_terms exists
            if "alias_search_terms" not in missing_info:
                missing_info["alias_search_terms"] = []
            return missing_info
        except json.JSONDecodeError:
            print("Error parsing missing info JSON")
            # Try to clean up and parse again
            json_text = match.group(1).strip()
            json_text = re.sub(r',\s*}', '}', json_text)  # Remove trailing commas in objects
            json_text = re.sub(r',\s*]', ']', json_text)  # Remove trailing commas in arrays
            try:
                missing_info = json.loads(json_text)
                if "alias_search_terms" not in missing_info:
                    missing_info["alias_search_terms"] = []
                return missing_info
            except json.JSONDecodeError:
                print("Failed to parse JSON even after cleanup")
    
    # Look for JSON object directly if code block extraction failed
    try:
        brace_pattern = r'\{[\s\S]*\}'
        match = re.search(brace_pattern, response_text)
        if match:
            json_str = match.group(0)
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*]', ']', json_str)
            missing_info = json.loads(json_str)
            if "alias_search_terms" not in missing_info:
                missing_info["alias_search_terms"] = []
            return missing_info
    except:
        print("JSON direct extraction failed")
    
    # Fallback to empty structure
    return {
        "missing_entities": missing_entities,
        "missing_relations": [],
        "query_expansion": [entity for entity in question_entities if entity],
        "reasoning_gaps": "Could not parse structured gaps from response",
        "alias_search_terms": []
    }


def retrieve_additional_paragraphs(llm, context, missing_info, previous_paragraphs, k=5):
    """
    Retrieve additional paragraphs based on missing information.
    
    Args:
        llm: The language model
        context: The full context
        missing_info: Information about what's missing
        previous_paragraphs: Previously retrieved paragraphs
        k: Number of paragraphs to retrieve
        
    Returns:
        List of additional paragraphs
    """
    # Make sure context is properly parsed if it's a string
    if isinstance(context, str):
        try:
            context = ast.literal_eval(context)
        except (SyntaxError, ValueError):
            print("Error: Could not parse context string")
            return []
    
    # Format the context for the prompt, with document titles prominently displayed
    formatted_context = ""
    for i, item in enumerate(context):
        if isinstance(item, list) and len(item) >= 2:
            title = item[0]
            paragraphs = item[1]
            
            formatted_context += f"### Document: {title}\n"
            
            # Handle both string and list paragraphs
            if isinstance(paragraphs, list):
                for j, paragraph in enumerate(paragraphs):
                    formatted_context += f"Paragraph {j}: {paragraph}\n"
            else:
                formatted_context += f"Text: {paragraphs}\n"
                
            formatted_context += "\n"
    
    # Extract previous paragraph titles to exclude
    previous_titles = set()
    for para in previous_paragraphs:
        title = para.get("title", "")
        if title:
            previous_titles.add(title)
    
    previous_titles_str = ", ".join([f'"{title}"' for title in previous_titles])
    
    # Format missing information with strong emphasis
    missing_entities = ", ".join([f'"{entity}"' for entity in missing_info.get("missing_entities", [])])
    missing_relations = ", ".join([f'"{relation}"' for relation in missing_info.get("missing_relations", [])])
    query_expansion = ", ".join([f'"{term}"' for term in missing_info.get("query_expansion", [])])
    reasoning_gaps = missing_info.get("reasoning_gaps", "")
    
    # Create search queries by combining entities and relations
    search_queries = []
    for entity in missing_info.get("missing_entities", []):
        search_queries.append(entity)
    for term in missing_info.get("query_expansion", []):
        search_queries.append(term)
    
    if not search_queries and missing_relations:
        # Extract entities from relations if no direct entities/queries
        for relation in missing_info.get("missing_relations", []):
            parts = relation.split("→")
            for part in parts:
                cleaned = part.strip().replace("relation", "").strip()
                if cleaned and cleaned != "?":
                    search_queries.append(cleaned)
    
    search_queries_str = ", ".join([f'"{query}"' for query in search_queries])
    
    # Prompt for additional paragraph retrieval with enhanced instructions
    prompt = f"""
    # Targeted Information Retrieval for Multi-Hop Question Answering
    
    ## Critical Information Needs
    We need to find EXACTLY the following missing information:
    
    - Missing entities: [{missing_entities}]
    - Missing relations: [{missing_relations}]
    - Search terms: [{search_queries_str}]
    - Information gaps: "{reasoning_gaps}"
    
    ## Previously Retrieved Documents (EXCLUDE THESE)
    {previous_titles_str}
    
    ## Document Corpus
    {context}
    
    ## Advanced Strategy
    1. Search THOROUGHLY for paragraphs containing ANY of the search terms or entities
    2. Look for IMPLICIT mentions (e.g., "his wife" instead of a direct name)
    3. Prioritize biographical information, family relationships, organizational affiliations
    4. Find paragraphs that connect entities from different documents (bridge documents)
    5. Pay attention to nationality, dates, locations, and proper names
    
    ## Specific Search Instructions
    - For each search term, find paragraphs with EXACT or PARTIAL matches
    - Look for synonyms and alternative expressions (e.g., "born in France" = "French nationality")
    - Consider pronouns that might refer to entities ("he", "she", "they", "it")
    - Identify paragraphs that mention relationships between people or organizations
    
    ## Output Format
    Return a JSON array of exactly {k} most relevant paragraphs:
    ```json
    [
        {{
            "title": "document_title",
            "paragraph_index": 0,
            "paragraph_text": "the full text of the paragraph",
            "explanation": "Specifically explains how this paragraph provides the missing information"
        }},
        ...
    ]
    ```
    
    IMPORTANT: Return exactly {k} paragraphs that are DIFFERENT from previously retrieved documents. The information to answer the question DOES exist in the corpus.
    """
    
    # Get additional paragraphs from LLM
    response = llm.complete(prompt)
    response_text = str(response).replace('\xa0', '').replace('\n', '').replace('\\', '')
    
    # Extract JSON from the response
    json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    match = re.search(json_pattern, response_text)
    
    if match:
        try:
            paragraphs = json.loads(simple_fix_json(match.group(1)))
            if isinstance(paragraphs, list):
                # Filter out any paragraphs that were already retrieved (double-check)
                filtered_paragraphs = []
                for para in paragraphs:
                    if para.get("title", "") not in previous_titles:
                        filtered_paragraphs.append(para)
                
                if filtered_paragraphs:
                    return filtered_paragraphs
                else:
                    print("All retrieved paragraphs were duplicates. Trying again with relaxed constraints.")
            
        except json.JSONDecodeError:
            print("Error parsing additional paragraphs JSON")
            print(f"Response text: ",match.group(1))
    
    # If JSON parsing failed or we got no new paragraphs, try a simpler fallback approach
    # Look for paragraphs with [{ and }] pattern
    fallback_pattern = r'\[\s*\{\s*"title"[\s\S]*?\}\s*\]'
    fallback_match = re.search(fallback_pattern, response_text)
    
    if fallback_match:
        try:
            fallback_text = fallback_match.group(0)
            # Fix common JSON issues
            fallback_text = re.sub(r',(\s*[\]}])', r'\1', fallback_text)  # Remove trailing commas

            paragraphs = json.loads(simple_fix_json(fallback_text))
            if isinstance(paragraphs, list) and paragraphs:
                return paragraphs
        except:
            pass
    
    # Last resort: create a simple structured paragraph from any text found
    print("Using fallback paragraph extraction")
    try:
        # Find any document or paragraph references
        doc_pattern = r'Document\s+\d+:\s+([^"\n]+)'
        para_pattern = r'Paragraph\s+\d+:\s+([^"\n]+)'
        
        doc_matches = re.findall(doc_pattern, formatted_context)
        para_matches = re.findall(para_pattern, formatted_context)
        
        if doc_matches and para_matches:
            fallback_paragraphs = []
            for i in range(min(k, len(doc_matches))):
                fallback_paragraphs.append({
                    "title": doc_matches[i % len(doc_matches)],
                    "paragraph_index": i,
                    "paragraph_text": para_matches[i % len(para_matches)],
                    "explanation": "Fallback extraction method"
                })
            return fallback_paragraphs
    except:
        print("Fallback extraction failed")
    
    return []
    

def print_knowledge_graph(kg, title="Knowledge Graph"):
    """
    Print knowledge graph in triple format using entity names rather than IDs.
    
    Args:
        kg: Knowledge graph dictionary with 'entities' and 'triples'
        title: Title to display before the graph
    """
    if not kg:
        print(f"{title}: Empty graph")
        return
    
    print(f"\n=== {title} ===")
    
    # Create mapping from entity IDs to entity names
    id_to_name = {}
    entities = kg.get('entities', [])
    for entity in entities:
        entity_id = entity.get('id', '')
        entity_name = entity.get('name', 'Unnamed')
        id_to_name[entity_id] = entity_name
    
    # Print entity count
    print(f"Entities: {len(entities)}")
    
    # Print triples
    triples = kg.get('triples', [])
    print(f"Triples: {len(triples)}")
    
    print("debug kg:", entities, triples)

    if triples:
        print("Knowledge Graph Triples:")
        for i, triple in enumerate(triples):
            subj = triple.get('subject', 'Unknown')
            pred = triple.get('predicate', 'Unknown')
            obj = triple.get('object', 'Unknown')
            
            # Replace IDs with names if available
            subj_name = id_to_name.get(subj, subj)
            obj_name = id_to_name.get(obj, obj)
            
            print(f"  ({subj_name}) --[{pred}]--> ({obj_name})")
    
    print("=" * 40)



def verify_entity_consistency(question_entities, kg, question):
    """
    Verify consistency between question entities and knowledge graph entities.
    
    Args:
        question_entities: List of entities extracted from the question
        kg: Knowledge graph dictionary with 'entities' and 'triples'
        question: Original question text
        
    Returns:
        Tuple of (is_consistent, updated_kg)
    """
    print("\n=== Entity Consistency Verification ===")
    print(f"Question entities: {question_entities}")
    
    # Extract all entity names from KG
    kg_entities = [entity.get('name', '') for entity in kg.get('entities', [])]
    print(f"KG entities: {kg_entities}")
    
    # Check for exact matches
    exact_matches = [entity for entity in question_entities if entity in kg_entities]
    print(f"Exact matches: {exact_matches}")
    
    # Check for partial matches (might indicate potential confusion)
    partial_matches = []
    for q_entity in question_entities:
        for kg_entity in kg_entities:
            # Check if one is substring of another but not exact match
            if q_entity != kg_entity and (q_entity in kg_entity or kg_entity in q_entity):
                partial_matches.append((q_entity, kg_entity))
    
    print(f"Potential entity confusions: {partial_matches}")
    
    # If there are potential confusions, verify with LLM
    updated_kg = kg
    is_consistent = len(partial_matches) == 0
    
    if not is_consistent:
        print("Detected potential entity confusion - requesting verification...")
        
        # Create a verification prompt
        verification_prompt = f"""
        # Entity Verification for Knowledge Graph

        I need to verify if certain entities in a knowledge graph are the same or different people/objects.

        ## Original Question
        "{question}"

        ## Entities from Question
        {question_entities}

        ## Entities in Knowledge Graph
        {kg_entities}

        ## Potential Confusion
        I've detected potential confusion between these entity pairs:
        {partial_matches}

        ## Task
        For each pair of potentially confused entities, determine:
        1. Are they the same entity (just different names for the same thing)?
        2. Are they completely different entities?
        3. Is there some other relationship between them?

        If they are different entities, explain how they might be related.

        Return your analysis in this JSON format:
        ```json
        {{
            "entity_pairs": [
                {{
                    "question_entity": "entity from question",
                    "kg_entity": "entity from knowledge graph",
                    "are_same": true/false,
                    "relationship": "explanation of relationship if different"
                }}
            ],
            "action_needed": "create_new_entities" or "merge_entities" or "none",
            "explanation": "brief explanation of what needs to be fixed"
        }}
        ```
        """
        
        # Get verification from LLM
        verification_result = llm.complete(verification_prompt)
        verification_text = str(verification_result)
        
        # Extract JSON from the response
        json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
        match = re.search(json_pattern, verification_text)
        
        if match:
            try:
                verification_data = json.loads(match.group(1))
                print(f"Verification result: {verification_data}")
                
                # Update KG based on verification result
                action_needed = verification_data.get('action_needed', 'none')
                
                if action_needed == 'create_new_entities':
                    # Implement logic to add missing entities to KG
                    # This is a simplified implementation
                    for pair in verification_data.get('entity_pairs', []):
                        if not pair.get('are_same', True):
                            question_entity = pair.get('question_entity', '')
                            if question_entity and question_entity not in kg_entities:
                                # Add missing entity to KG
                                new_entity = {
                                    'id': f"entity_{len(kg.get('entities', []))+1}",
                                    'name': question_entity,
                                    'type': ['entity'],
                                    'mentions': [question_entity]
                                }
                                kg['entities'].append(new_entity)
                    print("Added missing entities to knowledge graph")
                
                elif action_needed == 'merge_entities':
                    # Implement logic to correct entity confusion
                    # (In a full implementation, this would merge or separate entities)
                    print("Entity confusion detected, but merge not implemented")
                    
                # Check if we resolved the inconsistency
                is_consistent = action_needed == 'none'
                updated_kg = kg
                
            except json.JSONDecodeError as e:
                print(f"Error parsing verification JSON: {e}")
        else:
            print("No JSON found in verification response")
    
    print(f"Entity verification complete. Consistency: {is_consistent}")
    print("=" * 40)
    
    return is_consistent, updated_kg

def reason_over_kg(llm, question, kg, question_decomposition=None):
    """
    Reason over the knowledge graph ONLY to answer the question, guided by incomplete triples.
    
    Args:
        llm: The language model
        question: The question to answer
        kg: The knowledge graph
        paragraphs: Not used for reasoning but included for API compatibility
        question_decomposition: Dictionary with entities, relations, and incomplete triples
        
    Returns:
        Dictionary with answer and reasoning
    """
    # Format KG for the prompt
    kg_entities = ""
    for entity in kg.get("entities", []):
        entity_id = entity.get("id", "")
        entity_name = entity.get("name", "")
        entity_type = ", ".join(entity.get("type", [])) if isinstance(entity.get("type", []), list) else entity.get("type", "")
        attributes = ""
        for attr in entity.get("attributes", []):
            attr_name = attr.get("attribute", "")
            attr_value = attr.get("value", "")
            attributes += f"    - {attr_name}: {attr_value}\n"
        kg_entities += f"- {entity_id}: {entity_name} (Type: {entity_type})\n{attributes}"
    
    kg_relations = ""
    for triple in kg.get("triples", []):
        subject = triple.get("subject", "")
        predicate = triple.get("predicate", "")
        obj = triple.get("object", "")
        source = triple.get("source", "")
        kg_relations += f"- {subject} --{predicate}--> {obj} [Source: {source}]\n"
    
    # If question_decomposition is not provided, extract it from the question
    if question_decomposition is None:
        # Extract key entities from the question
        question_entities_prompt = f"""
        Extract all named entities from this question: "{question}"
        Return only the entities, not the whole question.
        Format: ["entity1", "entity2", ...]
        """
        
        response = llm.complete(question_entities_prompt)
        
        # Parse entities from response
        try:
            entities_pattern = r'\[(.*?)\]'
            match = re.search(entities_pattern, str(response))
            if match:
                entities_str = match.group(1)
                # Clean up the entities string
                entities_str = entities_str.replace('"', '"').replace('"', '"')  # Handle fancy quotes
                entities_list = json.loads(f"[{entities_str}]")
                question_entities = ", ".join([f'"{e}"' for e in entities_list])
            else:
                # Simple fallback
                question_entities = ", ".join([f'"{w}"' for w in question.split() if w[0].isupper()])
        except:
            question_entities = "Could not parse entities"
            
        # Create a simple triples decomposition
        triples_str = "Could not parse question into triples"
        reasoning_path = "Find relevant entities and their relationships"
    else:
        # Extract components from question_decomposition
        entities = question_decomposition.get("entities", [])
        relations = question_decomposition.get("relations", [])
        incomplete_triples = question_decomposition.get("incomplete_triples", [])
        reasoning_path = question_decomposition.get("reasoning_path", "")
        
        # Format question entities
        question_entities = ", ".join([f'"{entity}"' for entity in entities])
        
        # Format incomplete triples
        triples_str = ""
        for triple in incomplete_triples:
            subject, predicate, obj = triple
            triples_str += f"({subject}, {predicate}, {obj})\n"
    
    # Prompt for reasoning over KG ONLY with triple-guided reasoning
    reasoning_prompt = f"""
    # Knowledge Graph Reasoning for Question Answering
    You are a knowledge graph reasoning expert. Your task is to analyze ONLY the knowledge graph to answer the question. You must not use any external information or paragraphs - rely exclusively on the structured knowledge in the graph.

    ## Question
    "{question}"
    
    ## Key Question Entities
    {question_entities}
    
    ## Question Decomposition into Triples
    {triples_str}
    
    ## Reasoning Path
    {reasoning_path}
    
    ## Knowledge Graph
    
    ### Entities:
    {kg_entities}
    
    ### Relations:
    {kg_relations}
    
    ## Triple-Guided Reasoning Task
    Use ONLY the knowledge graph to answer the question by completing the reasoning path. For each step:
    
    1. Find entities and relations in the KG that match the question entities
    2. Use KG traversal to discover missing connections between entities
    3. Connect findings together according to the reasoning path
    4. Do NOT use any information outside the KG
    
    ## Critical Entity Verification Steps
    Before attempting to complete reasoning chains:
    
    1. For each entity in the question (like "James Henry Miller"), check if that EXACT entity exists in the knowledge graph
       
    2. If a question entity isn't in the KG, check if it has an alias in the KG (through "alias_of" relations or entity mentions)
       
    3. Only proceed with reasoning if you can confirm ALL question entities exist in the KG (either directly or through aliases)
    
    ## Step-by-Step Reasoning Process
    
    ### Step 1: Entity Mapping
    - Map each question entity to KG entities or their aliases
    - If any key entity cannot be mapped, the question is unanswerable with this KG
    
    ### Step 2: Relation Path Finding
    - For each relation in the reasoning path, find corresponding KG relation paths
    - Document exactly which KG triples support each reasoning step
    
    ### Step 3: Knowledge Chain Construction
    - Link the identified KG triples to form a connected reasoning chain
    - Each step must be supported by explicit KG triples
    - No assumptions or external knowledge should be used
    
    ### Step 4: Answer Extraction
    - The final answer should be directly derived from KG triples
    - If the reasoning chain has any gaps, the answer is "null"
    
    ## Important Guidelines
    - Use ONLY the provided knowledge graph for reasoning
    - Every reasoning step must be explicitly supported by KG triples
    - Do NOT fabricate connections or assume relationships not in the KG
    - If a critical entity or relation is missing from the KG, return "null" as the answer
    - Be explicit about which KG triples support each part of your reasoning
    - You must try to answer the question but not just answer one of the sub-questions
    

    ## Required Output Format
    Return a JSON object with the following structure:
    
    ```json
    {{
        "answer": "Your answer derived solely from KG (or 'null' if not answerable from KG)",
        "answerable": "Yes/No (if answerable from KG ONLY)",
        "KG_answerable": "Yes/No (must be the same as 'answerable' since ONLY using KG)",
        "reason": "Your detailed KG-based reasoning process",
        "reasoning_chain": [
            "Step 1: Used KG triple (entity1 --relation--> entity2) to establish first connection",
            "Step 2: Used KG triple (entity2 --relation--> entity3) to establish second connection",
            "Step 3: Final answer follows from KG triple (entity3 --relation--> answer_entity)"
        ],
        "found_aliases": [
            {{
                "question_entity": "Entity from question", 
                "alias_found": "Alias found in KG",
                "evidence": "KG relation or mention showing this alias"
            }}
        ],
        "missing_entities": ["Entities from question with no match in KG"],
        "missing_relations": ["Relations needed but missing from KG"]
    }}
    ```
    """

    # Get reasoning from LLM
    response = llm.complete(reasoning_prompt)
    response_text = str(response)
    response_text = response_text.replace("\\'", "'").replace('\n', '').replace('\\', '')

    # Extract JSON from the response - Method 1: Code blocks
    json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    match = re.search(json_pattern, response_text)

    if match:
        try:
            json_str = match.group(1).strip()
            # Clean up JSON
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*]', ']', json_str)
            result = json.loads(json_str)
            print("Successfully extracted JSON from code block")
            return result
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")

    # Method 2: Extract JSON object directly
    try:
        # Find balanced braces
        brace_count = 0
        start_pos = None
        
        for i, char in enumerate(response_text):
            if char == '{' and brace_count == 0:
                start_pos = i
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_pos is not None:
                    json_str = response_text[start_pos:i+1]
                    # Clean up JSON
                    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
                    result = json.loads(json_str)
                    print("Successfully extracted JSON object")
                    return result
    except Exception as e:
        print(f"JSON extraction error: {e}")

    # Fallback for extraction failures
    print("Failed to extract JSON response, returning default")
    return {
        "answer": "null",
        "answerable": "No",
        "KG_answerable": "No",
        "reason": "Could not extract valid reasoning from response",
        "reasoning_chain": [],
        "found_aliases": [],
        "missing_entities": ["Failed to extract entity information"],
        "missing_relations": ["Failed to extract relation information"]
    }


def merge_knowledge_graphs(kg1, kg2):
    """
    Merge two knowledge graphs.
    
    Args:
        kg1: First knowledge graph
        kg2: Second knowledge graph
        
    Returns:
        Merged knowledge graph
    """
    # Create sets of entity names to track duplicates
    entity_names = set(entity.get("name", "") for entity in kg1.get("entities", []))
    
    # Merge entities
    merged_entities = kg1.get("entities", []).copy()
    for entity in kg2.get("entities", []):
        entity_name = entity.get("name", "")
        if entity_name not in entity_names:
            merged_entities.append(entity)
            entity_names.add(entity_name)
    
    # Merge triples
    merged_triples = kg1.get("triples", []).copy()
    merged_triples.extend(kg2.get("triples", []))
    
    return {
        "entities": merged_entities,
        "triples": merged_triples
    }


def kg_enhanced_qa_pipeline(llm, question, context, initial_paragraphs, question_decomposition=None, max_iterations=3):
    """
    Main pipeline for KG-enhanced question answering with triple-guided reasoning.
    
    Args:
        llm: The language model
        question: The question to answer
        context: The full context
        initial_paragraphs: Initially retrieved paragraphs
        question_decomposition: Dictionary with entities, relations, and incomplete triples
        max_iterations: Maximum number of KG enhancement iterations
    
    Returns:
        Dictionary with answer and reasoning
    """
    # If question_decomposition is not provided, create it
    if question_decomposition is None:
        print("No question decomposition provided. Generating now...")
        question_decomposition = extract_entities_from_question(llm, question)
        print(f"Generated question decomposition with {len(question_decomposition.get('incomplete_triples', []))} reasoning triples")
    
    # Start with initial paragraphs
    all_paragraphs = copy.deepcopy(initial_paragraphs)

    print("Building knowledge graph from initial paragraphs...")
    # Always build knowledge graph from the initial paragraphs
    kg = extract_kg_from_paragraphs(llm, all_paragraphs)
    
    # Print initial KG
    print_knowledge_graph(kg, "Initial Knowledge Graph")
    
    # First attempt to answer with KG and triples
    max_retries = 3
    retry_attempts = 0
    initial_result = {}

    while True:
        if retry_attempts >= max_retries:
            print("Max retry attempts reached. Breaking loop.")
            break

        try:
            # Use question_decomposition for reasoning
            initial_result = reason_over_kg(llm, question, kg, question_decomposition)
            
            if initial_result.get("answer") is not None and initial_result.get("answerable") is not None and initial_result.get("reason") is not None:
                print("Successfully reasoned over KG using triple guidance")
                break
            else:
                print("Failed to reason over KG. Retrying...")
                retry_attempts += 1
        except Exception as e:
            retry_attempts += 1
            print(f"Error in reasoning: {e}")
            print("Retrying reasoning...")

    # Print raw result for debugging
    print("Initial KG-based answer result:")
    print(f"Answer: {initial_result.get('answer', 'No answer found')}")
    print(f"Reasoning chain length: {len(initial_result.get('reason', ''))}")
    
    print(f"Full response: {initial_result}")   

    # Print any aliases found
    found_aliases = initial_result.get('found_aliases', [])
    if found_aliases:
        print("\nDetected aliases:")
        for alias in found_aliases:
            q_entity = alias.get("question_entity", "")
            alias_found = alias.get("alias_found", "")
            print(f"- {q_entity} = {alias_found}")
    
    # Handle null results
    if initial_result.get('answer') is None:
        initial_result['answer'] = "null"
    
    # Check if the answer is confident
    answer_text = initial_result.get('answer', '').lower()
    
    if initial_result.get('answerable', '').lower() == 'yes':
        print("Initial KG-based answer is confident. Returning result.")
        return initial_result
    else:
        print(f"Initial KG-based answer '{answer_text}' is not confident. Enhancing knowledge graph...")
    
    # Begin enhancement iterations
    for iteration in range(max_iterations):
        print(f"==="*10)
        print(f"KG enhancement iteration {iteration+1}/{max_iterations}")
        print(f"==="*10)
        
        # Use question_decomposition to identify missing information
        missing_info = identify_missing_information(llm, question, kg, all_paragraphs, initial_result, question_decomposition)
        
        print(f"Missing entities: {missing_info.get('missing_entities', [])}")
        print(f"Missing relations: {missing_info.get('missing_relations', [])}")
        print(f"Alias search terms: {missing_info.get('alias_search_terms', [])}")
        
        # Add alias search terms to query expansion
        if 'alias_search_terms' in missing_info and missing_info['alias_search_terms']:
            if 'query_expansion' not in missing_info:
                missing_info['query_expansion'] = []
            missing_info['query_expansion'].extend(missing_info['alias_search_terms'])
        
        # Retrieve additional paragraphs with expanded search terms
        additional_paragraphs = retrieve_additional_paragraphs(llm, context, missing_info, all_paragraphs)
        
        if not additional_paragraphs:
            print("No additional paragraphs found. Breaking iteration.")
            break
        
        print(f"Retrieved {len(additional_paragraphs)} additional paragraphs")
        
        # Add new paragraphs to the collection
        all_paragraphs.extend(additional_paragraphs)
        
        # Extract KG from new paragraphs
        new_kg = extract_kg_from_paragraphs(llm, additional_paragraphs)
        
        # Merge KGs
        kg = merge_knowledge_graphs(kg, new_kg)
        
        # Print the merged KG
        print_knowledge_graph(kg, f"Knowledge Graph after iteration {iteration+1}")
        
        # Try to answer with enhanced KG
        retry_attempts = 0
        max_retries = 3
        result = {}

        while True:
            if retry_attempts >= max_retries:
                print("Max retry attempts reached. Breaking loop.")
                break
                
            try:
                # Use question_decomposition for reasoning
                result = reason_over_kg(llm, question, kg, question_decomposition)
                
                if result.get("answer") is not None and result.get("answerable") is not None and result.get("reason") is not None:
                    break
                else:
                    print("Failed to reason over enhanced KG. Retrying...")
                    retry_attempts += 1
            except Exception as e:
                retry_attempts += 1
                print(f"Error in reasoning: {e}")
                print("Retrying reasoning...")

        # Handle null results
        if result.get('answer') is None:
            result['answer'] = "null"

        # Print iteration result
        print(f"Iteration {iteration+1} result:")
        print(f"Answer: {result}")
        
        # Check if the answer is confident
        answer_text = result.get('answer', '').lower()
        
        if result.get('answerable', '').lower() == 'yes':
            print("Found confident answer. Returning result.")
            return result
    
    # If no confident answer after max iterations, return best effort
    final_result = reason_over_kg(llm, question, kg, question_decomposition)
    
    # Ensure answer field exists
    if final_result.get('answer') is None:
        final_result['answer'] = "null"
        
    return final_result
  