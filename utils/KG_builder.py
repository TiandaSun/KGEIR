import uuid
from typing import Dict, List, Optional
from tqdm import tqdm
import concurrent.futures
from functools import partial

def make_entity_ids_unique(triple_data: Dict) -> Dict:
    """
    Replace entity IDs in the triple data with unique identifiers.
    
    Args:
        triple_data (Dict): Dictionary containing entities and triples
        
    Returns:
        Dict: Updated dictionary with unique entity IDs
    """       
    if not triple_data or 'entities' not in triple_data:
        return triple_data
        
    # Create ID mapping
    id_mapping = {}
    
    # Update entity IDs
    for entity in triple_data['entities']:
        if 'id' in entity:
            old_id = entity['id']
            new_id = f"entity_{str(uuid.uuid4())[:8]}"  # Using first 8 chars of UUID
            id_mapping[old_id] = new_id
            entity['id'] = new_id
    
    # Update references in triples if they exist
    if 'triples' in triple_data:
        for triple in triple_data['triples']:
            for field in ['subject', 'object']:
                if triple.get(field) in id_mapping:
                    triple[field] = id_mapping[triple[field]]
    
    return triple_data


def process_batch(batch: List[Dict], llm, prompt_template_context) -> tuple[List[Dict], List[int]]:
    """
    Process a batch of context items in parallel.
    
    Args:
        batch (List[Dict]): A batch of context items, each containing 'title' and 'paragraphs'
        llm: The language model instance
        prompt_template_context: Template for context processing
    
    Returns:
        tuple[List[Dict], List[int]]: Processed triples and evaluation results
    """
    batch_triples = []
    batch_evaluations = []
    
    for item in batch:
        try:
            # Combine title and paragraphs into a single context string
            title = item.get('title', '')
            paragraphs = item.get('paragraphs', [])
            
            # Create a context string that includes the title and all paragraphs
            context_str = f"Title: {title}\n"
            context_str += "Content: " + " ".join(paragraphs)
            
            # Process with LLM
            result_context = llm.complete(prompt_template_context.format(context_str=context_str))
            json_knowledge_triple_last, json_knowledge_triple_all = extract_json_content(str(result_context))
            
            if json_knowledge_triple_last is not None:
                processed_triple = make_entity_ids_unique(json_knowledge_triple_last)
                batch_triples.append(processed_triple)
                batch_evaluations.append(1)
            else:
                batch_evaluations.append(0)
                
        except Exception as e:
            batch_evaluations.append(0)
            print(f"Error processing context item: {str(e)}")
            
    return batch_triples, batch_evaluations


def extract_context_triples(all_facts: List[Dict], llm, prompt_template_context, batch_size: int = 10) -> tuple[List[Dict], List[int]]:
    """
    Process context data in batches to generate triples.
    
    Args:
        all_facts (List[Dict]): List of context items, where each item is a dict with 'title' and 'paragraphs'
        llm: The language model instance
        prompt_template_context: Template for context processing
        batch_size (int): Number of items to process in each batch
        
    Returns:
        tuple[List[Dict], List[int]]: List of processed triple data with unique entity IDs and evaluation results
    """
    context_triples = []
    evaluation_results = []
    
    # Create batches from the list of dictionaries
    batches = [all_facts[i:i + batch_size] for i in range(0, len(all_facts), batch_size)]
    
    # Process batches with progress bar
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # Create a partial function with fixed arguments
        process_func = partial(process_batch, llm=llm, prompt_template_context=prompt_template_context)
        
        # Process batches with progress bar
        results = list(tqdm(
            executor.map(process_func, batches),
            total=len(batches),
            desc="Processing batches"
        ))
        
        # Combine results
        for batch_triples, batch_evaluations in results:
            context_triples.extend(batch_triples)
            evaluation_results.extend(batch_evaluations)
            
    return context_triples, evaluation_results

import json
import uuid
import re
from typing import Dict, List, Optional

def format_context_for_llm(context_items: List[Dict]) -> str:
    """
    Format all context items into a single string for processing by the LLM.
    
    Args:
        context_items (List[Dict]): List of context items, each with 'title' and 'paragraphs'
        
    Returns:
        str: Formatted context string
    """
    formatted_context = []
    
    for i, item in enumerate(context_items):
        title = item.get('title', 'Untitled')
        paragraphs = item.get('paragraphs', [])
        
        # Format each context item with a clear separator and index
        formatted_item = f"CONTEXT ITEM #{i+1}:\n"
        formatted_item += f"Title: {title}\n"
        formatted_item += "Content: " + " ".join(paragraphs) + "\n"
        formatted_item += "-" * 50  # Add a separator
        
        formatted_context.append(formatted_item)
    
    # Join all formatted items with newlines
    return "\n\n".join(formatted_context)

def make_entity_ids_unique(triple_data: Dict) -> Dict:
    """
    Replace entity IDs in the triple data with unique identifiers.
    
    Args:
        triple_data (Dict): Dictionary containing entities and triples
        
    Returns:
        Dict: Updated dictionary with unique entity IDs
    """
    if not triple_data or 'entities' not in triple_data:
        return triple_data
    
    # Create ID mapping
    id_mapping = {}
    
    # Update entity IDs
    for entity in triple_data['entities']:
        if 'id' in entity:
            old_id = entity['id']
            new_id = f"entity_{str(uuid.uuid4())[:8]}"  # Using first 8 chars of UUID
            id_mapping[old_id] = new_id
            entity['id'] = new_id
    
    # Update references in triples if they exist
    if 'triples' in triple_data:
        for triple in triple_data['triples']:
            for field in ['subject', 'object']:
                if triple.get(field) in id_mapping:
                    triple[field] = id_mapping[triple[field]]
    
    return triple_data

def build_knowledge_graph(context_items: List[Dict], llm, prompt_template: str) -> Dict:
    """
    Build a knowledge graph from context items with a single LLM call.
    
    Args:
        context_items (List[Dict]): List of context items
        llm: Language model instance
        prompt_template: Prompt template for the LLM
        
    Returns:
        Dict: Knowledge graph with entities and triples
    """
    # Format all context items into a single string
    formatted_context = format_context_for_llm(context_items)
    
    # Create the prompt with all context items
    prompt = prompt_template.format(context_str=formatted_context)

    # Make a single call to the LLM
    llm_response = llm.complete(prompt)
    
    # Extract JSON content from the response
    knowledge_graph = extract_json_content(str(llm_response))
    
    # Make entity IDs unique
    if knowledge_graph:
        knowledge_graph = make_entity_ids_unique(knowledge_graph)
    
    return knowledge_graph

def visualize_knowledge_graph(knowledge_graph):
    """
    Visualize a knowledge graph that might be in different formats.
    Handles both dictionary and tuple formats.
    
    Args:
        knowledge_graph: Either a dictionary with 'entities' and 'triples' keys,
                         or a tuple where the first element is such a dictionary.
    """
    # Handle the case where knowledge_graph is a tuple
    if isinstance(knowledge_graph, tuple) and len(knowledge_graph) > 0:
        # Extract the first element which contains the actual knowledge graph
        kg_data = knowledge_graph[0]
    else:
        # Use as is if it's already a dictionary
        kg_data = knowledge_graph
    
    if not kg_data or not isinstance(kg_data, dict):
        print("No valid knowledge graph data to visualize.")
        return
    
    # Print entities
    print("=== ENTITIES ===")
    for entity in kg_data.get('entities', []):
        # Handle entity type which could be a string or a list
        entity_type = entity.get('type', 'Unknown')
        if isinstance(entity_type, list):
            entity_type = ', '.join(entity_type)
        
        # Get mentions as a comma-separated string
        mentions = ', '.join(entity.get('mentions', []))
        
        # Print entity info
        print(f"Entity: {entity.get('name', 'Unnamed')} (Type: {entity_type})")
        if mentions:
            print(f"  Mentions: {mentions}")
        
        # Print attributes if they exist
        attributes = entity.get('attributes', [])
        if attributes:
            print("  Attributes:")
            for attr in attributes:
                attr_name = attr.get('attribute', 'unknown')
                attr_value = attr.get('value', 'unknown')
                print(f"    {attr_name}: {attr_value}")
    
    # Print triples
    print("\n=== RELATIONSHIPS ===")
    for triple in kg_data.get('triples', []):
        subject_id = triple.get('subject', 'Unknown')
        predicate = triple.get('predicate', 'Unknown')
        object_id = triple.get('object', 'Unknown')
        
        # Try to resolve entity names if they're using IDs
        subject_name = resolve_entity_name(kg_data, subject_id)
        object_name = resolve_entity_name(kg_data, object_id)
        
        # Print the triple
        print(f"({subject_name}) --[{predicate}]--> ({object_name})")

def resolve_entity_name(kg_data, entity_id):
    """
    Resolve an entity ID to its name if possible.
    
    Args:
        kg_data: Knowledge graph dictionary
        entity_id: ID of the entity to resolve
        
    Returns:
        The entity name if found, otherwise the original ID
    """
    # If the entity_id doesn't look like an ID reference, return as is
    if not isinstance(entity_id, str) or not entity_id.startswith('e'):
        return entity_id
    
    # Look for the entity with this ID
    for entity in kg_data.get('entities', []):
        if entity.get('id') == entity_id:
            return entity.get('name', entity_id)
    
    # Return the original ID if not found
    return entity_id


import json
import uuid
import re
from typing import Dict, List, Optional

def format_context_for_llm(context_items: List[Dict]) -> str:
    """
    Format all context items into a single string for processing by the LLM.
    
    Args:
        context_items (List[Dict]): List of context items, each with 'title' and 'paragraphs'
        
    Returns:
        str: Formatted context string
    """
    formatted_context = []
    
    for i, item in enumerate(context_items):
        title = item.get('title', 'Untitled')
        paragraphs = item.get('paragraphs', [])
        
        # Format each context item with a clear separator and index
        formatted_item = f"CONTEXT ITEM #{i+1}:\n"
        formatted_item += f"Title: {title}\n"
        formatted_item += "Content: " + " ".join(paragraphs) + "\n"
        formatted_item += "-" * 50  # Add a separator
        
        formatted_context.append(formatted_item)
    
    # Join all formatted items with newlines
    return "\n\n".join(formatted_context)

def extract_json_content(text: str) -> Optional[Dict]:
    """
    Extract JSON content from the LLM response.
    
    Args:
        text (str): The text response from the LLM
        
    Returns:
        Optional[Dict]: The extracted JSON object or None if not found
    """
    # Regular expression to find JSON-like content between triple backticks
    json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    
    # Find all matches
    json_matches = re.finditer(json_pattern, text)
    
    for match in json_matches:
        try:
            # Extract the JSON content from the match
            json_str = match.group(1).strip()
            json_str = re.sub(r'//.*$', '', json_str, flags=re.MULTILINE)  # Remove comments
            
            # Parse the JSON string into a Python object
            json_obj = json.loads(json_str)
            return json_obj
        except json.JSONDecodeError:
            continue
    
    # If we didn't find JSON between backticks, try to extract a JSON object directly
    try:
        # Look for anything that looks like a complete JSON object
        json_pattern = r'\{[\s\S]*\}'
        match = re.search(json_pattern, text)
        if match:
            json_str = match.group(0)
            return json.loads(json_str)
    except:
        pass
    
    return None

def make_entity_ids_unique(triple_data: Dict) -> Dict:
    """
    Replace entity IDs in the triple data with unique identifiers.
    
    Args:
        triple_data (Dict): Dictionary containing entities and triples
        
    Returns:
        Dict: Updated dictionary with unique entity IDs
    """
    if not triple_data or 'entities' not in triple_data:
        return triple_data
    
    # Create ID mapping
    id_mapping = {}
    
    # Update entity IDs
    for entity in triple_data['entities']:
        if 'id' in entity:
            old_id = entity['id']
            new_id = f"entity_{str(uuid.uuid4())[:8]}"  # Using first 8 chars of UUID
            id_mapping[old_id] = new_id
            entity['id'] = new_id
    
    # Update references in triples if they exist
    if 'triples' in triple_data:
        for triple in triple_data['triples']:
            for field in ['subject', 'object']:
                if triple.get(field) in id_mapping:
                    triple[field] = id_mapping[triple[field]]
    
    return triple_data

def build_knowledge_graph(context_items: List[Dict], llm, prompt_template: str) -> Dict:
    """
    Build a knowledge graph from context items with a single LLM call.
    
    Args:
        context_items (List[Dict]): List of context items
        llm: Language model instance
        prompt_template: Prompt template for the LLM
        
    Returns:
        Dict: Knowledge graph with entities and triples
    """
    # Format all context items into a single string
    formatted_context = format_context_for_llm(context_items)
    
    # Create the prompt with all context items
    prompt = prompt_template.format(context_str=formatted_context)
    
    # Make a single call to the LLM
    llm_response = llm.complete(prompt)
    
    # Extract JSON content from the response
    knowledge_graph = extract_json_content(str(llm_response))
    
    # Make entity IDs unique
    if knowledge_graph:
        knowledge_graph = make_entity_ids_unique(knowledge_graph)
    
    return knowledge_graph

def process_contexts_in_batches(context_items: List[Dict], llm, prompt_template: str, batch_size: int = 3) -> Dict:
    """
    Process contexts in small batches and then merge the results.
    This is a fallback if single processing doesn't work well.
    
    Args:
        context_items: List of context items
        llm: Language model instance
        prompt_template: Prompt template for the LLM
        batch_size: Size of each batch
        
    Returns:
        Dict: Combined knowledge graph
    """
    # Split into batches
    batches = [context_items[i:i+batch_size] for i in range(0, len(context_items), batch_size)]
    
    all_entities = []
    all_triples = []
    entity_names = set()  # To track entities we've seen
    
    # Process each batch
    for batch in batches:
        kg = build_knowledge_graph(batch, llm, prompt_template)
        if not kg:
            continue
            
        # Add entities (avoiding duplicates by name)
        for entity in kg.get('entities', []):
            if entity['name'] not in entity_names:
                all_entities.append(entity)
                entity_names.add(entity['name'])
                
        # Add all triples
        for triple in kg.get('triples', []):
            all_triples.append(triple)
    
    # Combine results
    return {
        'entities': all_entities,
        'triples': all_triples
    }

def visualize_knowledge_graph(knowledge_graph: Dict) -> None:
    """
    Simple visualization of the knowledge graph in text format.
    
    Args:
        knowledge_graph (Dict): Knowledge graph with entities and triples
    """
    if not knowledge_graph:
        print("No knowledge graph data to visualize.")
        return
    
    # Print entities
    print("=== ENTITIES ===")
    for entity in knowledge_graph.get('entities', []):
        entity_type = entity.get('type', 'Unknown')
        mentions = ', '.join(entity.get('mentions', []))
        print(f"Entity: {entity.get('name', 'Unnamed')} (Type: {entity_type})")
        if mentions:
            print(f"  Mentions: {mentions}")
    
    # Print triples
    print("\n=== RELATIONSHIPS ===")
    for triple in knowledge_graph.get('triples', []):
        subject = triple.get('subject', 'Unknown')
        predicate = triple.get('predicate', 'Unknown')
        obj = triple.get('object', 'Unknown')
        source = triple.get('source_sentence', '')[:50] + ('...' if len(triple.get('source_sentence', '')) > 50 else '')
        print(f"({subject}) --[{predicate}]--> ({obj})")
        if source:
            print(f"  Source: {source}")
from rdflib import Graph, Literal, Namespace, RDF, URIRef, XSD
from typing import Dict, Any, List, Union
import re

def build_rdflib_knowledge_graph(kg_data: Dict[str, Any]) -> Graph:
    """
    Convert a dictionary-based knowledge graph to an RDFlib Graph.
    
    Args:
        kg_data (Dict[str, Any]): The knowledge graph data with 'entities' and 'triples'
        
    Returns:
        Graph: An RDFlib Graph object containing all the knowledge
    """
    # Create a new graph
    g = Graph()
    
    # Define namespaces
    KG = Namespace("http://example.org/kg/")
    ATTR = Namespace("http://example.org/kg/attribute/")
    g.bind("kg", KG)
    g.bind("attr", ATTR)
    
    # Maps to store entity URIs by name and ID
    entity_uri_by_name = {}
    entity_uri_by_id = {}
    
    # Process entities
    for entity in kg_data.get('entities', []):
        entity_id = entity.get('id')
        entity_name = entity.get('name')
        
        if not entity_name:
            continue
            
        # Create a valid URI from the entity name
        uri_name = create_valid_uri(entity_name)
        entity_uri = KG[uri_name]
        
        # Store mappings for later use
        entity_uri_by_name[entity_name] = entity_uri
        if entity_id:
            entity_uri_by_id[entity_id] = entity_uri
        
        # Add entity as a resource
        g.add((entity_uri, RDF.type, KG.Entity))
        
        # Add entity name as a literal
        g.add((entity_uri, KG.name, Literal(entity_name)))
        
        # Add entity types if available
        entity_types = entity.get('type', [])
        if isinstance(entity_types, str):
            entity_types = [entity_types]
            
        for entity_type in entity_types:
            if entity_type:
                type_uri = create_valid_uri(entity_type)
                g.add((entity_uri, KG.hasType, KG[type_uri]))
                g.add((KG[type_uri], RDF.type, KG.EntityType))
        
        # Add entity mentions if available
        for mention in entity.get('mentions', []):
            if mention and mention != entity_name:
                g.add((entity_uri, KG.hasMention, Literal(mention)))
        
        # Add entity attributes if available
        for attribute in entity.get('attributes', []):
            if 'attribute' in attribute and 'value' in attribute:
                attr_name = attribute['attribute']
                attr_value = attribute['value']
                
                # Create URI for the attribute
                attr_uri = ATTR[create_valid_uri(attr_name)]
                
                # Add the attribute value
                if isinstance(attr_value, bool):
                    g.add((entity_uri, attr_uri, Literal(attr_value, datatype=XSD.boolean)))
                elif isinstance(attr_value, int):
                    g.add((entity_uri, attr_uri, Literal(attr_value, datatype=XSD.integer)))
                elif isinstance(attr_value, float):
                    g.add((entity_uri, attr_uri, Literal(attr_value, datatype=XSD.float)))
                else:
                    g.add((entity_uri, attr_uri, Literal(str(attr_value))))
    
    # Process triples
    for triple in kg_data.get('triples', []):
        subject_name = triple.get('subject')
        predicate_name = triple.get('predicate')
        object_name = triple.get('object')
        
        if not (subject_name and predicate_name):
            continue
        
        # Get or create subject URI
        subject_uri = entity_uri_by_name.get(subject_name)
        if not subject_uri:
            uri_name = create_valid_uri(subject_name)
            subject_uri = KG[uri_name]
            entity_uri_by_name[subject_name] = subject_uri
            g.add((subject_uri, RDF.type, KG.Entity))
            g.add((subject_uri, KG.name, Literal(subject_name)))
        
        # Create predicate URI
        predicate_uri = KG[create_valid_uri(predicate_name)]
        
        # Handle the object based on its type and existence
        if object_name is None:
            continue
            
        if isinstance(object_name, bool):
            # Boolean value
            object_term = Literal(object_name, datatype=XSD.boolean)
        elif isinstance(object_name, int):
            # Integer value
            object_term = Literal(object_name, datatype=XSD.integer)
        elif isinstance(object_name, float):
            # Float value
            object_term = Literal(object_name, datatype=XSD.float)
        else:
            # Check if the object is an entity
            object_uri = entity_uri_by_name.get(object_name)
            if object_uri:
                object_term = object_uri
            else:
                # Determine if the object should be a Literal or URI
                if should_be_literal(object_name, predicate_name):
                    object_term = Literal(object_name)
                else:
                    # Create a new entity for the object
                    uri_name = create_valid_uri(object_name)
                    object_uri = KG[uri_name]
                    entity_uri_by_name[object_name] = object_uri
                    g.add((object_uri, RDF.type, KG.Entity))
                    g.add((object_uri, KG.name, Literal(object_name)))
                    object_term = object_uri
        
        # Add the main triple
        g.add((subject_uri, predicate_uri, object_term))
        
        # Add metadata if available
        if 'source_sentence_index' in triple:
            source_info = triple['source_sentence_index']
            if isinstance(source_info, list) and len(source_info) >= 2:
                source_title = source_info[0]
                source_index = source_info[1]
                
                # Create a blank node for the metadata
                from rdflib.term import BNode
                bnode = BNode()
                g.add((bnode, KG.sourceTitle, Literal(source_title)))
                g.add((bnode, KG.sourceIndex, Literal(source_index, datatype=XSD.integer)))
                g.add((subject_uri, KG.hasTripleSource, bnode))
        
        if 'category' in triple:
            category = triple['category']
            g.add((predicate_uri, KG.category, Literal(category)))
            
    return g

def create_valid_uri(text: Union[str, List]) -> str:
    """
    Create a valid URI string from text.
    
    Args:
        text (Union[str, List]): Text or list to convert to URI
        
    Returns:
        str: Valid URI string
    """
    if isinstance(text, list):
        text = '_'.join(str(item) for item in text)
    
    # Replace spaces and special characters with underscores
    valid_uri = re.sub(r'[^a-zA-Z0-9_]', '_', str(text))
    
    # Remove leading/trailing underscores
    valid_uri = valid_uri.strip('_')
    
    # Ensure URI doesn't start with a number (not allowed in XML)
    if valid_uri and valid_uri[0].isdigit():
        valid_uri = 'n' + valid_uri
        
    return valid_uri

def should_be_literal(object_value: str, predicate: str) -> bool:
    """
    Determine if an object should be represented as a Literal rather than a URI.
    
    Args:
        object_value (str): The object value to check
        predicate (str): The predicate that connects subject and object
        
    Returns:
        bool: True if the object should be a Literal, False otherwise
    """
    # Check predicates that usually connect to literal values
    literal_predicates = [
        'has', 'value', 'date', 'year', 'name', 'title', 'description', 
        'age', 'height', 'weight', 'price', 'cost', 'circulation', 'founded_on',
        'launched_in', 'started_as'
    ]
    
    if any(pred in predicate.lower() for pred in literal_predicates):
        return True
        
    # Check if the object looks like a date/year
    if re.match(r'\d{4}(-\d{2}){0,2}', str(object_value)):
        return True
        
    # Check if the object contains special characters typical for literals
    if ',' in str(object_value) or any(char in str(object_value) for char in '\'"():'):
        return True
        
    # Check if the object is a long text (more than 3 words)
    if len(str(object_value).split()) > 3:
        return True
        
    # Default to False (treat as URI)
    return False

def query_kg(g: Graph, query_string: str) -> list:
    """
    Run a SPARQL query on the knowledge graph.
    
    Args:
        g (Graph): The RDFlib Graph
        query_string (str): SPARQL query string
        
    Returns:
        list: Query results
    """
    results = g.query(query_string)
    return list(results)

def get_entity_info(g: Graph, entity_name: str) -> Dict:
    """
    Get all information about an entity from the graph.
    
    Args:
        g (Graph): The RDFlib Graph
        entity_name (str): Name of the entity to look up
        
    Returns:
        Dict: Information about the entity
    """
    # Create valid URI for entity
    KG = Namespace("http://example.org/kg/")
    uri_name = create_valid_uri(entity_name)
    entity_uri = KG[uri_name]
    
    # Get all triples where entity is subject
    outgoing = []
    for s, p, o in g.triples((entity_uri, None, None)):
        p_str = str(p).split('/')[-1]
        if isinstance(o, Literal):
            o_str = str(o)
        else:
            o_str = str(o).split('/')[-1].replace('_', ' ')
        outgoing.append((p_str, o_str))
    
    # Get all triples where entity is object
    incoming = []
    for s, p, o in g.triples((None, None, entity_uri)):
        p_str = str(p).split('/')[-1]
        if isinstance(s, Literal):
            s_str = str(s)
        else:
            s_str = str(s).split('/')[-1].replace('_', ' ')
        incoming.append((s_str, p_str))
    
    return {
        'entity': entity_name,
        'outgoing_relations': outgoing,
        'incoming_relations': incoming
    }

def print_graph_stats(g: Graph) -> None:
    """
    Print statistics about the knowledge graph.
    
    Args:
        g (Graph): The RDFlib Graph
    """
    # Count entities
    entity_query = """
    SELECT DISTINCT ?entity WHERE {
        ?entity a <http://example.org/kg/Entity> .
    }
    """
    entities = query_kg(g, entity_query)
    
    # Count relationships
    relation_query = """
    SELECT DISTINCT ?p WHERE {
        ?s ?p ?o .
        FILTER(!isBlank(?s) && !isBlank(?o))
        FILTER(?p != <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>)
    }
    """
    relations = query_kg(g, relation_query)
    
    # Count triples
    triple_count = len(g)
    
    print(f"Knowledge Graph Statistics:")
    print(f"  - Entities: {len(entities)}")
    print(f"  - Relation types: {len(relations)}")
    print(f"  - Total triples: {triple_count}")

