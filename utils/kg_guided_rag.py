import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Union, Any
import re
import json
from tqdm import tqdm
from rdflib import Graph, Namespace
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class KGGuidedRAG:
    """
    Implements KG-guided iterative RAG for multi-hop question answering.
    This enhancement addresses the limitations of the current RAG approach
    by integrating KG exploration to guide subsequent retrieval steps.
    """
    
    def __init__(self, llm, encoder_model="all-MiniLM-L6-v2", max_iterations=3):
        """
        Initialize the KG-guided RAG system.
        
        Args:
            llm: The language model to use for triple extraction and query generation
            encoder_model: The sentence transformer model for embeddings
            max_iterations: Maximum number of retrieval iterations
        """
        self.llm = llm
        self.encoder = SentenceTransformer(encoder_model)
        self.max_iterations = max_iterations
        self.kg_namespace = Namespace("http://example.org/kg/")
    
    def extract_json_content(self, text: str) -> Tuple[Dict, List[Dict]]:
        """
        Extract JSON content from LLM output.
        
        Args:
            text: The text containing JSON content
            
        Returns:
            Tuple containing the last JSON object and a list of all JSON objects
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
            # Find potential JSON blocks (text between { and })
            start_indices = [m.start() for m in re.finditer(r'\{', text)]
            
            for start_idx in start_indices:
                # Track nested braces
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
                        
                    # If found a matching closing brace
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
            return None, None
        
        return all_json_objects[-1], all_json_objects
    
    def extract_entities_from_question(self, question: str) -> List[str]:
        """
        Extract potential entity mentions from a question.
        
        Args:
            question: The question to extract entities from
            
        Returns:
            List of potential entity mentions
        """
        # Extract named entities (capitalized words and phrases)
        entity_candidates = []
        
        # Extract multi-word entities (capitalized phrases)
        multi_word_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b'
        multi_word_entities = re.findall(multi_word_pattern, question)
        entity_candidates.extend(multi_word_entities)
        
        # Extract single-word entities (capitalized words)
        single_word_pattern = r'\b[A-Z][a-z]+\b'
        single_word_entities = re.findall(single_word_pattern, question)
        # Remove single words that are part of multi-word entities
        for multi in multi_word_entities:
            for word in multi.split():
                if word in single_word_entities and word in entity_candidates:
                    entity_candidates.remove(word)
        entity_candidates.extend(single_word_entities)
        
        return entity_candidates
    
    def initial_retrieval(self, question: str, contexts: List[List[str]]) -> List[Dict]:
        """
        Perform initial retrieval based on semantic similarity.
        
        Args:
            question: The question to find contexts for
            contexts: The context items (title, paragraphs)
            
        Returns:
            List of relevant context items
        """
        # Encode the question
        question_embedding = self.encoder.encode(question, show_progress_bar=False)
        
        # Flatten contexts and calculate embeddings
        flat_contexts = []
        for title, paragraphs in contexts:
            for para in paragraphs:
                flat_contexts.append({
                    'title': title,
                    'text': para,
                    'full_context': (title, paragraphs)
                })
        
        # Calculate embeddings for all flattened contexts
        texts = [item['text'] for item in flat_contexts]
        context_embeddings = self.encoder.encode(texts, show_progress_bar=False)
        
        # Calculate similarity scores
        similarities = cosine_similarity(
            question_embedding.reshape(1, -1),
            context_embeddings
        )[0]
        
        # Sort contexts by similarity
        for i, score in enumerate(similarities):
            flat_contexts[i]['score'] = score
        
        sorted_contexts = sorted(flat_contexts, key=lambda x: x['score'], reverse=True)
        
        # Extract unique full contexts from top results (avoiding duplicates)
        top_contexts = []
        seen_titles = set()
        
        for ctx in sorted_contexts[:10]:  # Take top 10 results initially
            title = ctx['title']
            if title not in seen_titles:
                seen_titles.add(title)
                # Add the full context (title and all paragraphs)
                top_contexts.append({
                    'title': ctx['title'],
                    'paragraphs': ctx['full_context'][1],
                    'score': ctx['score']
                })
        
        return top_contexts
    
    def build_preliminary_kg(self, contexts: List[Dict], prompt_template) -> Dict:
        """
        Build a preliminary knowledge graph from retrieved contexts.
        
        Args:
            contexts: List of retrieved context items
            prompt_template: Template for context processing
            
        Returns:
            Knowledge graph with entities and triples
        """
        # Prepare the context string for the LLM
        context_str = ""
        for i, ctx in enumerate(contexts):
            title = ctx['title']
            paragraphs = ctx['paragraphs']
            
            # Format as expected by the existing template
            context_str += f"['{title}', {paragraphs}]\n\n"
        
        # Extract triples using the LLM
        prompt = prompt_template.format(context_str=context_str)
        response = self.llm.complete(prompt)
        
        # Extract JSON from response
        kg_data, _ = self.extract_json_content(str(response))
        
        return kg_data
    
    def identify_missing_entities(self, question: str, kg_data: Dict) -> List[str]:
        """
        Identify entities mentioned in the question but missing from the KG.
        
        Args:
            question: The question being processed
            kg_data: The current knowledge graph data
            
        Returns:
            List of potentially missing entities
        """
        # Extract potential entities from the question
        question_entities = self.extract_entities_from_question(question)
        
        # Get entities in the KG
        kg_entities = set()
        if kg_data and 'entities' in kg_data:
            for entity in kg_data['entities']:
                kg_entities.add(entity['name'].lower())
                # Add mentions as well
                for mention in entity.get('mentions', []):
                    kg_entities.add(mention.lower())
        
        # Find entities not in the KG
        missing_entities = []
        for entity in question_entities:
            if entity.lower() not in kg_entities:
                missing_entities.append(entity)
        
        return missing_entities
    
    def extract_relation_paths(self, kg_data: Dict) -> List[List[str]]:
        """
        Extract possible multi-hop relation paths from the KG.
        
        Args:
            kg_data: The knowledge graph data
            
        Returns:
            List of relation paths
        """
        # Extract entity relationships
        entity_relations = {}
        
        if not kg_data or 'entities' not in kg_data or 'triples' not in kg_data:
            return []
        
        # Build entity ID to name mapping
        entity_id_to_name = {}
        for entity in kg_data['entities']:
            if 'id' in entity and 'name' in entity:
                entity_id_to_name[entity['id']] = entity['name']
        
        # Build relation graph
        for triple in kg_data['triples']:
            subject = triple['subject']
            # If subject is an ID, convert to name
            if subject in entity_id_to_name:
                subject = entity_id_to_name[subject]
                
            object_ = triple['object']
            # If object is an ID, convert to name
            if object_ in entity_id_to_name:
                object_ = entity_id_to_name[object_]
                
            predicate = triple['predicate']
            
            # Add to relations dict
            if subject not in entity_relations:
                entity_relations[subject] = []
            entity_relations[subject].append((predicate, object_))
        
        # Extract all possible 2-hop paths
        two_hop_paths = []
        for start_entity in entity_relations:
            for pred1, mid_entity in entity_relations.get(start_entity, []):
                for pred2, end_entity in entity_relations.get(mid_entity, []):
                    two_hop_paths.append([start_entity, pred1, mid_entity, pred2, end_entity])
        
        return two_hop_paths
    
    def generate_kg_guided_query(self, question: str, kg_data: Dict, missing_entities: List[str], relation_paths: List[List[str]]) -> str:
        """
        Generate a KG-guided query for the next retrieval iteration.
        
        Args:
            question: The original question
            kg_data: The current knowledge graph
            missing_entities: Entities missing from the KG
            relation_paths: Relation paths extracted from the KG
            
        Returns:
            An enhanced query string
        """
        # Extract entities from KG
        kg_entities = []
        if kg_data and 'entities' in kg_data:
            for entity in kg_data['entities']:
                kg_entities.append(entity['name'])
        
        # Prepare a prompt for the LLM to generate an enhanced query
        prompt = f"""
        Based on the following information, generate an enhanced search query to find missing information needed to answer a multi-hop question.
        
        Original question: "{question}"
        
        Entities already found in knowledge graph: {', '.join(kg_entities[:10])}
        
        Potential missing entities from the question: {', '.join(missing_entities)}
        
        Most relevant 2-hop relation paths found:
        {relation_paths[:3]}
        
        Your task is to create a search query that will help find information about the missing entities and 
        any connections needed to complete the reasoning chain. Focus on what information is still needed to answer the question.
        
        Enhanced search query:
        """
        
        # Generate the enhanced query
        response = self.llm.complete(prompt)
        
        # Extract just the query part
        enhanced_query = str(response).strip()
        
        # Combine with original question
        final_query = f"{question} {enhanced_query}"
        
        return final_query
    
    def retrieve_additional_contexts(self, query: str, contexts: List[List[str]], already_retrieved: List[Dict]) -> List[Dict]:
        """
        Retrieve additional contexts based on the enhanced query.
        
        Args:
            query: The enhanced query
            contexts: The full context list
            already_retrieved: Contexts already retrieved
            
        Returns:
            New relevant contexts
        """
        # Encode the enhanced query
        query_embedding = self.encoder.encode(query, show_progress_bar=False)
        
        # Track titles already retrieved
        already_retrieved_titles = set(ctx['title'] for ctx in already_retrieved)
        
        # Flatten contexts and calculate embeddings
        flat_contexts = []
        for title, paragraphs in contexts:
            # Skip already retrieved contexts
            if title in already_retrieved_titles:
                continue
                
            for para in paragraphs:
                flat_contexts.append({
                    'title': title,
                    'text': para,
                    'full_context': (title, paragraphs)
                })
        
        # If all contexts have been retrieved, return empty list
        if not flat_contexts:
            return []
        
        # Calculate embeddings for all flattened contexts
        texts = [item['text'] for item in flat_contexts]
        context_embeddings = self.encoder.encode(texts, show_progress_bar=False)
        
        # Calculate similarity scores
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1),
            context_embeddings
        )[0]
        
        # Sort contexts by similarity
        for i, score in enumerate(similarities):
            flat_contexts[i]['score'] = score
        
        sorted_contexts = sorted(flat_contexts, key=lambda x: x['score'], reverse=True)
        
        # Extract unique full contexts from top results (avoiding duplicates)
        new_contexts = []
        seen_titles = set()
        
        for ctx in sorted_contexts[:5]:  # Take top 5 new results
            title = ctx['title']
            if title not in seen_titles:
                seen_titles.add(title)
                # Add the full context (title and all paragraphs)
                new_contexts.append({
                    'title': ctx['title'],
                    'paragraphs': ctx['full_context'][1],
                    'score': ctx['score']
                })
        
        return new_contexts
    
    def build_rdflib_graph(self, kg_data: Dict) -> Graph:
        """
        Convert a dictionary-based knowledge graph to an RDFlib Graph.
        
        Args:
            kg_data: The knowledge graph data
            
        Returns:
            An RDFlib Graph object
        """
        from rdflib import Graph, Literal, Namespace, RDF, URIRef
        
        # Create a new graph
        g = Graph()
        
        # Define namespace
        KG = self.kg_namespace
        g.bind("kg", KG)
        
        # Process entities
        entity_uri_by_name = {}
        
        if not kg_data or 'entities' not in kg_data:
            return g
            
        for entity in kg_data['entities']:
            entity_name = entity.get('name')
            
            if not entity_name:
                continue
                
            # Create a valid URI from the entity name
            uri_name = self._create_valid_uri(entity_name)
            entity_uri = KG[uri_name]
            
            # Store mapping for later use
            entity_uri_by_name[entity_name] = entity_uri
            
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
                    type_uri = self._create_valid_uri(entity_type)
                    g.add((entity_uri, KG.hasType, KG[type_uri]))
            
            # Add entity mentions if available
            for mention in entity.get('mentions', []):
                if mention and mention != entity_name:
                    g.add((entity_uri, KG.hasMention, Literal(mention)))
        
        # Process triples
        if 'triples' in kg_data:
            for triple in kg_data['triples']:
                subject_name = triple.get('subject')
                predicate_name = triple.get('predicate')
                object_name = triple.get('object')
                
                if not (subject_name and predicate_name):
                    continue
                
                # Get or create subject URI
                subject_uri = entity_uri_by_name.get(subject_name)
                if not subject_uri:
                    uri_name = self._create_valid_uri(subject_name)
                    subject_uri = KG[uri_name]
                    entity_uri_by_name[subject_name] = subject_uri
                    g.add((subject_uri, RDF.type, KG.Entity))
                    g.add((subject_uri, KG.name, Literal(subject_name)))
                
                # Create predicate URI
                predicate_uri = KG[self._create_valid_uri(predicate_name)]
                
                # Handle the object
                if object_name is None:
                    continue
                    
                # Check if the object is an entity
                object_uri = entity_uri_by_name.get(object_name)
                if object_uri:
                    object_term = object_uri
                else:
                    object_term = Literal(object_name)
                
                # Add the triple
                g.add((subject_uri, predicate_uri, object_term))
        
        return g
    
    def _create_valid_uri(self, text):
        """Create a valid URI string from text."""
        if isinstance(text, list):
            text = '_'.join(str(item) for item in text)
        
        # Replace spaces and special characters with underscores
        valid_uri = re.sub(r'[^a-zA-Z0-9_]', '_', str(text))
        
        # Remove leading/trailing underscores
        valid_uri = valid_uri.strip('_')
        
        # Ensure URI doesn't start with a number
        if valid_uri and valid_uri[0].isdigit():
            valid_uri = 'n' + valid_uri
            
        return valid_uri
    
    def analyze_kg_coverage(self, question: str, kg_graph: Graph) -> float:
        """
        Analyze how well the KG covers the question entities and relations.
        
        Args:
            question: The question being processed
            kg_graph: The knowledge graph
            
        Returns:
            Coverage score (0-1)
        """
        # Extract entities from question
        question_entities = self.extract_entities_from_question(question)
        
        # Check how many entities are in the KG
        covered_entities = 0
        
        # Get all entity names from the graph
        entity_names = set()
        for s, p, o in kg_graph.triples((None, self.kg_namespace.name, None)):
            entity_names.add(str(o).lower())
        
        # Check coverage
        for entity in question_entities:
            if entity.lower() in entity_names:
                covered_entities += 1
        
        # Calculate coverage score
        if not question_entities:
            return 1.0  # No entities to cover
            
        coverage_score = covered_entities / len(question_entities)
        
        return coverage_score
    
    def rank_passages_by_graph_centrality(self, kg_graph: Graph, contexts: List[Dict]) -> List[Dict]:
        """
        Rank passages by their centrality in the knowledge graph.
        
        Args:
            kg_graph: The knowledge graph
            contexts: The retrieved contexts
            
        Returns:
            Contexts ranked by graph centrality
        """
        # Get all entity names from the graph
        entity_names = set()
        for s, p, o in kg_graph.triples((None, self.kg_namespace.name, None)):
            entity_names.add(str(o).lower())
        
        # Count entity mentions in each context
        for ctx in contexts:
            entity_count = 0
            relation_count = 0
            
            # Check title
            for entity in entity_names:
                if entity in ctx['title'].lower():
                    entity_count += 1
            
            # Check paragraphs
            for para in ctx['paragraphs']:
                for entity in entity_names:
                    if entity in para.lower():
                        entity_count += 1
            
            # Calculate a centrality score
            ctx['centrality_score'] = entity_count
            
            # Combine with similarity score
            ctx['combined_score'] = (ctx['score'] + ctx['centrality_score']) / 2
        
        # Rank by combined score
        ranked_contexts = sorted(contexts, key=lambda x: x['combined_score'], reverse=True)
        
        return ranked_contexts
    
    def iterative_retrieval(self, 
                           question: str, 
                           contexts: List[List[str]], 
                           prompt_template) -> Tuple[List[Dict], Dict]:
        """
        Perform iterative KG-guided retrieval.
        
        Args:
            question: The question to answer
            contexts: The full set of available contexts
            prompt_template: Template for context processing
            
        Returns:
            Tuple of (final ranked contexts, final knowledge graph)
        """
        # Step 1: Initial retrieval based on semantic similarity
        retrieved_contexts = self.initial_retrieval(question, contexts)
        
        # Step 2: Build preliminary knowledge graph
        kg_data = self.build_preliminary_kg(retrieved_contexts, prompt_template)
        
        # Convert to RDFlib Graph
        kg_graph = self.build_rdflib_graph(kg_data)
        
        # Iterative refinement
        for iteration in range(self.max_iterations):
            # Step 3: Identify missing entities in KG
            missing_entities = self.identify_missing_entities(question, kg_data)
            
            # If no missing entities and good coverage, stop iteration
            coverage = self.analyze_kg_coverage(question, kg_graph)
            if not missing_entities and coverage > 0.8:
                break
            
            # Step 4: Extract relation paths from KG
            relation_paths = self.extract_relation_paths(kg_data)
            
            # Step 5: Generate KG-guided query
            enhanced_query = self.generate_kg_guided_query(
                question, kg_data, missing_entities, relation_paths
            )
            
            # Step 6: Retrieve additional contexts
            new_contexts = self.retrieve_additional_contexts(
                enhanced_query, contexts, retrieved_contexts
            )
            
            # If no new contexts, stop iteration
            if not new_contexts:
                break
                
            # Step 7: Update retrieved contexts
            retrieved_contexts.extend(new_contexts)
            
            # Step 8: Update knowledge graph with new contexts
            new_kg_data = self.build_preliminary_kg(new_contexts, prompt_template)
            
            # Merge KG data
            if new_kg_data and 'entities' in new_kg_data and 'triples' in new_kg_data:
                # Merge entities (avoiding duplicates by name)
                entity_names = set(e['name'] for e in kg_data.get('entities', []))
                for entity in new_kg_data['entities']:
                    if entity['name'] not in entity_names:
                        kg_data.setdefault('entities', []).append(entity)
                        entity_names.add(entity['name'])
                
                # Merge triples (simply append)
                kg_data.setdefault('triples', []).extend(new_kg_data.get('triples', []))
                
                # Update RDFlib Graph
                kg_graph = self.build_rdflib_graph(kg_data)
        
        # Step 9: Rank passages by graph centrality
        ranked_contexts = self.rank_passages_by_graph_centrality(kg_graph, retrieved_contexts)
        
        return ranked_contexts, kg_data
    
    def extract_supporting_facts(self, ranked_contexts: List[Dict]) -> List[List]:
        """
        Extract supporting facts from ranked contexts.
        
        Args:
            ranked_contexts: The ranked context items
            
        Returns:
            List of supporting facts in the format [title, paragraph_index]
        """
        supporting_facts = []
        
        for ctx in ranked_contexts:
            title = ctx['title']
            
            for i, para in enumerate(ctx['paragraphs']):
                # Add all paragraphs as potential supporting facts
                supporting_facts.append([title, i])
        
        return supporting_facts

def kg_guided_rag_generate(dataset, llm, prompt_template):
    """
    Generate contexts using KG-guided RAG for a dataset.
    
    Args:
        dataset: Dataset containing questions and contexts
        llm: Language model instance
        prompt_template: Template for context processing
        
    Returns:
        DataFrame with retrieved contexts and supporting facts
    """
    # Initialize KG-guided RAG
    kg_rag = KGGuidedRAG(llm)
    
    # Initialize result dataframe
    coverage = pd.DataFrame(columns=[
        'Question', 'gold_answer', 'Supporting Facts', 
        'SPARQL query', 'Retrieval Facts', 'Retrieval Result',
        'coverage_all', 'coverage_rate'
    ])
    
    # Process each example
    for data_row in tqdm(dataset, desc="Processing dataset"):
        question = data_row['question']
        contexts = data_row['context']
        
        try:
            # Perform iterative KG-guided retrieval
            ranked_contexts, kg_data = kg_rag.iterative_retrieval(
                question, contexts, prompt_template
            )
            
            # Extract supporting facts
            retrieved_facts = kg_rag.extract_supporting_facts(ranked_contexts)
            
            # Format as expected by downstream processing
            passages_list = [
                {
                    'title': ctx['title'],
                    'paragraphs': ctx['paragraphs']
                }
                for ctx in ranked_contexts
            ]
            
            # Determine coverage
            gold_facts = data_row['supporting_facts']
            all_covered = all(sf in retrieved_facts for sf in gold_facts)
            
            # Calculate coverage rate
            total_facts = sum(len(context[1]) for context in data_row['context'])
            coverage_rate = len(retrieved_facts) / total_facts if total_facts > 0 else 0
            
            # Get SPARQL query if available
            sparql_query = ""
            if kg_data and 'sparql_pattern' in kg_data:
                sparql_query = kg_data['sparql_pattern']
            
            # Create row for dataframe
            new_row = pd.DataFrame({
                'Question': [question],
                'gold_answer': [data_row['answer']],
                'Supporting Facts': [data_row['supporting_facts']],
                'Retrieval Facts': [retrieved_facts],
                'SPARQL query': [sparql_query],
                'Retrieval Result': [passages_list],
                'coverage_all': ['1' if all_covered else '0'],
                'coverage_rate': [coverage_rate]
            })
            
            # Append to results
            coverage = pd.concat([coverage, new_row], ignore_index=True)
            
        except Exception as e:
            # Handle errors
            print(f"Error processing question: {question}")
            print(f"Error: {str(e)}")
            
            # Add row with empty retrieval
            new_row = pd.DataFrame({
                'Question': [question],
                'gold_answer': [data_row['answer']],
                'Supporting Facts': [data_row['supporting_facts']],
                'SPARQL query': [''],
                'Retrieval Facts': [[]],
                'Retrieval Result': [[]],
                'coverage_all': ['0'],
                'coverage_rate': [0.0]
            })
            
            coverage = pd.concat([coverage, new_row], ignore_index=True)
    
    return coverage