import re
from typing import List, Dict, Any, Tuple, Optional
from rdflib import Graph, Namespace, URIRef

class KGAwareQueryGenerator:
    """
    Enhanced query generation that first explores the KG structure
    and adapts prompts based on actual available entities and relationships.
    """
    
    def __init__(self):
        """Initialize the query generator."""
        self.kg_namespace = None
    
    def explore_kg_for_entities(self, kg, question):
        """
        Explore the KG to find entities related to the question.
        
        Args:
            kg: RDFlib Graph object
            question: The question to answer
            
        Returns:
            Dictionary with entity exploration results
        """
        # Set namespace if not already set
        if not self.kg_namespace:
            for prefix, namespace in kg.namespaces():
                if prefix == "kg":
                    self.kg_namespace = namespace
                    break
                    
            if not self.kg_namespace:
                self.kg_namespace = Namespace("http://example.org/kg/")
        
        # Extract potential entities from the question
        entities = self._extract_entity_candidates(question)
        
        # Find these entities in the KG
        entity_findings = {}
        for entity in entities:
            # Look for entities with similar names
            entity_info = self._find_entity_in_kg(kg, entity)
            if entity_info:
                entity_findings[entity] = entity_info
        
        # Get information about the found entities
        entity_details = {}
        for entity, info in entity_findings.items():
            uri = info['uri']
            # Get all predicates and objects for this entity
            triples = list(kg.triples((uri, None, None)))
            # Get all predicates and subjects for this entity
            triples_reverse = list(kg.triples((None, None, uri)))
            
            entity_details[entity] = {
                'uri': uri,
                'name': info['name'],
                'type': info.get('type'),
                'outgoing_relations': [self._format_triple(triple) for triple in triples],
                'incoming_relations': [self._format_triple(triple, reverse=True) for triple in triples_reverse]
            }
        
        # Get all available predicates (relationships) in the KG
        predicates = set()
        for s, p, o in kg:
            p_str = str(p)
            if p_str.startswith(str(self.kg_namespace)):
                pred_name = p_str.split('/')[-1]
                predicates.add(pred_name)
        
        # Get all available entity types in the KG
        entity_types = set()
        type_predicate = URIRef(str(self.kg_namespace) + "hasType")
        for s, p, o in kg.triples((None, type_predicate, None)):
            o_str = str(o)
            if o_str.startswith(str(self.kg_namespace)):
                type_name = o_str.split('/')[-1]
                entity_types.add(type_name)
        
        return {
            'entity_details': entity_details,
            'all_predicates': sorted(list(predicates)),
            'all_entity_types': sorted(list(entity_types)),
            'question_entities': entities
        }
    
    def _extract_entity_candidates(self, question):
        """Extract potential entity names from the question."""
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
                if word in single_word_entities:
                    single_word_entities.remove(word)
        entity_candidates.extend(single_word_entities)
        
        # Also extract domain-specific terms even if not capitalized
        domain_terms = ["hotel", "company", "family", "office", "head office", "headquarter", "city"]
        for term in domain_terms:
            if term in question.lower() and term not in entity_candidates:
                entity_candidates.append(term)
        
        return entity_candidates
    
    def _find_entity_in_kg(self, kg, entity_name):
        """
        Find an entity in the KG by name.
        
        Args:
            kg: RDFlib Graph object
            entity_name: The name of the entity to find
            
        Returns:
            Dictionary with entity information or None if not found
        """
        # Try exact name match
        name_predicate = URIRef(str(self.kg_namespace) + "name")
        for s, p, o in kg.triples((None, name_predicate, None)):
            o_str = str(o)
            if entity_name.lower() == o_str.lower():
                # Get the entity type if available
                entity_type = None
                type_predicate = URIRef(str(self.kg_namespace) + "hasType")
                for _, _, t in kg.triples((s, type_predicate, None)):
                    type_uri = str(t)
                    if type_uri.startswith(str(self.kg_namespace)):
                        entity_type = type_uri.split('/')[-1]
                        break
                
                return {
                    'uri': s,
                    'name': o_str,
                    'type': entity_type
                }
        
        # Try partial name match
        for s, p, o in kg.triples((None, name_predicate, None)):
            o_str = str(o)
            if entity_name.lower() in o_str.lower() or o_str.lower() in entity_name.lower():
                # Get the entity type if available
                entity_type = None
                type_predicate = URIRef(str(self.kg_namespace) + "hasType")
                for _, _, t in kg.triples((s, type_predicate, None)):
                    type_uri = str(t)
                    if type_uri.startswith(str(self.kg_namespace)):
                        entity_type = type_uri.split('/')[-1]
                        break
                
                return {
                    'uri': s,
                    'name': o_str,
                    'type': entity_type
                }
        
        return None
    
    def _format_triple(self, triple, reverse=False):
        """Format a triple for human-readable output."""
        s, p, o = triple
        
        # Format predicate
        p_str = str(p)
        if p_str.startswith(str(self.kg_namespace)):
            p_name = p_str.split('/')[-1]
        else:
            p_name = p_str
        
        # Format object or subject (depending on reverse)
        if reverse:
            node = s
        else:
            node = o
        
        node_str = str(node)
        if node_str.startswith(str(self.kg_namespace)):
            node_name = node_str.split('/')[-1]
        else:
            # Try to get a name if it's an entity
            name_predicate = URIRef(str(self.kg_namespace) + "name")
            if not reverse:  # If node is an object
                for _, _, name in Graph().triples((node, name_predicate, None)):
                    node_name = str(name)
                    break
                else:
                    node_name = node_str
            else:  # If node is a subject
                for _, _, name in Graph().triples((node, name_predicate, None)):
                    node_name = str(name)
                    break
                else:
                    node_name = node_str
        
        if reverse:
            return {"subject": node_name, "predicate": p_name}
        else:
            return {"predicate": p_name, "object": node_name}
    
    def generate_initial_exploration_prompt(self, question, kg_exploration):
        """
        Generate a prompt that first explores the KG to understand available entities and relationships.
        
        Args:
            question: The question to answer
            kg_exploration: Results from exploring the KG for relevant entities
            
        Returns:
            A prompt string for the LLM
        """
        # Format entity details
        entity_details_str = ""
        for entity, details in kg_exploration['entity_details'].items():
            entity_details_str += f"\n### Entity: {entity} (KG name: {details['name']})\n"
            entity_details_str += f"Type: {details['type']}\n"
            
            # Outgoing relations
            entity_details_str += "Outgoing relations:\n"
            for rel in details['outgoing_relations']:
                entity_details_str += f"- {rel['predicate']} -> {rel['object']}\n"
            
            # Incoming relations
            entity_details_str += "Incoming relations:\n"
            for rel in details['incoming_relations']:
                entity_details_str += f"- {rel['subject']} -> {rel['predicate']}\n"
        
        # Format all predicates and entity types
        predicates_str = ", ".join(kg_exploration['all_predicates'])
        entity_types_str = ", ".join(kg_exploration['all_entity_types'])
        
        prompt = f"""
        # Knowledge Graph Query Generation with Exploration
        
        I need to create a SPARQL query to answer a question. First, let me analyze what's available in the knowledge graph.
        
        ## Question
        "{question}"
        
        ## Knowledge Graph Exploration Results
        
        ### Entities Found in KG Related to Question
        {entity_details_str}
        
        ### All Available Predicates in KG
        {predicates_str}
        
        ### All Available Entity Types in KG
        {entity_types_str}
        
        ## Task: Generate a SPARQL query to answer the question
        
        Given the knowledge graph exploration above, follow these steps:
        
        ### Step 1: Analyze the question
        - What specific information is the question asking for?
        - Which entities from the KG exploration are relevant?
        
        ### Step 2: Identify the query structure
        - What patterns of triples will connect the relevant entities?
        - What variables do we need to include in the query?
        
        ### Step 3: Verify available predicates
        - Make sure to ONLY use predicates that were found in the KG
        - Do NOT use RDF/RDFS default predicates (like rdf:type) - use the kg: namespace predicates instead
        - Check if the relationships needed actually exist in the exploration results
        
        ### Step 4: Formulate the query
        - Build a syntactically correct SPARQL query
        - Start with the PREFIX declaration
        - Make sure all braces are balanced and properly formatted
        - Use kg:hasType instead of rdf:type for type declarations
        
        ### Step 5: Final review
        - Ensure you only use predicates and types that actually exist in the KG
        - Make sure the query is not too restrictive
        - Verify syntax is correct
        
        Generate ONLY the SPARQL query without any explanation. The query must:
        - Start with PREFIX kg: <http://example.org/kg/>
        - Only use predicates that were found in the KG
        - Use kg:hasType for type constraints, not rdf:type
        - Have all braces properly balanced
        """
        
        return prompt
    
    def generate_refinement_with_kg_awareness_prompt(self, question, previous_query, kg_exploration, error_info, failed_attempts):
        """
        Generate a refinement prompt with detailed KG awareness and error analysis.
        
        Args:
            question: The question to answer
            previous_query: The previous query that failed
            kg_exploration: Results from exploring the KG
            error_info: Error information from previous query
            failed_attempts: List of previously failed queries
            
        Returns:
            A prompt string for the LLM
        """
        # Format entity details
        entity_details_str = ""
        for entity, details in kg_exploration['entity_details'].items():
            entity_details_str += f"\n### Entity: {entity} (KG name: {details['name']})\n"
            entity_details_str += f"Type: {details['type']}\n"
            
            # Outgoing relations
            entity_details_str += "Outgoing relations:\n"
            for rel in details['outgoing_relations']:
                entity_details_str += f"- {rel['predicate']} -> {rel['object']}\n"
            
            # Incoming relations
            entity_details_str += "Incoming relations:\n"
            for rel in details['incoming_relations']:
                entity_details_str += f"- {rel['subject']} -> {rel['predicate']}\n"
        
        # Format all predicates and entity types
        predicates_str = ", ".join(kg_exploration['all_predicates'])
        entity_types_str = ", ".join(kg_exploration['all_entity_types'])
        
        # Format failed attempts
        failed_attempts_str = ""
        for i, query in enumerate(failed_attempts):
            failed_attempts_str += f"\n### Failed Query {i+1}:\n```\n{query}\n```\n"
        
        prompt = f"""
        # SPARQL Query Refinement with KG Awareness
        
        I need to refine a SPARQL query that didn't return results.
        
        ## Question
        "{question}"
        
        ## Previous Failed Query
        ```
        {previous_query}
        ```
        
        ## Error Information
        {error_info}
        
        ## All Previously Failed Attempts
        {failed_attempts_str}
        
        ## Knowledge Graph Exploration Results
        
        ### Entities Found in KG Related to Question
        {entity_details_str}
        
        ### All Available Predicates in KG
        {predicates_str}
        
        ### All Available Entity Types in KG
        {entity_types_str}
        
        ## Task: Refine the SPARQL query based on KG exploration
        
        ### Step 1: Analyze why the previous query failed
        - Did it use predicates that don't exist in the KG?
        - Were the entity names or URIs incorrect?
        - Was the query too restrictive?
        - Were there syntax errors?
        
        ### Step 2: Plan a new approach using available data
        - Which entities and relationships from the KG exploration can connect to the answer?
        - Is there an alternative path to get the information?
        - Can we make the query more general?
        
        ### Step 3: Modify the query
        - ONLY use predicates that actually exist in the KG exploration
        - Pay special attention to how entities and types are referenced
        - Use kg:hasType for type constraints, not rdf:type
        - Make sure all braces are balanced
        
        ### Step 4: Consider simplifying
        - Sometimes a simpler query that returns more results is better
        - Focus on the most essential patterns
        
        Generate ONLY the refined SPARQL query without any explanation. The query must:
        - Start with PREFIX kg: <http://example.org/kg/>
        - Only use predicates that were found in the KG
        - Use kg:hasType for type constraints, not rdf:type
        - Have all braces properly balanced
        """
        
        return prompt
    
    def generate_context_analysis_prompt(self, question, kg_exploration, context_data):
        """
        Generate a prompt for analyzing context data when KG queries fail.
        
        Args:
            question: The question to answer
            kg_exploration: Results from exploring the KG
            context_data: Context information
            
        Returns:
            A prompt string for the LLM
        """
        # Format entity details
        entity_details_str = ""
        for entity, details in kg_exploration['entity_details'].items():
            entity_details_str += f"\n### Entity: {entity} (KG name: {details['name']})\n"
            entity_details_str += f"Type: {details['type']}\n"
            
            # Limit relations for readability
            outgoing = details['outgoing_relations'][:5]
            incoming = details['incoming_relations'][:5]
            
            # Outgoing relations
            entity_details_str += "Outgoing relations:\n"
            for rel in outgoing:
                entity_details_str += f"- {rel['predicate']} -> {rel['object']}\n"
            
            # Incoming relations
            entity_details_str += "Incoming relations:\n"
            for rel in incoming:
                entity_details_str += f"- {rel['subject']} -> {rel['predicate']}\n"
        
        # Format context data
        context_str = self._format_context_data(context_data)
        
        prompt = f"""
        # Question Answering with KG Exploration and Context Analysis
        
        The SPARQL queries failed to return results, so we'll use the KG entity information along with the context to answer the question.
        
        ## Question
        "{question}"
        
        ## Knowledge Graph Entity Information
        {entity_details_str}
        
        ## Context Information
        {context_str}
        
        ## Task: Use both KG exploration results and context to answer the question
        
        Think through this step by step:
        
        ### Step 1: Analyze what information we have from the KG exploration
        - What entities were found in the KG?
        - What relationships do they have?
        - How does this relate to the question?
        
        ### Step 2: Identify relevant information in the context
        - Which parts of the context mention entities from the KG exploration?
        - What additional information does the context provide?
        - Quote the specific relevant passages from the context.
        
        ### Step 3: Connect KG information with context information
        - How does the KG structure relate to the information in the context?
        - Do they complement each other?
        - How can we use both to answer the question?
        
        ### Step 4: Answer the question with reasoning
        - Provide a clear, concise answer based on both KG and context
        - Explain how you arrived at this answer
        - Support your answer with specific evidence from both KG and context
        
        ### Final Answer
        Based on your analysis, provide your final answer as a single word or short phrase 
        enclosed in <ANSWER> tags like this: <ANSWER>your answer</ANSWER>
        """
        
        return prompt
    
    def _format_context_data(self, context_data):
        """Format context data for the prompt."""
        if isinstance(context_data, list):
            # Format a list of context items
            context_parts = []
            for i, item in enumerate(context_data):
                if isinstance(item, dict) and 'title' in item and 'paragraphs' in item:
                    title = item['title']
                    content = " ".join(item['paragraphs'])
                    context_parts.append(f"### Source {i+1}: {title}\n{content}")
                else:
                    context_parts.append(f"### Source {i+1}\n{str(item)}")
            
            return "\n\n".join(context_parts)
        else:
            # Handle other formats
            return str(context_data)
    
    def extract_answer_from_llm_response(self, response_text, max_length=1000):
        """
        Extract a concise answer from an LLM response, prioritizing tagged formats.
        
        Args:
            response_text: The text response from the LLM
            max_length: Maximum length for the extracted answer
            
        Returns:
            The extracted answer or None if not found
        """
        # Convert response to string if needed
        if not isinstance(response_text, str):
            if hasattr(response_text, 'text'):
                response_text = response_text.text
            elif hasattr(response_text, 'content'):
                response_text = response_text.content
            else:
                response_text = str(response_text)
        
        # Clean up the response
        response_text = response_text.strip()
        
        import re
        
        # Look for tagged answer format first (highest priority)
        tag_patterns = [
            r'<ANSWER>(.*?)</ANSWER>',
            r'Answer:\s*(.*?)(?:\n|$)',
            r'Final Answer:\s*(.*?)(?:\n|$)',
        ]
        
        for pattern in tag_patterns:
            match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
                # Return only the first sentence if the answer is too long
                if len(answer) > max_length:
                    sentences = re.split(r'[.!?]\s+', answer)
                    if sentences:
                        return sentences[0].strip()
                return answer
        
        # If no tags found, use existing methods
        # 1. Try other common patterns
        common_patterns = [
            r'(?:Therefore,|Thus,|In conclusion,|The answer is|Hence,)\s*(.*?)(?:\n|$)',
        ]
        
        for pattern in common_patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # 2. Look for the last paragraph if it's short
        paragraphs = [p.strip() for p in response_text.split("\n\n") if p.strip()]
        if paragraphs and len(paragraphs[-1]) <= max_length:
            return paragraphs[-1]
        
        # 3. Take the last sentence if reasonable
        sentences = [s.strip() for s in re.split(r'[.!?]\s+', response_text) if s.strip()]
        if sentences and len(sentences[-1]) <= max_length:
            return sentences[-1]
        
        # Fallback: Return the first short sentence
        for sentence in sentences:
            if len(sentence) <= max_length:
                return sentence
        
        # Last resort: truncate the first sentence
        if sentences:
            return sentences[0][:max_length].strip()
        
        return "No answer found"

def process_question_with_kg_awareness(kg, context_data, question, llm, max_iterations=4):
    """
    Process a question with KG-aware exploration and refinement.
    
    Args:
        kg: RDFlib Graph object
        context_data: Context information
        question: Question to answer
        llm: Language model
        max_iterations: Maximum number of refinement iterations
        
    Returns:
        Dictionary with the answer, process details, and full LLM responses
    """
    import re
    
    print(f"Processing question: {question}")
    
    # Initialize query generator
    query_generator = KGAwareQueryGenerator()
    
    # Step 1: Explore the KG to find relevant entities and relationships
    print("Exploring knowledge graph for relevant entities...")
    kg_exploration = query_generator.explore_kg_for_entities(kg, question)
    
    print(f"Found {len(kg_exploration['entity_details'])} entities related to the question")
    print(f"Available predicates: {', '.join(kg_exploration['all_predicates'][:5])}...")
    print(f"Available entity types: {', '.join(kg_exploration['all_entity_types'][:5])}...")
    
    # Function to extract SPARQL from response
    def extract_sparql_from_response(response):
        if hasattr(response, 'text'):
            response_text = response.text
        elif hasattr(response, 'content'):
            response_text = response.content
        else:
            response_text = str(response)
            
        # Try to find SPARQL with PREFIX
        sparql_pattern = r'PREFIX\s+kg:\s+<[^>]+>\s*SELECT\s+.*?WHERE\s*\{.*?\}'
        match = re.search(sparql_pattern, response_text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(0)
            
        # Try without PREFIX if not found
        sparql_pattern = r'SELECT\s+.*?WHERE\s*\{.*?\}'
        match = re.search(sparql_pattern, response_text, re.DOTALL | re.IGNORECASE)
        if match:
            query = match.group(0)
            # Add prefix if missing
            if not query.lower().startswith('prefix'):
                query = 'PREFIX kg: <http://example.org/kg/> \n' + query
            return query
            
        return None
    
    # Function to extract response text
    def get_response_text(response):
        if hasattr(response, 'text'):
            return response.text
        elif hasattr(response, 'content'):
            return response.content
        else:
            return str(response)
    
    # Function to execute query with error capture
    def execute_query_with_error_info(query):
        if not query:
            return False, [], "No valid query provided"
            
        try:
            results = list(kg.query(query))
            if results:
                return True, results, None
            else:
                return False, [], "Query executed successfully but returned no results"
        except Exception as e:
            return False, [], str(e)
    
    # Generate initial query with KG awareness
    initial_prompt = query_generator.generate_initial_exploration_prompt(
        question, kg_exploration
    )
    
    print("Generating initial query with KG awareness...")
    response = llm.complete(initial_prompt)
    query = extract_sparql_from_response(response)
    
    if not query:
        print("Failed to generate valid initial query")
        fallback_query = f"""
        PREFIX kg: <http://example.org/kg/>
        
        SELECT ?entity ?relation ?value WHERE {{
            ?entity ?relation ?value .
        }}
        LIMIT 20
        """
        query = fallback_query
    
    print(f"Initial query:\n{query}")
    
    # Track queries and results
    all_queries = [query]
    failed_attempts = []
    
    # Execute initial query
    success, results, error_info = execute_query_with_error_info(query)
    
    print(f"Initial query success: {success}")
    if error_info:
        print(f"Error info: {error_info}")
        failed_attempts.append(query)
    
    # Refinement loop
    iteration = 0
    current_query = query
    
    while not success and iteration < max_iterations:
        iteration += 1
        print(f"\n--- Query refinement iteration {iteration} ---")
        
        # Generate refinement prompt with error info and KG awareness
        refinement_prompt = query_generator.generate_refinement_with_kg_awareness_prompt(
            question, current_query, kg_exploration, error_info, failed_attempts
        )
        
        print("Generating refined query...")
        response = llm.complete(refinement_prompt)
        refined_query = extract_sparql_from_response(response)
        
        if not refined_query:
            print("Failed to generate valid refined query")
            # Try a simple query as fallback
            if iteration == max_iterations - 1:
                refined_query = """
                PREFIX kg: <http://example.org/kg/>
                
                SELECT ?s ?p ?o WHERE {
                    ?s ?p ?o .
                }
                LIMIT 10
                """
            else:
                # Skip this iteration
                continue
        
        current_query = refined_query
        all_queries.append(current_query)
        
        print(f"Refined query:\n{current_query}")
        
        # Execute the refined query
        success, results, error_info = execute_query_with_error_info(current_query)
        
        print(f"Query success: {success}")
        if error_info:
            print(f"Error info: {error_info}")
            failed_attempts.append(current_query)
            
        if success:
            print(f"Query successfully returned results on iteration {iteration}")
            break
    
    # Process results or fall back to context
    final_response_text = None  # Store the full final response
    
    if success:
        # Process query results
        try:
            # Get the variable names
            if hasattr(results, 'vars'):
                variables = results.vars
            else:
                variables = ["var" + str(i) for i in range(len(results[0]))]
                
            # Format results for analysis
            formatted_results = []
            for i, row in enumerate(results[:10]):  # Limit to first 10 results
                result_dict = {}
                for j, value in enumerate(row):
                    var_name = variables[j] if j < len(variables) else f"var{j}"
                    result_dict[var_name] = str(value)
                formatted_results.append(result_dict)
                
            # Formulate a prompt to interpret results
            result_str = "\n".join([str(r) for r in formatted_results])
            
            interpret_prompt = f"""
            Based on these query results, what is the answer to the question: "{question}"?
            
            Query results:
            {result_str}
            
            Please explain your reasoning step by step. After your explanation, provide your final answer
in a single phrase or word using this exact format: <ANSWER>your answer</ANSWER>
            """
            
            # Get interpretation
            interpret_response = llm.complete(interpret_prompt)
            final_response_text = get_response_text(interpret_response)  # Store full response
            answer = query_generator.extract_answer_from_llm_response(interpret_response)
            evidence = "Knowledge Graph"
            confidence = 0.9
            
        except Exception as e:
            print(f"Error processing results: {e}")
            # Fallback to simple extraction
            if results and len(results) > 0 and len(results[0]) > 0:
                answer = str(results[0][0])
                evidence = "Knowledge Graph (simple extraction)"
                confidence = 0.7
                final_response_text = f"Simple extraction from query results: {str(results)}"
            else:
                answer = "Error processing results"
                evidence = "Error"
                confidence = 0.0
                final_response_text = f"Error processing results: {str(e)}"
    else:
        # Use context data as fallback with KG entity information
        print("\nFalling back to context analysis with KG information...")
        
        # Generate context analysis prompt
        analysis_prompt = query_generator.generate_context_analysis_prompt(
            question, kg_exploration, context_data
        )
        
        # Get answer from context with KG awareness
        analysis_response = llm.complete(analysis_prompt)
        final_response_text = get_response_text(analysis_response)  # Store full response
        answer = query_generator.extract_answer_from_llm_response(analysis_response)
        
        evidence = "KG Exploration + Context"
        confidence = 0.8
        
        if not answer or answer.lower() in ["unknown", "not found", "cannot determine"]:
            answer = "Could not find an answer"
            evidence = "No answer found"
            confidence = 0.0
    
    # Prepare result
    result = {
        "question": question,
        "answer": answer,
        "queries_tried": all_queries,
        "kg_exploration": kg_exploration,
        "evidence_source": evidence,
        "confidence": confidence,
        "full_response_for_final": final_response_text  # Include the full final response
    }
    
    print(f"\nFinal answer: {answer}")
    print(f"Evidence source: {evidence}")
    print(f"Confidence: {confidence}")
    
    return result
