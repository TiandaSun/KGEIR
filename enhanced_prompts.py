class EnhancedQueryGenerator:
    """
    Enhanced prompt generation for more accurate SPARQL queries
    that integrate KG structure, context information, and step-by-step reasoning.
    """
    
    def generate_initial_query_prompt(self, question, kg_structure, context_data):
        """
        Generate an initial query prompt that combines KG and context understanding.
        
        Args:
            question: The question to answer
            kg_structure: Knowledge graph structure information
            context_data: Context data related to the question
            
        Returns:
            A prompt string for the LLM
        """
        # Format the KG structure information
        kg_info = self._format_kg_structure(kg_structure)
        
        # Format the context data
        context_info = self._format_context_data(context_data)
        
        prompt = f"""
        # Knowledge Graph Question Answering Task

        I need to answer this question using a combination of a knowledge graph (KG) and supporting text:
        
        ## Question
        "{question}"
        
        ## Knowledge Graph Structure
        {kg_info}
        
        ## Supporting Text Context
        {context_info}
        
        ## Task: Generate a SPARQL query to answer the question
        
        Think through this step by step:
        
        ### Step 1: Analyze what the question is asking
        Identify the key entities, relationships, and the target information we need to find.
        
        ### Step 2: Identify entities and relationships in the knowledge graph
        Look at the KG structure and find which entities and predicates could be relevant.
        
        ### Step 3: Check if information might be missing from the KG
        Check if some information might only exist in the text context, not in the KG.
        
        ### Step 4: Construct a SPARQL query
        Based on the above analysis, construct a SPARQL query that:
        - Uses the PREFIX kg: <http://example.org/kg/> namespace
        - Uses only entity types and predicates that actually exist in the KG
        - Is focused on the specific information needed to answer the question
        - Has correct SPARQL syntax (proper spacing, variable names, etc.)
        
        ### Step 5: Verify query validity
        Check that the query:
        - Has balanced braces and proper syntax
        - Uses variables consistently
        - Includes all necessary triple patterns
        - Does not include any syntactically incorrect constructs
        
        ### Final Output
        Return only the final, syntactically correct SPARQL query. Start with PREFIX and end with the final closing brace. Do not include any additional explanation.
        """
        
        return prompt
    
    def generate_refinement_prompt(self, question, previous_query, kg_structure, context_data, error_info=None):
        """
        Generate a query refinement prompt with error analysis and KG/context integration.
        
        Args:
            question: The question to answer
            previous_query: The previous query that failed
            kg_structure: Knowledge graph structure information
            context_data: Context data related to the question
            error_info: Optional error information from previous query
            
        Returns:
            A prompt string for the LLM
        """
        # Format the KG structure information
        kg_info = self._format_kg_structure(kg_structure)
        
        # Format the context data
        context_info = self._format_context_data(context_data)
        
        # Format error information if provided
        error_guidance = ""
        if error_info:
            error_guidance = f"""
            ## Error Information
            The previous query failed with this error: {error_info}
            
            Carefully analyze this error and make sure to fix it in your refined query.
            """
        
        prompt = f"""
        # SPARQL Query Refinement Task
        
        I need to refine a SPARQL query that didn't produce results for this question:
        
        ## Question
        "{question}"
        
        ## Previous Query That Failed
        ```sparql
        {previous_query}
        ```
        {error_guidance}
        
        ## Knowledge Graph Structure
        {kg_info}
        
        ## Supporting Text Context
        {context_info}
        
        ## Task: Refine the SPARQL query to get meaningful results
        
        Think through this step by step:
        
        ### Step 1: Diagnose what went wrong with the previous query
        Identify potential issues:
        - Syntax errors
        - Entities or predicates that don't exist in the KG
        - Overly restrictive patterns
        - Missing connections between entities
        
        ### Step 2: Check if the KG contains the needed information
        Examine the KG structure to determine if:
        - The entities mentioned in the question exist
        - The relationships needed to answer the question exist
        - There are alternative paths to connect the relevant entities
        
        ### Step 3: Look for insights in the supporting text
        Check if the text contains:
        - Entity names that might be represented differently in the KG
        - Relationships that should be in the KG
        - The answer itself (which might not be in the KG)
        
        ### Step 4: Create a refined query
        Based on your analysis:
        - Correct any syntax errors
        - Use only entities and predicates that exist in the KG
        - Make the query more general if necessary
        - Try alternative paths between entities
        - Ensure balanced braces and proper syntax
        
        ### Step 5: Verify query validity
        Check that the query:
        - Has balanced braces and proper syntax
        - Uses variables consistently
        - Includes all necessary triple patterns
        - Does not include any syntactically incorrect constructs
        
        ### Final Output
        Return only the final, syntactically correct SPARQL query. Start with PREFIX and end with the final closing brace. Do not include any additional explanation.
        """
        
        return prompt
    
    def generate_context_extraction_prompt(self, question, context_data, entities=None):
        """
        Generate a prompt for extracting answers directly from context when KG queries fail.
        
        Args:
            question: The question to answer
            context_data: Context data related to the question
            entities: Optional list of entities to focus on
            
        Returns:
            A prompt string for the LLM
        """
        # Format the context data
        context_info = self._format_context_data(context_data)
        
        # Format entities information if provided
        entities_info = ""
        if entities and len(entities) > 0:
            entities_str = ", ".join(entities)
            entities_info = f"""
            ## Relevant Entities
            These entities are relevant to the question: {entities_str}
            
            Focus on passages containing these entities.
            """
        
        prompt = f"""
        # Context-Based Answer Extraction
        
        The knowledge graph doesn't contain all the information needed to answer this question. 
        Please extract the answer directly from the provided context.
        
        ## Question
        "{question}"
        
        {entities_info}
        
        ## Context Information
        {context_info}
        
        ## Task: Extract the answer from the context
        
        Think through this step by step:
        
        ### Step 1: Analyze what the question is asking for
        Determine exactly what information we need to find.
        
        ### Step 2: Search for relevant passages
        Find passages in the context that contain:
        - Entities mentioned in the question
        - Information related to the specific thing being asked
        
        ### Step 3: Extract the precise answer
        From the relevant passages, extract the specific piece of information that answers the question.
        
        ### Step 4: Verify the answer
        Check that the answer:
        - Directly addresses the question
        - Is supported by the context
        - Is concise and specific
        
        ### Final Answer
        Provide only the answer as a concise phrase or sentence. Don't include any reasoning or explanation.
        """
        
        return prompt
    
    def _format_kg_structure(self, kg_structure):
        """Format knowledge graph structure information for the prompt."""
        if isinstance(kg_structure, dict):
            # Assuming a dictionary with entity_types, relationships, and examples
            entity_types = kg_structure.get('entity_types', [])
            relationships = kg_structure.get('relationships', [])
            entity_examples = kg_structure.get('entity_examples', {})
            
            # Format entity types
            entity_types_str = "\n".join([f"- {et}" for et in entity_types[:15]])
            
            # Format relationships
            relationships_str = "\n".join([f"- {rel}" for rel in relationships[:15]])
            
            # Format entity examples
            examples_str = ""
            for type_name, examples in list(entity_examples.items())[:5]:
                examples_list = ", ".join(examples[:3])
                examples_str += f"- {type_name}: {examples_list}\n"
            
            return f"""
            ### Entity Types
            {entity_types_str}
            
            ### Relationship Types
            {relationships_str}
            
            ### Entity Examples
            {examples_str}
            """
        else:
            # Assume it's a string or other format
            return str(kg_structure)
    
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
    
    def extract_answer_from_llm_response(self, response_text, max_length=100):
        """
        Extract a concise answer from an LLM response that might contain reasoning.
        
        Args:
            response_text: The text response from the LLM
            max_length: Maximum length for the extracted answer
            
        Returns:
            The extracted answer
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
        
        # If the response is already concise, return it
        if len(response_text) <= max_length and "\n" not in response_text:
            return response_text
        
        # Try patterns that might indicate a final answer
        import re
        
        # Pattern 1: Look for "Final Answer:" or similar
        final_answer_patterns = [
            r"(?:Final Answer|Answer|Result):\s*(.*?)(?:\n|$)",
            r"(?:The answer is|The result is):\s*(.*?)(?:\n|$)",
            r"(?:Therefore,|Thus,|In conclusion,)\s*(.*?)(?:\n|$)"
        ]
        
        for pattern in final_answer_patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Pattern 2: Take the last paragraph if it's short
        paragraphs = [p.strip() for p in response_text.split("\n\n") if p.strip()]
        if paragraphs and len(paragraphs[-1]) <= max_length:
            return paragraphs[-1]
        
        # Pattern 3: Take the last sentence if it's reasonable
        sentences = [s.strip() for s in re.split(r'[.!?]\s+', response_text) if s.strip()]
        if sentences and len(sentences[-1]) <= max_length:
            return sentences[-1]
        
        # Fallback: Just take the first part that fits
        return response_text[:max_length].strip()


def extract_kg_structure(kg):
    """
    Extract structured information about a knowledge graph.
    
    Args:
        kg: An RDFlib Graph object
        
    Returns:
        Dictionary with entity_types, relationships, and entity_examples
    """
    # Define namespaces
    try:
        kg_namespace = None
        for prefix, namespace in kg.namespaces():
            if prefix == "kg":
                kg_namespace = namespace
                break
                
        if not kg_namespace:
            kg_namespace = rdflib.Namespace("http://example.org/kg/")
    except:
        kg_namespace = rdflib.Namespace("http://example.org/kg/")
    
    # Extract entity types
    entity_types = []
    try:
        query = """
        SELECT DISTINCT ?type WHERE {
          ?entity <http://example.org/kg/hasType> ?type .
        }
        """
        for row in kg.query(query):
            type_uri = str(row[0])
            if type_uri.startswith(str(kg_namespace)):
                type_name = type_uri.split('/')[-1]
                entity_types.append(type_name)
    except Exception as e:
        print(f"Error extracting entity types: {e}")
    
    # Extract relationships
    relationships = []
    try:
        query = """
        SELECT DISTINCT ?predicate WHERE {
          ?s ?predicate ?o .
          FILTER(STRSTARTS(STR(?predicate), STR(<http://example.org/kg/>)))
          FILTER(?predicate != <http://example.org/kg/hasType> && ?predicate != <http://example.org/kg/name>)
        }
        """
        for row in kg.query(query):
            pred_uri = str(row[0])
            if pred_uri.startswith(str(kg_namespace)):
                pred_name = pred_uri.split('/')[-1]
                relationships.append(pred_name)
    except Exception as e:
        print(f"Error extracting relationships: {e}")
    
    # Get entity examples for each type
    entity_examples = {}
    for entity_type in entity_types:
        examples = []
        try:
            query = f"""
            SELECT ?name WHERE {{
              ?entity <http://example.org/kg/hasType> <http://example.org/kg/{entity_type}> .
              ?entity <http://example.org/kg/name> ?name .
            }}
            LIMIT 5
            """
            for row in kg.query(query):
                examples.append(str(row[0]))
            
            if examples:
                entity_examples[entity_type] = examples
        except Exception as e:
            print(f"Error extracting examples for {entity_type}: {e}")
    
    # Return the structured information
    return {
        'entity_types': entity_types,
        'relationships': relationships,
        'entity_examples': entity_examples
    }


def process_question_with_enhanced_prompts(kg, context_data, question, llm, max_iterations=3):
    """
    Process a question using enhanced prompts for SPARQL generation.
    
    Args:
        kg: RDFlib Graph object
        context_data: Context information
        question: Question to answer
        llm: Language model
        max_iterations: Maximum number of refinement iterations
        
    Returns:
        Dictionary with the answer and process details
    """
    import re
    from rdflib import Graph, Namespace
    
    print(f"Processing question: {question}")
    
    # Extract knowledge graph structure
    kg_structure = extract_kg_structure(kg)
    
    # Create prompt generator
    prompt_generator = EnhancedQueryGenerator()
    
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
    
    # Generate initial query
    initial_prompt = prompt_generator.generate_initial_query_prompt(
        question, kg_structure, context_data
    )
    
    print("Generating initial query...")
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
    
    # Execute initial query
    success, results, error_info = execute_query_with_error_info(query)
    
    print(f"Initial query success: {success}")
    if error_info:
        print(f"Error info: {error_info}")
    
    # Refinement loop
    iteration = 0
    current_query = query
    
    while not success and iteration < max_iterations:
        iteration += 1
        print(f"\n--- Query refinement iteration {iteration} ---")
        
        # Generate refinement prompt with error info
        refinement_prompt = prompt_generator.generate_refinement_prompt(
            question, current_query, kg_structure, context_data, error_info
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
            
        if success:
            print(f"Query successfully returned results on iteration {iteration}")
            break
    
    # Process results or fall back to context
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
            
            Provide only the concise answer.
            """
            
            # Get interpretation
            interpret_response = llm.complete(interpret_prompt)
            answer = prompt_generator.extract_answer_from_llm_response(interpret_response)
            evidence = "Knowledge Graph"
            confidence = 0.9
            
        except Exception as e:
            print(f"Error processing results: {e}")
            # Fallback to simple extraction
            if results and len(results) > 0 and len(results[0]) > 0:
                answer = str(results[0][0])
                evidence = "Knowledge Graph (simple extraction)"
                confidence = 0.7
            else:
                answer = "Error processing results"
                evidence = "Error"
                confidence = 0.0
    else:
        # Use context data as fallback
        print("\nFalling back to context extraction...")
        
        # Extract entities from question for context search
        entity_pattern = r'\b[A-Z][a-zA-Z]*\b'
        entities = re.findall(entity_pattern, question)
        
        # Generate context extraction prompt
        extraction_prompt = prompt_generator.generate_context_extraction_prompt(
            question, context_data, entities
        )
        
        # Get answer from context
        extraction_response = llm.complete(extraction_prompt)
        answer = prompt_generator.extract_answer_from_llm_response(extraction_response)
        
        evidence = "Context (fallback)"
        confidence = 0.7
        
        if not answer or answer.lower() in ["unknown", "not found", "cannot determine"]:
            answer = "Could not find an answer"
            evidence = "No answer found"
            confidence = 0.0
    
    # Prepare result
    result = {
        "question": question,
        "answer": answer,
        "queries_tried": all_queries,
        "evidence_source": evidence,
        "confidence": confidence
    }
    
    print(f"\nFinal answer: {answer}")
    print(f"Evidence source: {evidence}")
    print(f"Confidence: {confidence}")
    
    return result