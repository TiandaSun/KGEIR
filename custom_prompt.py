class QueryRefinementPromptGenerator:
    """
    Class for generating custom prompts for SPARQL query refinement
    that integrates KG structure and textual context information.
    """
    
    def generate_initial_query_prompt(self, question, entity_types, relation_types):
        """
        Generate a prompt for creating the initial SPARQL query.
        
        Args:
            question: The question to answer
            entity_types: List of entity types in the KG
            relation_types: List of relationship types in the KG
            
        Returns:
            Prompt string
        """
        # Format KG information for the prompt
        entity_types_str = "\n".join([f"- {entity_type}" for entity_type in entity_types[:20]])
        relation_types_str = "\n".join([f"- {relation}" for relation in relation_types[:20]])
        
        prompt = f"""
        Generate a SPARQL query to answer the following question using a knowledge graph:
        
        Question: "{question}"
        
        The knowledge graph has these entity types:
        {entity_types_str}
        
        And these relationship types:
        {relation_types_str}
        
        Follow these steps to create a well-formed SPARQL query:
        
        1. Identify the key entities in the question
        2. Determine what relationship types might connect these entities
        3. Identify what information we're looking for (the target variable)
        4. Create a SPARQL query with appropriate triple patterns
        
        Make sure to:
        - Use the PREFIX kg: <http://example.org/kg/> namespace
        - Include relevant entity names and relationship types from the provided lists
        - Create a query that's specific enough to answer the question but not too restrictive
        
        Return only the SPARQL query with no additional explanation.
        """
        
        return prompt
    
    def generate_refinement_prompt(self, question, previous_query, context_info, kg_info, iteration):
        """
        Generate a prompt for refining a SPARQL query that didn't produce results.
        
        Args:
            question: The question to answer
            previous_query: The previous query that failed
            context_info: Relevant context information
            kg_info: Knowledge graph structure information
            iteration: Current refinement iteration
            
        Returns:
            Prompt string
        """
        # Define refinement strategies based on iteration
        strategies = [
            "Make the query more general by removing constraints.",
            "Try different relationship paths between entities.",
            "Check if entities or relationships might be represented differently in the KG.",
            "Look for indirect connections through intermediate entities.",
            "Consider if the answer might only be in the textual context, not the KG."
        ]
        
        current_strategy = strategies[min(iteration, len(strategies) - 1)]
        
        prompt = f"""
        Refine this SPARQL query that didn't return results to answer the question.
        
        Question: "{question}"
        
        Previous query:
        ```sparql
        {previous_query}
        ```
        
        Relevant context information:
        {context_info}
        
        Knowledge graph information:
        {kg_info}
        
        REFINEMENT STRATEGY: {current_strategy}
        
        Please carefully analyze:
        
        1. Whether the entity names in the query match exactly with the KG
        2. Whether the relationship types are used correctly
        3. If the query structure makes logical sense for the question
        4. What information from the context might help refine the query
        
        IMPORTANT: Remember that some information might only be in the text context and not in the KG.
        In that case, focus on finding what IS in the KG that relates to the question.
        
        Return only the refined SPARQL query with no additional explanation.
        """
        
        return prompt
    
    def generate_answer_extraction_prompt(self, question, query_results, context_info):
        """
        Generate a prompt for extracting the final answer from query results and context.
        
        Args:
            question: The question to answer
            query_results: Results from the successful query
            context_info: Relevant context information
            
        Returns:
            Prompt string
        """
        # Format query results for the prompt
        results_str = self._format_results_for_prompt(query_results)
        
        prompt = f"""
        Extract the answer to the question based on the knowledge graph query results and context.
        
        Question: "{question}"
        
        Query results:
        {results_str}
        
        Context information:
        {context_info}
        
        Follow these steps:
        
        1. Analyze the query results to see what information they provide
        2. Check if the results directly answer the question
        3. If not, look for additional information in the context
        4. Combine information from both sources to form the answer
        
        Return your answer in this JSON format:
        ```json
        {{
            "answer": "your concise answer",
            "reasoning": "brief explanation of how you determined the answer",
            "source": "knowledge graph" or "context" or "both",
            "confidence": a number between 0 and 1
        }}
        ```
        """
        
        return prompt
    
    def _format_results_for_prompt(self, query_results):
        """Format query results in a human-readable way for the prompt."""
        if not query_results:
            return "No results found"
            
        formatted_results = []
        for i, row in enumerate(query_results[:10]):  # Limit to first 10 results
            result_str = f"Result {i+1}: "
            for j, value in enumerate(row):
                var_name = query_results.vars[j] if hasattr(query_results, 'vars') else f"var{j}"
                result_str += f"{var_name}={value}, "
            formatted_results.append(result_str.rstrip(", "))
            
        return "\n".join(formatted_results)
    
    def generate_fallback_prompt(self, question, context_info):
        """
        Generate a prompt for extracting the answer from context when KG queries fail.
        
        Args:
            question: The question to answer
            context_info: Relevant context information
            
        Returns:
            Prompt string
        """
        prompt = f"""
        The knowledge graph queries didn't provide an answer. Extract the answer from the context information.
        
        Question: "{question}"
        
        Context information:
        {context_info}
        
        Follow these steps:
        
        1. Identify key entities mentioned in the question
        2. Look for these entities in the context
        3. Find sentences that relate these entities to the information being asked for
        4. Extract the specific answer from these sentences
        
        Return your answer in this JSON format:
        ```json
        {{
            "answer": "your concise answer",
            "evidence": "the specific sentence or text that contains the answer",
            "confidence": a number between 0 and 1
        }}
        ```
        
        If you cannot find the answer in the context, set "answer" to "Unknown" and "confidence" to 0.
        """
        
        return prompt


# Example usage:
if __name__ == "__main__":
    from kg_reasoner import KnowledgeGraphReasoner, process_question_with_kg
    
    # Create a custom prompt generator
    prompt_generator = QueryRefinementPromptGenerator()
    
    # Create a custom LLM wrapper with the prompt generator
    class CustomLLMWrapper:
        def __init__(self, llm, prompt_generator):
            self.llm = llm
            self.prompt_generator = prompt_generator
            self.current_stage = "initial"
            self.iteration = 0
            self.context_info = None
            self.kg_info = None
        
        def set_context(self, context_info, kg_info):
            self.context_info = context_info
            self.kg_info = kg_info
        
        def complete(self, prompt):
            # Modify the prompt based on the current stage
            if "Generate a SPARQL query" in prompt and self.current_stage == "initial":
                # Extract entity types and relation types from the original prompt
                lines = prompt.split("\n")
                question = [line for line in lines if "Question:" in line][0].split("Question:")[1].strip().strip('"')
                
                # Find the entity types and relation types sections
                entity_start = prompt.find("entity types:")
                relation_start = prompt.find("relationship types:")
                
                if entity_start != -1 and relation_start != -1:
                    entity_section = prompt[entity_start:relation_start]
                    relation_section = prompt[relation_start:]
                    
                    entity_types = [line.strip("- ") for line in entity_section.split("\n") if line.strip().startswith("-")]
                    relation_types = [line.strip("- ") for line in relation_section.split("\n") if line.strip().startswith("-")]
                    
                    # Use the custom prompt generator
                    prompt = self.prompt_generator.generate_initial_query_prompt(question, entity_types, relation_types)
                
                self.current_stage = "refinement"
                
            elif "Refine the following SPARQL query" in prompt and self.current_stage == "refinement":
                lines = prompt.split("\n")
                question = [line for line in lines if "Question:" in line][0].split("Question:")[1].strip().strip('"')
                
                # Extract the previous query
                query_start = prompt.find("```")
                query_end = prompt.find("```", query_start + 3) + 3
                previous_query = prompt[query_start:query_end].strip("`").strip()
                
                # Use the custom prompt generator
                prompt = self.prompt_generator.generate_refinement_prompt(
                    question, 
                    previous_query, 
                    self.context_info, 
                    self.kg_info, 
                    self.iteration
                )
                
                self.iteration += 1
                
            # Pass the modified prompt to the actual LLM
            return self.llm.complete(prompt)
    
    # Example question and data
    question = "The Oberoi family is part of a hotel company that has a head office in what city?"
    
    # Load knowledge graph and context data
    # ... (same as in the main code)
    
    # Initialize the custom LLM wrapper with your actual LLM
    # custom_llm = CustomLLMWrapper(your_llm, prompt_generator)
    
    # Process the question
    # result = process_question_with_kg(question, kg_data, context_data, custom_llm)