import json
import re
from typing import Dict, List, Tuple, Optional, Any, Union
from rdflib import Graph, Namespace, URIRef, Literal
import rdflib
import pandas as pd
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def extract_sparql_from_response(response: str) -> Optional[str]:
    """
    Extract a SPARQL query from an LLM response.
    
    Args:
        response: The LLM response text
        
    Returns:
        Extracted SPARQL query or None if not found
    """
    # Try to find a SPARQL query in the LLM response
    sparql_pattern = r'(?:PREFIX.*?)?SELECT.*?WHERE\s*\{.*?\}'
    match = re.search(sparql_pattern, response, re.DOTALL | re.IGNORECASE)
    
    if match:
        return match.group(0)
    
    return None


class KnowledgeGraphReasoner:
    """
    A reasoning system for multi-hop question answering using knowledge graphs 
    and context information.
    """
    
    def __init__(self, llm=None, max_iterations: int = 5):
        """
        Initialize the reasoner with optional language model and configuration.
        
        Args:
            llm: Language model for query generation and refinement (optional)
            max_iterations: Maximum number of query refinement iterations
        """
        self.kg = None
        self.llm = llm
        self.max_iterations = max_iterations
        self.kg_namespace = None
        self.context_data = None
        self.question = None
        self.queries_tried = []
        self.query_results = []
    

    def load_kg_from_file(self, file_path: str) -> None:
        """
        Load a knowledge graph from a Turtle (.ttl) file.
        
        Args:
            file_path: Path to the RDF/Turtle file
        """
        self.kg = Graph()
        self.kg.parse(file_path, format="turtle")
        
        # Extract the namespace from the knowledge graph
        for prefix, namespace in self.kg.namespaces():
            if prefix == "kg":
                self.kg_namespace = namespace
                break
                
        # If no kg namespace found, create a default one
        if not self.kg_namespace:
            self.kg_namespace = Namespace("http://example.org/kg/")
            
        logger.info(f"Loaded knowledge graph with {len(self.kg)} triples")
        
    def load_kg_from_string(self, kg_content: str) -> None:
        """
        Load a knowledge graph from a string containing Turtle RDF.
        
        Args:
            kg_content: String containing the RDF/Turtle content
        """
        self.kg = Graph()
        self.kg.parse(data=kg_content, format="turtle")
        
        # Extract the namespace from the knowledge graph
        for prefix, namespace in self.kg.namespaces():
            if prefix == "kg":
                self.kg_namespace = namespace
                break
                
        # If no kg namespace found, create a default one
        if not self.kg_namespace:
            self.kg_namespace = Namespace("http://example.org/kg/")
            
        logger.info(f"Loaded knowledge graph with {len(self.kg)} triples")
        
    def set_kg(self, kg: Graph) -> None:
        """
        Set the knowledge graph directly using an existing RDFlib Graph object.
        
        Args:
            kg: RDFlib Graph object containing the knowledge graph
        """
        self.kg = kg
        
        # Extract the namespace from the knowledge graph
        for prefix, namespace in self.kg.namespaces():
            if prefix == "kg":
                self.kg_namespace = namespace
                break
                
        # If no kg namespace found, create a default one
        if not self.kg_namespace:
            self.kg_namespace = Namespace("http://example.org/kg/")
            
        logger.info(f"Set knowledge graph with {len(self.kg)} triples")
    def load_context(self, context_data: Any) -> None:
        """
        Load context information for answering questions.
        
        Args:
            context_data: Context information (can be string, dict, or list)
        """
        self.context_data = context_data
        logger.info("Loaded context information")
        
    def _generate_initial_query(self, question: str) -> str:
        """
        Generate an initial SPARQL query based on the question.
        
        Args:
            question: The question to answer
            
        Returns:
            A SPARQL query string
        """
        # If LLM is available, use it to generate the query
        if self.llm:
            prompt = self._create_initial_query_prompt(question)
            response = self.llm.complete(prompt)
            query = extract_sparql_from_response(response)
            if query:
                return query
        
        # Fallback for when LLM is not available or generates invalid query
        return self._generate_baseline_query(question)
    
    def _create_initial_query_prompt(self, question: str) -> str:
        """
        Create a prompt for the LLM to generate an initial SPARQL query.
        
        Args:
            question: The question to answer
            
        Returns:
            A prompt string for the LLM
        """
        # Extract entities and predicates from the knowledge graph for context
        entity_types = self._extract_entity_types()
        relation_types = self._extract_relation_types()
        
        # Format KG information for the prompt
        entity_types_str = "\n".join([f"- {entity_type}" for entity_type in entity_types[:20]])
        relation_types_str = "\n".join([f"- {relation}" for relation in relation_types[:20]])
        
        # Create the prompt
        prompt = f"""
        Generate a SPARQL query to answer the following question using a knowledge graph:
        
        Question: "{question}"
        
        The knowledge graph has these entity types:
        {entity_types_str}
        
        And these relationship types:
        {relation_types_str}
        
        Generate a valid SPARQL query that uses the PREFIX kg: <http://example.org/kg/> namespace.
        Return only the SPARQL query with no additional explanation.
        """
        
        return prompt
    
    def _extract_entity_types(self) -> List[str]:
        """
        Extract entity types from the knowledge graph.
        
        Returns:
            List of entity type names
        """
        if not self.kg:
            return []
            
        query = """
        PREFIX kg: <http://example.org/kg/>
        SELECT DISTINCT ?type WHERE {
            ?entity kg:hasType ?type .
        }
        """
        
        results = self.kg.query(query)
        entity_types = []
        
        for row in results:
            entity_type = str(row[0])
            if entity_type.startswith(str(self.kg_namespace)):
                entity_type = entity_type.replace(str(self.kg_namespace), "kg:")
            entity_types.append(entity_type)
            
        return entity_types
    
    def _extract_relation_types(self) -> List[str]:
        """
        Extract relationship types from the knowledge graph.
        
        Returns:
            List of relationship type names
        """
        if not self.kg:
            return []
            
        query = """
        SELECT DISTINCT ?predicate WHERE {
            ?s ?predicate ?o .
            FILTER(STRSTARTS(STR(?predicate), STR(kg:)))
            FILTER(?predicate != kg:hasType && ?predicate != kg:name)
        }
        """
        
        results = self.kg.query(query)
        relation_types = []
        
        for row in results:
            relation = str(row[0])
            if relation.startswith(str(self.kg_namespace)):
                relation = relation.replace(str(self.kg_namespace), "kg:")
            relation_types.append(relation)
            
        return relation_types
    
    def _generate_baseline_query(self, question: str) -> str:
        """
        Generate a baseline SPARQL query when LLM is not available.
        
        Args:
            question: The question to answer
            
        Returns:
            A SPARQL query string
        """
        # Extract potential entities from the question
        entities = self._extract_entities_from_question(question)
        
        # Create a simple query that looks for connections between entities
        entity_clauses = []
        for i, entity in enumerate(entities[:2]):  # Use at most 2 entities
            entity_var = f"?entity{i}"
            entity_clauses.append(f"{entity_var} kg:name ?name{i} . FILTER(CONTAINS(LCASE(?name{i}), LCASE(\"{entity}\"))) ")
        
        entity_pattern = " ".join(entity_clauses)
        
        query = f"""
        PREFIX kg: <http://example.org/kg/>
        
        SELECT ?entity ?relation ?target WHERE {{
            {entity_pattern}
            ?entity ?relation ?target .
        }}
        LIMIT 10
        """
        
        return query
    
    def _extract_entities_from_question(self, question: str) -> List[str]:
        """
        Extract potential entity names from the question.
        
        Args:
            question: The question to answer
            
        Returns:
            List of potential entity names
        """
        # Simple extraction based on capitalized words
        capitalized_pattern = r'\b[A-Z][a-z]*\b'
        entities = re.findall(capitalized_pattern, question)
        
        # Extract multi-word entities by looking for title case phrases
        title_case_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b'
        title_case_entities = re.findall(title_case_pattern, question)
        
        return entities + title_case_entities
    

    
    def execute_query(self, query: str) -> Tuple[bool, List]:
        """
        Execute a SPARQL query on the knowledge graph.
        
        Args:
            query: The SPARQL query to execute
            
        Returns:
            Tuple of (success, results)
        """
        if not self.kg:
            logger.error("No knowledge graph loaded")
            return False, []
            
        try:
            # Execute the query
            results = list(self.kg.query(query))
            
            # Check if we got any results
            if results:
                logger.info(f"Query successful with {len(results)} results")
                return True, results
                
            logger.info("Query executed successfully but returned no results")
            return False, []
            
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            return False, []
    
    def _refine_query(self, question: str, previous_query: str, iteration: int) -> str:
        """
        Refine a SPARQL query that didn't produce results.
        
        Args:
            question: The question to answer
            previous_query: The previous query that failed
            iteration: The current iteration number
            
        Returns:
            A refined SPARQL query
        """
        if self.llm:
            # Create a refinement prompt
            prompt = self._create_refinement_prompt(question, previous_query, iteration)
            response = self.llm.complete(prompt)
            refined_query = extract_sparql_from_response(response)
            
            if refined_query:
                return refined_query
        
        # Fallback to basic refinement strategies if LLM fails
        return self._refine_query_heuristically(previous_query, iteration)
    
    def _create_refinement_prompt(self, question: str, previous_query: str, iteration: int) -> str:
        """
        Create a prompt for the LLM to refine a SPARQL query.
        
        Args:
            question: The question to answer
            previous_query: The previous query that failed
            iteration: The current iteration number
            
        Returns:
            A prompt string for the LLM
        """
        # Extract relevant context for the refinement
        context_info = self._extract_relevant_context(question)
        kg_info = self._extract_kg_structure_info()
        
        # Strategy changes based on iteration
        strategies = [
            "Make the query more general by removing constraints",
            "Try different relationship paths between entities",
            "Include alternative entity names or literals",
            "Look for indirect connections (entities connected through a common third entity)",
            "Integrate information from the context that might not be in the KG"
        ]
        
        current_strategy = strategies[min(iteration, len(strategies) - 1)]
        
        prompt = f"""
        Refine the following SPARQL query that didn't return results:
        
        Question: "{question}"
        
        Previous query:
        ```
        {previous_query}
        ```
        
        Relevant context information:
        {context_info}
        
        Knowledge graph information:
        {kg_info}
        
        Refinement strategy to try: {current_strategy}
        
        Generate a valid SPARQL query that uses the PREFIX kg: <http://example.org/kg/> namespace.
        Return only the SPARQL query with no additional explanation.
        """
        
        return prompt
    
    def _extract_relevant_context(self, question: str) -> str:
        """
        Extract context information relevant to the question.
        
        Args:
            question: The question to answer
            
        Returns:
            String containing relevant context
        """
        if not self.context_data:
            return "No context information available."
            
        # Convert context data to a standardized format
        if isinstance(self.context_data, dict):
            # Assume dictionary with title and paragraphs
            context_str = ""
            for key, value in self.context_data.items():
                if key == 'title':
                    context_str += f"Title: {value}\n"
                elif key == 'paragraphs' or key == 'content':
                    if isinstance(value, list):
                        context_str += f"Content: {' '.join(value)}\n"
                    else:
                        context_str += f"Content: {value}\n"
                else:
                    context_str += f"{key}: {value}\n"
            return context_str
                
        elif isinstance(self.context_data, list):
            # Assume list of context items
            context_items = []
            for item in self.context_data:
                if isinstance(item, dict) and 'title' in item and 'paragraphs' in item:
                    title = item['title']
                    paragraphs = ' '.join(item['paragraphs'])
                    context_items.append(f"Title: {title}\nContent: {paragraphs}")
                else:
                    context_items.append(str(item))
            return "\n\n".join(context_items)
                
        else:
            # Assume string
            return str(self.context_data)
    
    def _extract_kg_structure_info(self) -> str:
        """
        Extract structural information from the knowledge graph.
        
        Returns:
            String containing KG structure information
        """
        if not self.kg:
            return "No knowledge graph available."
            
        # Get entity and relationship counts
        entity_types = self._extract_entity_types()
        relation_types = self._extract_relation_types()
        
        # Sample entities by type
        entity_samples = {}
        for entity_type in entity_types[:5]:  # Limit to 5 types
            query = f"""
            PREFIX kg: <http://example.org/kg/>
            SELECT ?entity ?name
            WHERE {{
                ?entity kg:hasType kg:{entity_type.replace('kg:', '')} .
                ?entity kg:name ?name .
            }}
            LIMIT 3
            """
            
            try:
                results = list(self.kg.query(query))
                if results:
                    entity_samples[entity_type] = [str(row[1]) for row in results]
            except:
                pass
        
        # Format the information
        structure_info = f"Entity types ({len(entity_types)}): {', '.join(entity_types[:10])}\n"
        structure_info += f"Relationship types ({len(relation_types)}): {', '.join(relation_types[:10])}\n\n"
        
        structure_info += "Sample entities by type:\n"
        for entity_type, samples in entity_samples.items():
            structure_info += f"- {entity_type}: {', '.join(samples)}\n"
        
        return structure_info
    
    def _refine_query_heuristically(self, previous_query: str, iteration: int) -> str:
        """
        Apply heuristic refinements to a SPARQL query.
        
        Args:
            previous_query: The previous query that failed
            iteration: The current iteration number
            
        Returns:
            A refined SPARQL query
        """
        # Apply different heuristics based on the iteration
        if iteration == 1:
            # Make the query more general by removing some filters
            return re.sub(r'FILTER\s*\([^)]*\)', '', previous_query)
            
        elif iteration == 2:
            # Try different variable patterns
            return previous_query.replace("?entity ?relation ?target", "?target ?relation ?entity")
            
        elif iteration == 3:
            # Add a LIMIT to get at least some results
            if "LIMIT" not in previous_query:
                return previous_query + " LIMIT 20"
            else:
                return previous_query
                
        elif iteration == 4:
            # Make a very general query to see what's in the KG
            return """
            PREFIX kg: <http://example.org/kg/>
            
            SELECT ?s ?p ?o WHERE {
              ?s ?p ?o .
            }
            LIMIT 20
            """
            
        else:
            # Final fallback - just return a simple query about entity types
            return """
            PREFIX kg: <http://example.org/kg/>
            
            SELECT ?entity ?type ?name WHERE {
              ?entity kg:hasType ?type .
              ?entity kg:name ?name .
            }
            LIMIT 20
            """
    
    def find_answer_in_context(self, question: str, entities: List[str]) -> Optional[str]:
        """
        Search for an answer in the context when KG queries fail.
        
        Args:
            question: The question to answer
            entities: List of entities found in the question or KG
            
        Returns:
            Potential answer string or None
        """
        if not self.context_data:
            return None
            
        context_str = self._extract_relevant_context(question)
        
        # If LLM is available, use it to extract the answer from context
        if self.llm:
            prompt = f"""
            Find the answer to the following question using only the provided context:
            
            Question: "{question}"
            
            Context:
            {context_str}
            
            Entities mentioned: {', '.join(entities)}
            
            Return only the answer with no additional explanation.
            """
            
            response = self.llm.complete(prompt)
            return response.strip()
            
        # Fallback to simple text matching
        for entity in entities:
            # Look for sentences containing the entity and potential answer patterns
            sentences = self._extract_sentences_with_entity(context_str, entity)
            for sentence in sentences:
                # Check for common answer patterns like "is located in", "has its head office in", etc.
                for pattern in ["is located in", "located in", "based in", "head office in", "headquarters in"]:
                    if pattern in sentence.lower():
                        parts = sentence.lower().split(pattern)
                        if len(parts) > 1:
                            # Extract the part after the pattern
                            potential_answer = parts[1].strip().strip('.,')
                            return potential_answer
                            
        return None
    
    def _extract_sentences_with_entity(self, text: str, entity: str) -> List[str]:
        """
        Extract sentences from text that contain a specific entity.
        
        Args:
            text: The text to search
            entity: The entity to look for
            
        Returns:
            List of sentences containing the entity
        """
        # Split text into sentences
        sentences = re.split(r'[.!?]', text)
        
        # Filter sentences containing the entity
        return [sentence.strip() for sentence in sentences if entity.lower() in sentence.lower()]
    
    def answer_question(self, question: str) -> Dict:
        """
        Answer a question using the knowledge graph and context.
        
        Args:
            question: The question to answer
            
        Returns:
            Dict containing the answer, query, and reasoning process
        """
        self.question = question
        self.queries_tried = []
        self.query_results = []
        
        logger.info(f"Answering question: {question}")
        
        # Step 1: Generate initial SPARQL query
        initial_query = self._generate_initial_query(question)
        current_query = initial_query
        self.queries_tried.append(current_query)
        
        logger.info(f"Initial query: {current_query}")
        
        # Step 2: Execute the initial query
        success, results = self.execute_query(current_query)
        self.query_results.append(results)
        
        # Step 3-7: Iterative refinement if needed
        iteration = 0
        while not success and iteration < self.max_iterations:
            iteration += 1
            logger.info(f"Query refinement iteration {iteration}")
            
            # Refine the query
            current_query = self._refine_query(question, current_query, iteration)
            self.queries_tried.append(current_query)
            
            logger.info(f"Refined query: {current_query}")
            
            # Execute the refined query
            success, results = self.execute_query(current_query)
            self.query_results.append(results)
            
            if success:
                logger.info(f"Query successful on iteration {iteration}")
                break
        
        # Step 8: Process results or use context as fallback
        if success:
            answer = self._process_query_results(results)
            evidence = "Knowledge graph"
            confidence = 0.9
        else:
            # Extract entities from the question for context search
            entities = self._extract_entities_from_question(question)
            
            # Try to supplement with information from the KG
            kg_entities = self._extract_related_entities()
            entities = list(set(entities + kg_entities))
            
            # Try to find the answer in the context
            answer = self.find_answer_in_context(question, entities)
            evidence = "Context (fallback)"
            confidence = 0.7
            
            if not answer:
                answer = "Could not find an answer"
                evidence = "No answer found"
                confidence = 0.0
        
        # Prepare the result
        result = {
            "question": question,
            "answer": answer,
            "queries_tried": self.queries_tried,
            "last_query_results": self.query_results[-1] if self.query_results else [],
            "evidence_source": evidence,
            "confidence": confidence
        }
        
        logger.info(f"Answer: {answer}")
        return result
    
    def _process_query_results(self, results: List) -> str:
        """
        Process query results to extract an answer.
        
        Args:
            results: The query results
            
        Returns:
            Answer string
        """
        if not results:
            return "No results found"
            
        # Extract the first value from each result row
        values = []
        for row in results:
            if len(row) > 0:
                # Convert RDFLib terms to strings
                value = str(row[0])
                
                # Remove namespace prefix if present
                if value.startswith(str(self.kg_namespace)):
                    value = value.replace(str(self.kg_namespace), "")
                
                values.append(value)
        
        # If we have entity URIs, try to get their names
        named_values = []
        for value in values:
            name = self._get_entity_name(value)
            if name:
                named_values.append(name)
        
        if named_values:
            return ", ".join(named_values)
        else:
            return ", ".join(values)
    
    def _get_entity_name(self, entity_uri: str) -> Optional[str]:
        """
        Get the name of an entity from its URI.
        
        Args:
            entity_uri: The entity URI string
            
        Returns:
            Entity name or None if not found
        """
        if not self.kg:
            return None
            
        # Create the full URI if it's not already
        if not entity_uri.startswith("http"):
            entity_uri = str(self.kg_namespace) + entity_uri
            
        # Query for the entity's name
        query = f"""
        PREFIX kg: <http://example.org/kg/>
        
        SELECT ?name WHERE {{
          <{entity_uri}> kg:name ?name .
        }}
        LIMIT 1
        """
        
        try:
            results = list(self.kg.query(query))
            if results and len(results[0]) > 0:
                return str(results[0][0])
        except:
            pass
            
        return None
    
    def _extract_related_entities(self) -> List[str]:
        """
        Extract entities related to the question from the knowledge graph.
        
        Returns:
            List of entity names
        """
        if not self.kg or not self.question:
            return []
            
        # Extract word stems from the question
        words = re.findall(r'\b\w+\b', self.question.lower())
        words = [word for word in words if len(word) > 3]  # Filter out short words
        
        # Find entities with names containing these words
        entity_names = []
        for word in words:
            query = f"""
            PREFIX kg: <http://example.org/kg/>
            
            SELECT ?name WHERE {{
              ?entity kg:name ?name .
              FILTER(CONTAINS(LCASE(?name), "{word}"))
            }}
            LIMIT 5
            """
            
            try:
                results = list(self.kg.query(query))
                entity_names.extend([str(row[0]) for row in results])
            except:
                pass
                
        return entity_names


def process_question_with_kg(question: str, kg_data: str, context_data: Any, llm=None) -> Dict:
    """
    Process a question using knowledge graph and context.
    
    Args:
        question: The question to answer
        kg_data: Knowledge graph data (string)
        context_data: Context information
        llm: Optional language model
        
    Returns:
        Dict containing the answer and reasoning process
    """
    # Initialize the reasoner
    reasoner = KnowledgeGraphReasoner(llm=llm)
    
    # Load the knowledge graph
    reasoner.load_kg_from_string(kg_data)
    
    # Load the context
    reasoner.load_context(context_data)
    
    # Answer the question
    return reasoner.answer_question(question)


def process_batch(questions: List[str], kg_data: str, context_data: Any, llm=None) -> List[Dict]:
    """
    Process a batch of questions using the same knowledge graph and context.
    
    Args:
        questions: List of questions to answer
        kg_data: Knowledge graph data (string)
        context_data: Context information
        llm: Optional language model
        
    Returns:
        List of results for each question
    """
    results = []
    
    # Initialize the reasoner once
    reasoner = KnowledgeGraphReasoner(llm=llm)
    
    # Load the knowledge graph
    reasoner.load_kg_from_string(kg_data)
    
    # Load the context
    reasoner.load_context(context_data)
    
    # Process each question
    for question in questions:
        result = reasoner.answer_question(question)
        results.append(result)
        
    return results


def process_dataset(dataset_path: str, output_path: str, llm=None) -> None:
    """
    Process a dataset of questions with their corresponding KGs and contexts.
    
    Args:
        dataset_path: Path to the dataset
        output_path: Path to save the results
        llm: Optional language model
    """
    # Load the dataset
    dataset = pd.read_json(dataset_path)
    
    results = []
    
    # Process each row in the dataset
    for i, row in dataset.iterrows():
        question = row['question']
        kg_data = row['kg']
        context_data = row['context']
        
        # Process the question
        result = process_question_with_kg(question, kg_data, context_data, llm)
        results.append(result)
        
    # Save the results
    results_df = pd.DataFrame(results)
    results_df.to_json(output_path, orient='records')
    
    logger.info(f"Processed {len(results)} questions. Results saved to {output_path}")

