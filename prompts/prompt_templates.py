from llama_index.core import PromptTemplate


template_qa_context2triple = (

# Context Triple Extraction Task

"""


You are tasked with extracting knowledge triples from context passages to build a comprehensive knowledge graph. Please thinking step by step, Your goal is to identify entities, their relationships, attributes, and temporal information with high granularity.

## Task Requirements:
1. Extract both explicit and implicit relationships between entities
2. Break down complex statements into atomic facts (e.g., "French professor" → "has_nationality" AND "has_occupation")
3. Capture temporal ordering of events when present
4. Preserve original entity names exactly as they appear
5. Include entity types, attributes, and properties when mentioned
6. Connect related facts across different sentences
7. Extract biographical information (birth dates, nationalities, occupations)
8. Ensure that the JSON output is valid and parsable. Pay close attention to commas, brackets, and quotes.

## Input Format:

['Pargraph Title',
  ['Paragraph 1',
   'Paragraph 2']]

## Output Format:
Return a JSON object with the following structure:
```json
{
"entities": [
    {
        "id": string,
        "name": string,
        "type": string | string[] | null,  // Allow multiple types
        "mentions": [string], 
        "attributes": [                     // New field for entity attributes
            {
                "attribute": string,
                "value": string,
            }
        ]
    }
],
"triples": [
    {
        "subject": string,
        "predicate": string,
        "object": string,
        "category": string,                       // e.g., "biographical", "temporal", "relationship", "attribute"
        "source_sentence_index": ['Pargraph Title',index],         // e.g., ["Pierre Dubois",0]
    }
]
}


Example:

Input: 
['Guerra de Titanes (1998)',
  ['"Guerra de Titanes" (1998) ("War of the Titans") was the second "Guerra de Titanes" professional wrestling show promoted by AAA.',
   ' The show took place on December 13, 1998 in Chihuahua, Chihuahua, Mexico.']],

Output:


```json
{
  "entities": [
    {
      "id": "e1",
      "name": "Guerra de Titanes (1998)",
      "type": ["professional wrestling show"],
      "mentions": ["Guerra de Titanes (1998)", "War of the Titans", "Guerra de Titanes"],
      "attributes": [
        {
          "attribute": "year",
          "value": "1998"
        }
      ]
    },
    {
      "id": "e2",
      "name": "AAA",
      "type": ["organization"],
      "mentions": ["AAA"]
    },
    {
      "id": "e3",
      "name": "Chihuahua",
      "type": ["place"],
      "mentions": ["Chihuahua"]
    }
  ],
  "triples": [
    {
      "subject": "Guerra de Titanes (1998)",
      "predicate": "is_a",
      "object": "professional wrestling show",
      "category": "relationship",
      "source_sentence_index": ["Guerra de Titanes (1998)", 0]
    },
    {
      "subject": "Guerra de Titanes (1998)",
      "predicate": "promoted_by",
      "object": "AAA",
      "category": "relationship",
      "source_sentence_index": ["Guerra de Titanes (1998)", 0]
    },
    {
      "subject": "Guerra de Titanes (1998)",
      "predicate": "took_place_in",
      "object": "Chihuahua",
      "category": "temporal",
      "source_sentence_index": ["Guerra de Titanes (1998)", 1]
    },
        {
      "subject": "Guerra de Titanes (1998)",
      "predicate": "took_place_on",
      "object": "December 13, 1998",
      "category": "temporal",
      "source_sentence_index": ["Guerra de Titanes (1998)", 1]
    }

  ]
}

```
Important Guidelines:

Predicates should be in snake_case format
Use null for unknown values
Assign temporal order numbers based on sequence of events
Include exact source sentences for verification
Link entity mentions to their canonical forms
Include confidence scores for uncertain relationships
Break down compound descriptions into atomic facts
Capture all biographical details as separate triples

Now, please analyze ALL of the following contexts and extract relevant triples from EACH context in the specified JSON format:
{context_str}\n\nIMPORTANT: You must process ALL context items above, not just the first one. Extract entities and triples from EVERY context item.

"""
)

# template_qa_context2triple = (

# # Context Triple Extraction Task

# """


# You are tasked with extracting knowledge triples from context passages to build a comprehensive knowledge graph. Your goal is to identify entities, their relationships, attributes, and temporal information with high granularity.

# ## Task Requirements:
# 1. Extract both explicit and implicit relationships between entities
# 2. Break down complex statements into atomic facts (e.g., "French professor" → "has_nationality" AND "has_occupation")
# 3. Capture temporal ordering of events when present
# 4. Preserve original entity names exactly as they appear
# 5. Include entity types, attributes, and properties when mentioned
# 6. Connect related facts across different sentences
# 7. Extract biographical information (birth dates, nationalities, occupations)

# ## Output Format:
# Return a JSON object with the following structure:
# ```json
# {
# "entities": [
#     {
#         "id": string,
#         "name": string,
#         "type": string | string[] | null,  // Allow multiple types
#         "mentions": [string],
#         "attributes": [                     // New field for entity attributes
#             {
#                 "attribute": string,
#                 "value": string,
#                 "confidence": number
#             }
#         ]
#     }
# ],
# "triples": [
#     {
#         "subject": string,
#         "predicate": string,
#         "object": string,
#         "confidence": number (0-1),
#         "source_sentence": string,
#         "category": string  // e.g., "biographical", "temporal", "relationship", "attribute"
#     }
# ]
# }


# Example:
# Input: "Pierre Dubois (born May 15, 1952) is a French professor at the Sorbonne University who has published extensively on medieval history."
# Output:


# ```json
# {
# "entities": [
#     {
#         "id": "e1",
#         "name": "Pierre Dubois",
#         "type": ["person", "academic"],
#         "mentions": ["Pierre Dubois"],
#         "attributes": [
#             {
#                 "attribute": "nationality",
#                 "value": "French",
#                 "confidence": 1.0
#             },
#             {
#                 "attribute": "birth_date",
#                 "value": "May 15, 1952",
#                 "confidence": 1.0
#             }
#         ]
#     },
#     {
#         "id": "e2",
#         "name": "Sorbonne University",
#         "type": ["university"],
#         "mentions": ["Sorbonne University"]
#     }
# ],
# "triples": [
#     {
#         "subject": "Pierre Dubois",
#         "predicate": "has_nationality",
#         "object": "French",
#         "confidence": 1.0,
#         "source_sentence": "Pierre Dubois (born May 15, 1952) is a French professor at the Sorbonne University who has published extensively on medieval history.",
#         "category": "biographical"
#     },
#     {
#         "subject": "Pierre Dubois",
#         "predicate": "has_occupation",
#         "object": "professor",
#         "confidence": 1.0,
#         "source_sentence": "Pierre Dubois (born May 15, 1952) is a French professor at the Sorbonne University who has published extensively on medieval history.",
#         "category": "biographical"
#     },
#     {
#         "subject": "Pierre Dubois",
#         "predicate": "works_at",
#         "object": "Sorbonne University",
#         "confidence": 1.0,
#         "source_sentence": "Pierre Dubois (born May 15, 1952) is a French professor at the Sorbonne University who has published extensively on medieval history.",
#         "category": "relationship"
#     },
#     {
#         "subject": "Pierre Dubois",
#         "predicate": "born_on",
#         "object": "May 15, 1952",
#         "confidence": 1.0,
#         "source_sentence": "Pierre Dubois (born May 15, 1952) is a French professor at the Sorbonne University who has published extensively on medieval history.",
#         "category": "temporal"
#     },
#     {
#         "subject": "Pierre Dubois",
#         "predicate": "publishes_on",
#         "object": "medieval history",
#         "confidence": 1.0,
#         "source_sentence": "Pierre Dubois (born May 15, 1952) is a French professor at the Sorbonne University who has published extensively on medieval history.",
#         "category": "attribute"
#     }
# ]
# }

# ```
# Important Guidelines:

# Predicates should be in snake_case format
# Use null for unknown values
# Assign temporal order numbers based on sequence of events
# Include exact source sentences for verification
# Link entity mentions to their canonical forms
# Include confidence scores for uncertain relationships
# Break down compound descriptions into atomic facts
# Capture all biographical details as separate triples

# Now, please analyze the following context and extract relevant triples in the specified JSON format:
# {context_str}

# """
# )


template_qa_query2triple = (
# Triple Extraction Task
"""
You are tasked with extracting knowledge triples from questions. A knowledge triple consists of (subject, predicate, object) where:
- Subject: The entity performing the action or being described
- Predicate: The relationship or action
- Object: The target entity or value of the relationship

Your task is to:
1. Identify key entities and relationships in the question
2. Form complete or partial triples (using null for unknown elements)
3. Only extract triples that are directly relevant to answering the question
4. Include both explicit and implicit relationships
5. Preserve the original entity names exactly as they appear

Output Format:
Return a JSON object with the following structure:
```json
{
    "triples": [
        {
            "subject": string | null,
            "predicate": string,
            "object": string | null,
        }
    ]
}
```

Examples:

Input: "Who directed the movie Inception, which was released in 2010?"
Output:
```json
{
    "triples": [
        {
            "subject": null,
            "predicate": "directed",
            "object": "Inception",
        },
        {
            "subject": "Inception",
            "predicate": "released_in",
            "object": "2010",
        }
    ]
}
```

Input: "What university did John Nash teach at before winning the Nobel Prize?"
Output:
```json
{
    "triples": [
        {
            "subject": "John Nash",
            "predicate": "taught_at",
            "object": null,
        },
        {
            "subject": "John Nash",
            "predicate": "won",
            "object": "Nobel Prize",
        },
        {
            "subject": "teaching",
            "predicate": "happened_before",
            "object": "winning Nobel Prize",
        }
    ]
}
```

Important Guidelines:
1. Use null for unknown elements, not "?"
2. Predicates should be in snake_case format
3. Keep entity names exactly as they appear in the question
4. Explanations should be brief but informative

For the given question:
{question_str}
"Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?"

Please directly give me the triples but not give me python code for me.
Please extract relevant triples in the specified JSON format for me:

"""
)


template_qa_query2sparql = (
"""
You are tasked with generating SPARQL queries to answer questions using a knowledge graph. You will be provided with both existing entities and relations from the knowledge graph.

The knowledge graph uses the namespace: "http://example.org/kg/"
All entities and predicates are prefixed with this namespace.

The existing entities and their mentions in the knowledge graph are:
{entities_str}

The existing relations (predicates) in the knowledge graph are:
{relations_str}

Your task is to:
1. Identify key entities and relationships in the question
2. Match these with existing KG entities and relations
3. Form a SPARQL query using exact names from the KG
4. Use proper namespace prefixing

Step-by-step process:
1. First, identify all entities and required relationships in the question
2. Match entities with KG entities (using both names and mentions)
3. Match relationships with existing KG relations
4. Determine what information needs to be queried
5. Construct the SPARQL query using matched entities and relations

Example:

Question: "Who created Milhouse?"
Thinking:
1. Entity in question: "Milhouse" matches KG entity "Milhouse Van Houten"
2. Relationship needed: "created" matches KG relation "created"
3. Need to find the creator, so this will be our variable

SPARQL Query:
PREFIX kg: <http://example.org/kg/>
SELECT DISTINCT ?creator
WHERE {{
    ?creator kg:created kg:Milhouse_Van_Houten .
}}

For the given question:
{question_str}

Please think through each step and provide the final SPARQL query:
"""
)



# Knowledge Graph Alignment Task, the input is a list of dictionaries containing 'entities' and 'triples' with the following structure:

prompt_template_context = PromptTemplate(template_qa_context2triple)
prompt_template_query = PromptTemplate(template_qa_query2sparql)
prompt_template_query2triple = PromptTemplate(template_qa_query2triple)

__all__ = [
    "prompt_template_context",   
    "prompt_template_query",
    "prompt_template_query2triple"
]






# The code seems to work well, but it seems cannot find the answer but only try to locate the given entity and the relation's direct entity.

# Here is the output from the LLM:

# INFO:root:Executing reasoning step 1

# INFO:httpx:HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK"

# INFO:root:LLM Decision: {'should_continue': True, 'answer': None, 'next_action': {'type': 'FOLLOW_RELATION', 'entity': "James Henry Miller's wife", 'relation': 'nationality'}, 'reasoning': "We need to find the nationality of James Henry Miller's wife. Let's follow the relation 'nationality' for this entity."}

# INFO:root:Executing reasoning step 2

# INFO:httpx:HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK"

# INFO:root:LLM Decision: {'should_continue': True, 'answer': None, 'next_action': {'type': 'FOLLOW_RELATION', 'entity': "James Henry Miller's wife", 'relation': 'nationality'}, 'reasoning': "We need to find out the nationality of James Henry Miller's wife, possibly by looking up her name or learning more about James Henry Miller and his family."}

# INFO:root:Executing reasoning step 3

# INFO:httpx:HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK"

# INFO:root:LLM Decision: {'should_continue': True, 'answer': None, 'next_action': {'type': 'FOLLOW_RELATION', 'entity': "James Henry Miller's wife", 'relation': 'nationality'}, 'reasoning': "We need to follow the relation 'nationality' for James Henry Miller's wife to find out her nationality."}

# INFO:root:Executing reasoning step 4

# INFO:httpx:HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK"

# INFO:root:LLM Decision: {'should_continue': True, 'answer': None, 'next_action': {'type': 'QUERY_ENTITY', 'entity': "James Henry Miller's wife"}, 'reasoning': "We need to find out who James Henry Miller's wife was before we can determine her nationality."}

# INFO:root:Executing reasoning step 5

# INFO:httpx:HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK"

# INFO:root:LLM Decision: {'should_continue': True, 'answer': None, 'next_action': {'type': 'FOLLOW_RELATION', 'entity': "James Henry Miller's wife", 'relation': 'nationality'}, 'reasoning': "Given the context, it is likely that James Henry Miller's nationality can provide information about his wife's nationality as well."}

# INFO:root:Reasoning path:

# INFO:root:Step 1: Queried entity: James Henry Miller's wife

# INFO:root:Results: []