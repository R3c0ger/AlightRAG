from __future__ import annotations
from typing import Any


PROMPTS: dict[str, Any] = {}

# All delimiters must be formatted as "<|UPPER_CASE_STRING|>"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|#|>"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

PROMPTS["entity_extraction_system_prompt"] = """---Role---
You are a Knowledge Graph Specialist responsible for extracting entities and relationships from the input text.

---Instructions---
1.  **Entity Extraction & Output:**
    *   **Identification:** Identify clearly defined and meaningful entities in the input text.
    *   **Entity Details:** For each identified entity, extract the following information:
        *   `entity_name`: The name of the entity. If the entity name is case-insensitive, capitalize the first letter of each significant word (title case). Ensure **consistent naming** across the entire extraction process.
        *   `entity_type`: Categorize the entity using one of the following types: `{entity_types}`. If none of the provided entity types apply, do not add new entity type and classify it as `Other`.
        *   `entity_description`: Provide a concise yet comprehensive description of the entity's attributes and activities, based *solely* on the information present in the input text.
    *   **Output Format - Entities:** Output a total of 4 fields for each entity, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `entity`.
        *   Format: `entity{tuple_delimiter}entity_name{tuple_delimiter}entity_type{tuple_delimiter}entity_description`

2.  **Relationship Extraction & Output:**
    *   **Identification:** Identify direct, clearly stated, and meaningful relationships between previously extracted entities.
    *   **N-ary Relationship Decomposition:** If a single statement describes a relationship involving more than two entities (an N-ary relationship), decompose it into multiple binary (two-entity) relationship pairs for separate description.
        *   **Example:** For "Alice, Bob, and Carol collaborated on Project X," extract binary relationships such as "Alice collaborated with Project X," "Bob collaborated with Project X," and "Carol collaborated with Project X," or "Alice collaborated with Bob," based on the most reasonable binary interpretations.
    *   **Relationship Details:** For each binary relationship, extract the following fields:
        *   `source_entity`: The name of the source entity. Ensure **consistent naming** with entity extraction. Capitalize the first letter of each significant word (title case) if the name is case-insensitive.
        *   `target_entity`: The name of the target entity. Ensure **consistent naming** with entity extraction. Capitalize the first letter of each significant word (title case) if the name is case-insensitive.
        *   `relationship_keywords`: One or more high-level keywords summarizing the overarching nature, concepts, or themes of the relationship. Multiple keywords within this field must be separated by a comma `,`. **DO NOT use `{tuple_delimiter}` for separating multiple keywords within this field.**
        *   `relationship_description`: A concise explanation of the nature of the relationship between the source and target entities, providing a clear rationale for their connection.
    *   **Output Format - Relationships:** Output a total of 5 fields for each relationship, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `relation`.
        *   Format: `relation{tuple_delimiter}source_entity{tuple_delimiter}target_entity{tuple_delimiter}relationship_keywords{tuple_delimiter}relationship_description`

3.  **Delimiter Usage Protocol:**
    *   The `{tuple_delimiter}` is a complete, atomic marker and **must not be filled with content**. It serves strictly as a field separator.
    *   **Incorrect Example:** `entity{tuple_delimiter}Tokyo<|location|>Tokyo is the capital of Japan.`
    *   **Correct Example:** `entity{tuple_delimiter}Tokyo{tuple_delimiter}location{tuple_delimiter}Tokyo is the capital of Japan.`

4.  **Relationship Direction & Duplication:**
    *   Treat all relationships as **undirected** unless explicitly stated otherwise. Swapping the source and target entities for an undirected relationship does not constitute a new relationship.
    *   Avoid outputting duplicate relationships.

5.  **Output Order & Prioritization:**
    *   Output all extracted entities first, followed by all extracted relationships.
    *   Within the list of relationships, prioritize and output those relationships that are **most significant** to the core meaning of the input text first.

6.  **Context & Objectivity:**
    *   Ensure all entity names and descriptions are written in the **third person**.
    *   Explicitly name the subject or object; **avoid using pronouns** such as `this article`, `this paper`, `our company`, `I`, `you`, and `he/she`.

7.  **Language & Proper Nouns:**
    *   The entire output (entity names, keywords, and descriptions) must be written in `{language}`.
    *   Proper nouns (e.g., personal names, place names, organization names) should be retained in their original language if a proper, widely accepted translation is not available or would cause ambiguity.

8.  **Completion Signal:** Output the literal string `{completion_delimiter}` only after all entities and relationships, following all criteria, have been completely extracted and outputted.

---Examples---
{examples}

---Real Data to be Processed---
<Input>
Entity_types: [{entity_types}]
Text:
```
{input_text}
```
"""

PROMPTS["entity_extraction_user_prompt"] = """---Task---
Extract entities and relationships from the input text to be processed.

---Instructions---
1.  **Strict Adherence to Format:** Strictly adhere to all format requirements for entity and relationship lists, including output order, field delimiters, and proper noun handling, as specified in the system prompt.
2.  **Output Content Only:** Output *only* the extracted list of entities and relationships. Do not include any introductory or concluding remarks, explanations, or additional text before or after the list.
3.  **Completion Signal:** Output `{completion_delimiter}` as the final line after all relevant entities and relationships have been extracted and presented.
4.  **Output Language:** Ensure the output language is {language}. Proper nouns (e.g., personal names, place names, organization names) must be kept in their original language and not translated.

<Output>
"""

PROMPTS["entity_continue_extraction_user_prompt"] = """---Task---
Based on the last extraction task, identify and extract any **missed or incorrectly formatted** entities and relationships from the input text.

---Instructions---
1.  **Strict Adherence to System Format:** Strictly adhere to all format requirements for entity and relationship lists, including output order, field delimiters, and proper noun handling, as specified in the system instructions.
2.  **Focus on Corrections/Additions:**
    *   **Do NOT** re-output entities and relationships that were **correctly and fully** extracted in the last task.
    *   If an entity or relationship was **missed** in the last task, extract and output it now according to the system format.
    *   If an entity or relationship was **truncated, had missing fields, or was otherwise incorrectly formatted** in the last task, re-output the *corrected and complete* version in the specified format.
3.  **Output Format - Entities:** Output a total of 4 fields for each entity, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `entity`.
4.  **Output Format - Relationships:** Output a total of 5 fields for each relationship, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `relation`.
5.  **Output Content Only:** Output *only* the extracted list of entities and relationships. Do not include any introductory or concluding remarks, explanations, or additional text before or after the list.
6.  **Completion Signal:** Output `{completion_delimiter}` as the final line after all relevant missing or corrected entities and relationships have been extracted and presented.
7.  **Output Language:** Ensure the output language is {language}. Proper nouns (e.g., personal names, place names, organization names) must be kept in their original language and not translated.

<Output>
"""

PROMPTS["entity_extraction_examples"] = [
    """<Input Text>
```
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.

Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. "If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us."

The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.

It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths
```

<Output>
entity{tuple_delimiter}Alex{tuple_delimiter}person{tuple_delimiter}Alex is a character who experiences frustration and is observant of the dynamics among other characters.
entity{tuple_delimiter}Taylor{tuple_delimiter}person{tuple_delimiter}Taylor is portrayed with authoritarian certainty and shows a moment of reverence towards a device, indicating a change in perspective.
entity{tuple_delimiter}Jordan{tuple_delimiter}person{tuple_delimiter}Jordan shares a commitment to discovery and has a significant interaction with Taylor regarding a device.
entity{tuple_delimiter}Cruz{tuple_delimiter}person{tuple_delimiter}Cruz is associated with a vision of control and order, influencing the dynamics among other characters.
entity{tuple_delimiter}The Device{tuple_delimiter}equipment{tuple_delimiter}The Device is central to the story, with potential game-changing implications, and is revered by Taylor.
relation{tuple_delimiter}Alex{tuple_delimiter}Taylor{tuple_delimiter}power dynamics, observation{tuple_delimiter}Alex observes Taylor's authoritarian behavior and notes changes in Taylor's attitude toward the device.
relation{tuple_delimiter}Alex{tuple_delimiter}Jordan{tuple_delimiter}shared goals, rebellion{tuple_delimiter}Alex and Jordan share a commitment to discovery, which contrasts with Cruz's vision.)
relation{tuple_delimiter}Taylor{tuple_delimiter}Jordan{tuple_delimiter}conflict resolution, mutual respect{tuple_delimiter}Taylor and Jordan interact directly regarding the device, leading to a moment of mutual respect and an uneasy truce.
relation{tuple_delimiter}Jordan{tuple_delimiter}Cruz{tuple_delimiter}ideological conflict, rebellion{tuple_delimiter}Jordan's commitment to discovery is in rebellion against Cruz's vision of control and order.
relation{tuple_delimiter}Taylor{tuple_delimiter}The Device{tuple_delimiter}reverence, technological significance{tuple_delimiter}Taylor shows reverence towards the device, indicating its importance and potential impact.
{completion_delimiter}

""",
    """<Input Text>
```
Stock markets faced a sharp downturn today as tech giants saw significant declines, with the global tech index dropping by 3.4% in midday trading. Analysts attribute the selloff to investor concerns over rising interest rates and regulatory uncertainty.

Among the hardest hit, nexon technologies saw its stock plummet by 7.8% after reporting lower-than-expected quarterly earnings. In contrast, Omega Energy posted a modest 2.1% gain, driven by rising oil prices.

Meanwhile, commodity markets reflected a mixed sentiment. Gold futures rose by 1.5%, reaching $2,080 per ounce, as investors sought safe-haven assets. Crude oil prices continued their rally, climbing to $87.60 per barrel, supported by supply constraints and strong demand.

Financial experts are closely watching the Federal Reserve's next move, as speculation grows over potential rate hikes. The upcoming policy announcement is expected to influence investor confidence and overall market stability.
```

<Output>
entity{tuple_delimiter}Global Tech Index{tuple_delimiter}category{tuple_delimiter}The Global Tech Index tracks the performance of major technology stocks and experienced a 3.4% decline today.
entity{tuple_delimiter}Nexon Technologies{tuple_delimiter}organization{tuple_delimiter}Nexon Technologies is a tech company that saw its stock decline by 7.8% after disappointing earnings.
entity{tuple_delimiter}Omega Energy{tuple_delimiter}organization{tuple_delimiter}Omega Energy is an energy company that gained 2.1% in stock value due to rising oil prices.
entity{tuple_delimiter}Gold Futures{tuple_delimiter}product{tuple_delimiter}Gold futures rose by 1.5%, indicating increased investor interest in safe-haven assets.
entity{tuple_delimiter}Crude Oil{tuple_delimiter}product{tuple_delimiter}Crude oil prices rose to $87.60 per barrel due to supply constraints and strong demand.
entity{tuple_delimiter}Market Selloff{tuple_delimiter}category{tuple_delimiter}Market selloff refers to the significant decline in stock values due to investor concerns over interest rates and regulations.
entity{tuple_delimiter}Federal Reserve Policy Announcement{tuple_delimiter}category{tuple_delimiter}The Federal Reserve's upcoming policy announcement is expected to impact investor confidence and market stability.
entity{tuple_delimiter}3.4% Decline{tuple_delimiter}category{tuple_delimiter}The Global Tech Index experienced a 3.4% decline in midday trading.
relation{tuple_delimiter}Global Tech Index{tuple_delimiter}Market Selloff{tuple_delimiter}market performance, investor sentiment{tuple_delimiter}The decline in the Global Tech Index is part of the broader market selloff driven by investor concerns.
relation{tuple_delimiter}Nexon Technologies{tuple_delimiter}Global Tech Index{tuple_delimiter}company impact, index movement{tuple_delimiter}Nexon Technologies' stock decline contributed to the overall drop in the Global Tech Index.
relation{tuple_delimiter}Gold Futures{tuple_delimiter}Market Selloff{tuple_delimiter}market reaction, safe-haven investment{tuple_delimiter}Gold prices rose as investors sought safe-haven assets during the market selloff.
relation{tuple_delimiter}Federal Reserve Policy Announcement{tuple_delimiter}Market Selloff{tuple_delimiter}interest rate impact, financial regulation{tuple_delimiter}Speculation over Federal Reserve policy changes contributed to market volatility and investor selloff.
{completion_delimiter}

""",
    """<Input Text>
```
At the World Athletics Championship in Tokyo, Noah Carter broke the 100m sprint record using cutting-edge carbon-fiber spikes.
```

<Output>
entity{tuple_delimiter}World Athletics Championship{tuple_delimiter}event{tuple_delimiter}The World Athletics Championship is a global sports competition featuring top athletes in track and field.
entity{tuple_delimiter}Tokyo{tuple_delimiter}location{tuple_delimiter}Tokyo is the host city of the World Athletics Championship.
entity{tuple_delimiter}Noah Carter{tuple_delimiter}person{tuple_delimiter}Noah Carter is a sprinter who set a new record in the 100m sprint at the World Athletics Championship.
entity{tuple_delimiter}100m Sprint Record{tuple_delimiter}category{tuple_delimiter}The 100m sprint record is a benchmark in athletics, recently broken by Noah Carter.
entity{tuple_delimiter}Carbon-Fiber Spikes{tuple_delimiter}equipment{tuple_delimiter}Carbon-fiber spikes are advanced sprinting shoes that provide enhanced speed and traction.
entity{tuple_delimiter}World Athletics Federation{tuple_delimiter}organization{tuple_delimiter}The World Athletics Federation is the governing body overseeing the World Athletics Championship and record validations.
relation{tuple_delimiter}World Athletics Championship{tuple_delimiter}Tokyo{tuple_delimiter}event location, international competition{tuple_delimiter}The World Athletics Championship is being hosted in Tokyo.
relation{tuple_delimiter}Noah Carter{tuple_delimiter}100m Sprint Record{tuple_delimiter}athlete achievement, record-breaking{tuple_delimiter}Noah Carter set a new 100m sprint record at the championship.
relation{tuple_delimiter}Noah Carter{tuple_delimiter}Carbon-Fiber Spikes{tuple_delimiter}athletic equipment, performance boost{tuple_delimiter}Noah Carter used carbon-fiber spikes to enhance performance during the race.
relation{tuple_delimiter}Noah Carter{tuple_delimiter}World Athletics Championship{tuple_delimiter}athlete participation, competition{tuple_delimiter}Noah Carter is competing at the World Athletics Championship.
{completion_delimiter}

""",
]

PROMPTS["summarize_entity_descriptions"] = """---Role---
You are a Knowledge Graph Specialist, proficient in data curation and synthesis.

---Task---
Your task is to synthesize a list of descriptions of a given entity or relation into a single, comprehensive, and cohesive summary.

---Instructions---
1. Input Format: The description list is provided in JSON format. Each JSON object (representing a single description) appears on a new line within the `Description List` section.
2. Output Format: The merged description will be returned as plain text, presented in multiple paragraphs, without any additional formatting or extraneous comments before or after the summary.
3. Comprehensiveness: The summary must integrate all key information from *every* provided description. Do not omit any important facts or details.
4. Context: Ensure the summary is written from an objective, third-person perspective; explicitly mention the name of the entity or relation for full clarity and context.
5. Context & Objectivity:
  - Write the summary from an objective, third-person perspective.
  - Explicitly mention the full name of the entity or relation at the beginning of the summary to ensure immediate clarity and context.
6. Conflict Handling:
  - In cases of conflicting or inconsistent descriptions, first determine if these conflicts arise from multiple, distinct entities or relationships that share the same name.
  - If distinct entities/relations are identified, summarize each one *separately* within the overall output.
  - If conflicts within a single entity/relation (e.g., historical discrepancies) exist, attempt to reconcile them or present both viewpoints with noted uncertainty.
7. Length Constraint:The summary's total length must not exceed {summary_length} tokens, while still maintaining depth and completeness.
8. Language: The entire output must be written in {language}. Proper nouns (e.g., personal names, place names, organization names) may in their original language if proper translation is not available.
  - The entire output must be written in {language}.
  - Proper nouns (e.g., personal names, place names, organization names) should be retained in their original language if a proper, widely accepted translation is not available or would cause ambiguity.

---Input---
{description_type} Name: {description_name}

Description List:

```
{description_list}
```

---Output---
"""

PROMPTS["fail_response"] = (
    "Sorry, I'm not able to provide an answer to that question.[no-context]"
)

PROMPTS["rag_response"] = """---Role---

You are an expert AI assistant specializing in synthesizing information from a provided knowledge base. Your primary function is to answer user queries accurately by ONLY using the information within the provided **Context**.

---Goal---

Generate a comprehensive, well-structured answer to the user query.
The answer must integrate relevant facts from the Knowledge Graph and Document Chunks found in the **Context**.
Consider the conversation history if provided to maintain conversational flow and avoid repeating information.

---Instructions---

1. Step-by-Step Instruction:
  - Carefully determine the user's query intent in the context of the conversation history to fully understand the user's information need.
  - Scrutinize both `Knowledge Graph Data` and `Document Chunks` in the **Context**. Identify and extract all pieces of information that are directly relevant to answering the user query.
  - Weave the extracted facts into a coherent and logical response. Your own knowledge must ONLY be used to formulate fluent sentences and connect ideas, NOT to introduce any external information.
  - Track the reference_id of the document chunk which directly support the facts presented in the response. Correlate reference_id with the entries in the `Reference Document List` to generate the appropriate citations.
  - Generate a references section at the end of the response. Each reference document must directly support the facts presented in the response.
  - Do not generate anything after the reference section.

2. Content & Grounding:
  - Strictly adhere to the provided context from the **Context**; DO NOT invent, assume, or infer any information not explicitly stated.
  - If the answer cannot be found in the **Context**, state that you do not have enough information to answer. Do not attempt to guess.

3. Formatting & Language:
  - The response MUST be in the same language as the user query.
  - The response MUST utilize Markdown formatting for enhanced clarity and structure (e.g., headings, bold text, bullet points).
  - The response should be presented in {response_type}.

4. References Section Format:
  - The References section should be under heading: `### References`
  - Reference list entries should adhere to the format: `* [n] Document Title`. Do not include a caret (`^`) after opening square bracket (`[`).
  - The Document Title in the citation must retain its original language.
  - Output each citation on an individual line
  - Provide maximum of 5 most relevant citations.
  - Do not generate footnotes section or any comment, summary, or explanation after the references.

5. Reference Section Example:
```
### References

- [1] Document Title One
- [2] Document Title Two
- [3] Document Title Three
```

6. Additional Instructions: {user_prompt}


---Context---

{context_data}
"""

PROMPTS["naive_rag_response"] = """---Role---

You are an expert AI assistant specializing in synthesizing information from a provided knowledge base. Your primary function is to answer user queries accurately by ONLY using the information within the provided **Context**.

---Goal---

Generate a comprehensive, well-structured answer to the user query.
The answer must integrate relevant facts from the Document Chunks found in the **Context**.
Consider the conversation history if provided to maintain conversational flow and avoid repeating information.

---Instructions---

1. Step-by-Step Instruction:
  - Carefully determine the user's query intent in the context of the conversation history to fully understand the user's information need.
  - Scrutinize `Document Chunks` in the **Context**. Identify and extract all pieces of information that are directly relevant to answering the user query.
  - Weave the extracted facts into a coherent and logical response. Your own knowledge must ONLY be used to formulate fluent sentences and connect ideas, NOT to introduce any external information.
  - Track the reference_id of the document chunk which directly support the facts presented in the response. Correlate reference_id with the entries in the `Reference Document List` to generate the appropriate citations.
  - Generate a **References** section at the end of the response. Each reference document must directly support the facts presented in the response.
  - Do not generate anything after the reference section.

2. Content & Grounding:
  - Strictly adhere to the provided context from the **Context**; DO NOT invent, assume, or infer any information not explicitly stated.
  - If the answer cannot be found in the **Context**, state that you do not have enough information to answer. Do not attempt to guess.

3. Formatting & Language:
  - The response MUST be in the same language as the user query.
  - The response MUST utilize Markdown formatting for enhanced clarity and structure (e.g., headings, bold text, bullet points).
  - The response should be presented in {response_type}.

4. References Section Format:
  - The References section should be under heading: `### References`
  - Reference list entries should adhere to the format: `* [n] Document Title`. Do not include a caret (`^`) after opening square bracket (`[`).
  - The Document Title in the citation must retain its original language.
  - Output each citation on an individual line
  - Provide maximum of 5 most relevant citations.
  - Do not generate footnotes section or any comment, summary, or explanation after the references.

5. Reference Section Example:
```
### References

- [1] Document Title One
- [2] Document Title Two
- [3] Document Title Three
```

6. Additional Instructions: {user_prompt}


---Context---

{content_data}
"""

PROMPTS["kg_query_context"] = """
Knowledge Graph Data (Entity):

```json
{entities_str}
```

Knowledge Graph Data (Relationship):

```json
{relations_str}
```

Document Chunks (Each entry has a reference_id refer to the `Reference Document List`):

```json
{text_chunks_str}
```

Reference Document List (Each entry starts with a [reference_id] that corresponds to entries in the Document Chunks):

```
{reference_list_str}
```

"""

PROMPTS["naive_query_context"] = """
Document Chunks (Each entry has a reference_id refer to the `Reference Document List`):

```json
{text_chunks_str}
```

Reference Document List (Each entry starts with a [reference_id] that corresponds to entries in the Document Chunks):

```
{reference_list_str}
```

"""

PROMPTS["keywords_extraction"] = """---Role---
You are an expert keyword extractor, specializing in analyzing user queries for a Retrieval-Augmented Generation (RAG) system. Your purpose is to identify both high-level and low-level keywords in the user's query that will be used for effective document retrieval.

---Goal---
Given a user query, your task is to extract two distinct types of keywords:
1. **high_level_keywords**: for overarching concepts or themes, capturing user's core intent, the subject area, or the type of question being asked.
2. **low_level_keywords**: for specific entities or details, identifying the specific entities, proper nouns, technical jargon, product names, or concrete items.

---Instructions & Constraints---
1. **Output Format**: Your output MUST be a valid JSON object and nothing else. Do not include any explanatory text, markdown code fences (like ```json), or any other text before or after the JSON. It will be parsed directly by a JSON parser.
2. **Source of Truth**: All keywords must be explicitly derived from the user query, with both high-level and low-level keyword categories are required to contain content.
3. **Concise & Meaningful**: Keywords should be concise words or meaningful phrases. Prioritize multi-word phrases when they represent a single concept. For example, from "latest financial report of Apple Inc.", you should extract "latest financial report" and "Apple Inc." rather than "latest", "financial", "report", and "Apple".
4. **Handle Edge Cases**: For queries that are too simple, vague, or nonsensical (e.g., "hello", "ok", "asdfghjkl"), you must return a JSON object with empty lists for both keyword types.

---Examples---
{examples}

---Real Data---
User Query: {query}

---Output---
Output:"""

PROMPTS["keywords_extraction_examples"] = [
    """Example 1:

Query: "How does international trade influence global economic stability?"

Output:
{
  "high_level_keywords": ["International trade", "Global economic stability", "Economic impact"],
  "low_level_keywords": ["Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"]
}

""",
    """Example 2:

Query: "What are the environmental consequences of deforestation on biodiversity?"

Output:
{
  "high_level_keywords": ["Environmental consequences", "Deforestation", "Biodiversity loss"],
  "low_level_keywords": ["Species extinction", "Habitat destruction", "Carbon emissions", "Rainforest", "Ecosystem"]
}

""",
    """Example 3:

Query: "What is the role of education in reducing poverty?"

Output:
{
  "high_level_keywords": ["Education", "Poverty reduction", "Socioeconomic development"],
  "low_level_keywords": ["School access", "Literacy rates", "Job training", "Income inequality"]
}

""",
]

# alightrag-insert TODO
PROMPTS["alightrag_reasoning"] = """
You are an expert in knowledge graph reasoning. Your task is to analyze a given question and construct one or more relation paths that can logically answer it. Each path must be a chain in the format: entity -> relationship -> entity -> ... -> entity, where the final entity in the path represents the potential answer to the question. 

### Key Instructions:
- Base the paths strictly on the provided entities and relationships retrieved from the knowledge graph. Do not invent new entities, relationships, or assumptions.
- Entities are provided as a comma-separated list of `entity_name`s (e.g., "Entity1, Entity2, Entity3"). Use only these exact names (in title case, as extracted).
- Relationships are provided as a list of triples in the format "(source_entity, relationship_keywords, target_entity)", separated by semicolons (e.g., "(EntityA, rel1, EntityB); (EntityB, rel2, EntityC)"). Use `relationship_keywords` (comma-separated if multiple) as the relationship label in paths; treat multi-keyword labels as a single unit (e.g., "directed by, produced by").
- Identify starting entities relevant to the question's subject (e.g., matching the question's focus like a person, place, or event), then chain relationships and entities to reach an answer entity.
- Paths can be 0-hop (a single entity), 1-hop or multi-hop, but keep them as concise as possible while ensuring they fully address the question. Decompose N-ary relationships if needed by chaining binary pairs.
- If multiple paths are possible, list up to 3 of the most relevant ones, prioritizing those most significant to the core meaning of the question.
- If no valid path can be constructed, output "No valid paths found."
- Focus on paths that directly lead to the answer; avoid redundant, circular, or undirected swaps (treat relationships as undirected unless direction is implied by the question).
- Ensure consistent entity naming (title case) and third-person objectivity.

### Output Format:
Output your response in the following strict raw JSON format, with no additional text, do NOT wrap your response in code blocks (no ```json or ```):
{
  "paths": [
    "Entity1",
    "Entity2 -> rel1 -> Entity3",
    "Entity4 -> rel2 -> Entity5 -> rel3 -> Entity6"
  ],
  "explanation": "Brief explanation of how the paths answer the question."
}

### Few-Shot Examples:

Example 1:
Entities: Inception, Christopher Nolan, Film
Relationships: (Inception, directed by, Christopher Nolan); (Inception, is a, Film)
Question: Who is the director of Inception?
Output:
{
  "paths": [
    "Inception -> directed by -> Christopher Nolan"
  ],
  "explanation": "The path directly connects the movie Inception to its director via the 'directed by' relationship, answering who directed it."
}

Example 2:
Entities: Eiffel Tower, Paris, France
Relationships: (Eiffel Tower, located in, Paris); (Paris, is capital of, France)
Question: What is the capital of the country where the Eiffel Tower is located?
Output:
{
  "paths": [
    "Eiffel Tower -> located in -> Paris",
    "Eiffel Tower -> located in -> Paris -> is capital of -> France"
  ],
  "explanation": "The first path finds the city (Paris) where the Eiffel Tower is, which is the capital. The second path extends to the country (France) for context, confirming Paris as its capital."
}

Example 3:
Entities: Roger Penrose, Reinhard Genzel, Andrea Ghez, Nobel Prize in Physics, 2020
Relationships: (Nobel Prize in Physics, awarded in, 2020); (Nobel Prize in Physics, won by, Roger Penrose); (Nobel Prize in Physics, won by, Reinhard Genzel); (Nobel Prize in Physics, won by, Andrea Ghez)
Question: Who won the Nobel Prize in Physics in 2020?
Output:
{
  "paths": [
    "Nobel Prize in Physics -> awarded in -> 2020 -> won by -> Roger Penrose",
    "Nobel Prize in Physics -> awarded in -> 2020 -> won by -> Reinhard Genzel",
    "Nobel Prize in Physics -> awarded in -> 2020 -> won by -> Andrea Ghez"
  ],
  "explanation": "Multiple paths connect the 2020 Nobel Prize in Physics to its co-winners via the 'won by' relationship."
}
"""

PROMPTS["alightrag_reasoning_query"] = """
Now, construct the paths for the following:
Entities: {entities}
Relationships: {relationships}
Question: {question}
"""

PROMPTS["alightrag_reflection"] = """
You are an expert in knowledge graph validation. Your task is to evaluate one or more proposed relation paths for a given question, determining if they are valid based on the provided entities and relationships retrieved from the knowledge graph. A valid path must adhere to these rules:

1. Fidelity to KG: Each link in the path (entity -> relationship -> entity) must directly correspond to an existing triple in the provided relationships. Do not allow invented, inferred, or external knowledge—stick strictly to the given entities and relationships.
2. Coherence: The target entity of one link must exactly match the source entity of the next link (case-sensitive, using title case as provided). No gaps, cycles, or mismatches.
3. Relevance to Question: The path must logically chain from a starting entity relevant to the question's subject to a final entity that plausibly represents or leads to the answer. The final entity should directly address the question (e.g., be the person, place, or thing asked for).

- Evaluate each path independently.
- If a path violates any rule, it is invalid.
- For multiple paths, output a list of validated paths (one per path).
- Base validation solely on the provided entities and relationships—assume they are accurate extractions from the source text, not real-world facts.
- After validation, filter the entities and relationships to retain only those that directly appear in the valid paths (if any). Entities: Collect unique entity_names (in title case) from all valid paths. Relationships: Collect unique triples (as "(source_entity, relationship_keywords, target_entity)") that match links in valid paths.
- Finally, assess if the valid paths (if any), combined with the filtered entities and relationships, sufficiently answer the question. If not (e.g., no valid paths, partial coverage, or missing key details), you must raise 1-3 supplementary questions that could elicit additional entities/relationships needed to fully answer the original question.

### Output Format:
Output your response in the following strict raw JSON format, with no additional text, do NOT wrap your response in code blocks (no ```json or ```):
{
  "validated_paths": [
    {
      "path": "Original path string",
      "is_valid": true/false,
      "reason": "Brief explanation of why the path is valid or invalid, referencing the rules."
    }
  ],
  "filtered_entities": "Comma-separated list of unique entities from valid paths (or empty string if none).",
  "filtered_relationships": "Semicolon-separated list of unique triples from valid paths (or empty string if none).",
  "overall_explanation": "Summary of how the valid paths (if any) collectively answer the question, or why none do and what is missing.",
  "supplementary_questions": ["If insufficient, list 1-3 new questions here as strings; otherwise, omit this key."]
}

### Few-Shot Examples:

Example 1:
Entities: Inception, Christopher Nolan, Film
Relationships: (Inception, directed by, Christopher Nolan); (Inception, is a, Film)
Proposed Paths: 
["Inception -> directed by -> Christopher Nolan"]
Question: Who is the director of Inception?
Output:
{
  "validated_paths": [
    {
      "path": "Inception -> directed by -> Christopher Nolan",
      "is_valid": true,
      "reason": "Fidelity: Matches the triple (Inception, directed by, Christopher Nolan). Coherence: Single-hop, no mismatches. Relevance: Starts with the movie and ends with the director, directly answering the question."
    }
  ],
  "filtered_entities": "Inception, Christopher Nolan",
  "filtered_relationships": "(Inception, directed by, Christopher Nolan)",
  "overall_explanation": "The valid path connects the movie to its director, providing the answer entity 'Christopher Nolan'."
}

Example 2:
Entities: Eiffel Tower, Paris, France
Relationships: (Eiffel Tower, located in, Paris); (Paris, is capital of, France)
Proposed Paths: 
["Eiffel Tower -> located in -> Paris", "Eiffel Tower -> located in -> Paris -> is capital of -> France"]
Question: What is the capital of the country where the Eiffel Tower is located?
Output:
{
  "validated_paths": [
    {
      "path": "Eiffel Tower -> located in -> Paris",
      "is_valid": true,
      "reason": "Fidelity: Matches (Eiffel Tower, located in, Paris). Coherence: Single-hop. Relevance: Ends with Paris, which is the capital city answering the question."
    },
    {
      "path": "Eiffel Tower -> located in -> Paris -> is capital of -> France",
      "is_valid": true,
      "reason": "Fidelity: Matches both triples. Coherence: Paris links the hops. Relevance: Extends to France for context, but final entity France confirms the country, with Paris implied as capital."
    }
  ],
  "filtered_entities": "Eiffel Tower, Paris, France",
  "filtered_relationships": "(Eiffel Tower, located in, Paris); (Paris, is capital of, France)",
  "overall_explanation": "Both paths logically derive Paris as the capital, with the second providing additional country context."
}

Example 3:
Entities: Roger Penrose, Reinhard Genzel, Andrea Ghez, Nobel Prize in Physics, 2020
Relationships: (Nobel Prize in Physics, awarded in, 2020); (Nobel Prize in Physics, won by, Roger Penrose); (Nobel Prize in Physics, won by, Reinhard Genzel); (Nobel Prize in Physics, won by, Andrea Ghez)
Proposed Paths: 
["Nobel Prize in Physics -> awarded in -> 2020 -> won by -> Roger Penrose", "Nobel Prize in Physics -> awarded in -> 2020 -> won by -> Reinhard Genzel", "Nobel Prize in Physics -> awarded in -> 2020 -> won by -> Andrea Ghez"]
Question: Who won the Nobel Prize in Physics in 2020?
Output:
{
  "validated_paths": [
    {
      "path": "Nobel Prize in Physics -> awarded in -> 2020 -> won by -> Roger Penrose",
      "is_valid": false,
      "reason": "Fidelity: 'won by -> Roger Penrose' matches, but chaining '2020 -> won by' does not correspond to an existing triple—2020 is not a source entity for 'won by'. Coherence: Mismatch in linking 2020 directly to winners."
    },
    {
      "path": "Nobel Prize in Physics -> awarded in -> 2020 -> won by -> Reinhard Genzel",
      "is_valid": false,
      "reason": "Same as above: No direct triple from 2020 as source for 'won by'."
    },
    {
      "path": "Nobel Prize in Physics -> awarded in -> 2020 -> won by -> Andrea Ghez",
      "is_valid": false,
      "reason": "Same as above: No direct triple from 2020 as source for 'won by'."
    }
  ],
  "filtered_entities": "",
  "filtered_relationships": "",
  "overall_explanation": "None of the paths are valid due to improper chaining; the data links the prize to 2020 and winners separately, but no paths directly connect them to confirm 2020-specific winners, leaving the question unanswered.",
  "supplementary_questions": [
    "Who specifically won the Nobel Prize in Physics that was awarded in 2020?",
    "Are there relationships connecting the 2020 award year directly to the winners?"
  ]
}
"""

PROMPTS["alightrag_reflection_query"] = """
Now, validate the following:
Entities: {entities}
Relationships: {relationships}
Proposed Paths: {paths}
Question: {question}
"""

PROMPTS["alightrag_response"] = """
You are an expert in knowledge graph question answering. Your task is to synthesize a final, comprehensive response to a given question using ONLY the provided entities, relationships, validated relation paths from the knowledge graph and document chunks. Do not introduce external knowledge, assumptions, or inferences—stick strictly to the given data.

### Key Instructions:
1. Analyze the question to determine its intent (e.g., who, what, where, why, how).
2. Use the provided entities, relationships, validated paths, and document chunks as the sole knowledge base.
3. Leverage the validated paths to derive the answer:
   - If multiple paths exist, integrate them coherently
   - If no valid paths are provided, state that insufficient information is available
4. Ground every claim in the response directly in one or more paths, entities, relationships or document chunks
5. Keep the response concise, objective, and in third-person
6. If the question cannot be fully answered based on the data, acknowledge limitations honestly
7. Track the reference_id of the document chunks which directly support the facts presented
8. Generate a references section at the end of the response using the Reference Document List
9. Do not generate anything after the reference section

### Output Format:
Your response MUST follow this exact structure:

# Answer
[Direct, concise answer to the question, followed by detailed reasoning steps based on the reasoning paths]

## Reasoning Paths
- Path 1: [Copy the first path exactly as provided]  
  Explanation: [Briefly explain how this path derives the answer, referencing specific entities, relationships, and document chunks when applicable.]
- Path 2: [If applicable, copy the second path]  
  Explanation: [Brief explanation.]
- [Continue for additional paths]

### References [If there are no references, DON'T include this section]
[Format each reference on its own line as shown below, using the Reference Document List provided]

### Reference Format Rules:
1. Use heading: `### References` (exactly three hash symbols)
2. Each reference must be in the format: `* [n] Document Title: Additional context details`
   - Replace `n` with the reference number from the Reference Document List
   - `Document Title` must match exactly from the Reference Document List
   - Add a colon and brief context details about how this document supports the answer
3. Include only references that directly support facts in your answer
4. Maximum 5 references
5. Do not include any footnotes, summaries, or explanations after the references

### Few-Shot Examples:

Example 1:
Question: Who is the director of Inception?
Entities: Inception, Christopher Nolan, Film
Relationships: (Inception, directed by, Christopher Nolan); (Inception, is a, Film)
Validated Paths: ["Inception -> directed by -> Christopher Nolan"]
Document Chunks: [{"reference_id": "1", "content": "The film Inception was directed by Christopher Nolan..."}]
Reference Document List: [1] film_database.txt
Output:
# Answer
Christopher Nolan.

## Reasoning Paths
- Path 1: Inception -> directed by -> Christopher Nolan  
  Explanation: This path starts with the film Inception and directly links to its director via the 'directed by' relationship, identifying Christopher Nolan as the answer entity. Supported by document [1].

### References
* [1] film_database.txt: Provides details about the film Inception and its director Christopher Nolan.

Example 2:
Question: What is the capital of the country where the Eiffel Tower is located?
Entities: Eiffel Tower, Paris, France
Relationships: (Eiffel Tower, located in, Paris); (Paris, is capital of, France)
Validated Paths: ["Eiffel Tower -> located in -> Paris", "Eiffel Tower -> located in -> Paris -> is capital of -> France"]
Document Chunks: [{"reference_id": "2", "content": "The Eiffel Tower is located in Paris, France..."}, {"reference_id": "3", "content": "Paris serves as the capital city of France..."}]
Reference Document List: [2] geography_guide.txt; [3] capitals_database.txt
Output:
# Answer
Paris.

## Reasoning Paths
- Path 1: Eiffel Tower -> located in -> Paris  
  Explanation: This path locates the Eiffel Tower in Paris, which serves as the capital city answering the question. Supported by document [2].
- Path 2: Eiffel Tower -> located in -> Paris -> is capital of -> France  
  Explanation: This extended path confirms Paris as the capital of France, providing additional context. Supported by documents [2] and [3].

### References
* [2] geography_guide.txt: Documents the location of the Eiffel Tower in Paris.
* [3] capitals_database.txt: Confirms Paris as the capital of France.

Example 3:
Question: Who won the Nobel Prize in Physics in 2020?
Entities: Roger Penrose, Reinhard Genzel, Andrea Ghez, Nobel Prize in Physics, 2020
Relationships: (Nobel Prize in Physics, awarded in, 2020); (Nobel Prize in Physics, won by, Roger Penrose); (Nobel Prize in Physics, won by, Reinhard Genzel); (Nobel Prize in Physics, won by, Andrea Ghez)
Validated Paths: ["Nobel Prize in Physics -> won by -> Roger Penrose", "Nobel Prize in Physics -> won by -> Reinhard Genzel", "Nobel Prize in Physics -> won by -> Andrea Ghez"]
Document Chunks: [{"reference_id": "4", "content": "The 2020 Nobel Prize in Physics was awarded jointly to Roger Penrose, Reinhard Genzel, and Andrea Ghez..."}]
Reference Document List: [4] nobel_prize_records.txt
Output:
# Answer
Roger Penrose, Reinhard Genzel, and Andrea Ghez (co-winners).

## Reasoning Paths
- Path 1: Nobel Prize in Physics -> won by -> Roger Penrose  
  Explanation: This path links the prize to Roger Penrose as a winner via the 'won by' relationship. Supported by document [4].
- Path 2: Nobel Prize in Physics -> won by -> Reinhard Genzel  
  Explanation: This path identifies Reinhard Genzel as another winner. Supported by document [4].
- Path 3: Nobel Prize in Physics -> won by -> Andrea Ghez  
  Explanation: This path identifies Andrea Ghez as the third winner. Supported by document [4].

### References
* [4] nobel_prize_records.txt: Documents the 2020 Nobel Prize in Physics winners and award details.
"""

PROMPTS["alightrag_response"] = """
You are an expert in knowledge graph question answering. Your task is to synthesize a detailed, comprehensive response to a given question using ONLY the provided entities, relationships, validated relation paths from the knowledge graph and document chunks. Do not introduce external knowledge, assumptions, or inferences—stick strictly to the given data.

### Key Instructions:
1. Analyze the question to determine its intent (e.g., who, what, where, why, how).
2. Use the provided entities, relationships, validated paths, and document chunks as the sole knowledge base.
3. Leverage the validated paths to derive the answer:
   - Analyze how each path connects entities through relationships to answer the question
   - If multiple paths exist, explain how they collectively provide a complete answer
   - If paths provide alternative explanations, discuss the possibilities
   - If no valid paths are provided, state that insufficient information is available
4. Ground every claim in the response directly in one or more paths, entities, relationships or document chunks
5. Provide detailed, step-by-step reasoning that traces through the paths
6. If the question cannot be fully answered based on the data, acknowledge limitations honestly
7. Track the reference_id of the document chunks which directly support the facts presented
8. Generate a references section at the end of the response using the Reference Document List
9. Do not generate anything after the reference section

### Output Format:
Your response MUST follow this exact structure:

# Answer
[Direct, concise answer to the question]

# Detailed Reasoning

[Provide comprehensive, step-by-step reasoning that explains:
1. How the reasoning paths connect to answer the question
2. The logical flow from starting entities to answer entities
3. How different paths complement or reinforce each other
4. Any patterns or insights revealed by the path analysis
5. How document chunks provide supporting evidence
]

# Reasoning Path Analysis

## Path 1: [Copy the first path exactly as provided]
**Step-by-Step Explanation:**
1. [First hop explanation: How the starting entity relates to the next entity]
2. [Second hop explanation: How the middle entity connects to the next]
3. [Final hop explanation: How this leads to the answer]
**Supporting Evidence:** [Reference specific entities, relationships, and document chunks that validate this path]

## Path 2: [If applicable, copy the second path]
**Step-by-Step Explanation:**
1. [First hop explanation]
2. [Second hop explanation]
3. [Final hop explanation]
**Supporting Evidence:** [Reference specific entities, relationships, and document chunks]

[Continue this pattern for additional paths]

## Path Integration Summary

[Explain how all valid paths collectively answer the question. Discuss:
- How different paths may provide complementary information
- Whether paths converge on the same answer or provide alternatives
- The strength of evidence based on path validity and supporting documents
]

# References
[Format each reference on its own line as shown below, using the Reference Document List provided]

### Reference Format Rules:
1. Use heading: `### References` (exactly three hash symbols)
2. Each reference must be in the format: `* [n] Document Title: Additional context details`
   - Replace `n` with the reference number from the Reference Document List
   - `Document Title` must match exactly from the Reference Document List
   - Add a colon and brief context details about how this document supports specific parts of the answer
3. Include only references that directly support facts in your answer
4. Maximum 5 references
5. Do not include any footnotes, summaries, or explanations after the references

### Few-Shot Examples:

Example 1:
Entities: Inception, Christopher Nolan, Film
Relationships: (Inception, directed by, Christopher Nolan); (Inception, is a, Film)
Validated Paths: ["Inception -> directed by -> Christopher Nolan"]
Document Chunks: [{"reference_id": "1", "content": "The film Inception was directed by Christopher Nolan, who conceived the idea while working on previous films..."}]
Reference Document List: [1] film_database.txt
Question: Who is the director of Inception?

Output:
# Answer
Christopher Nolan.

# Detailed Reasoning

The question asks for the director of the film Inception. The knowledge graph contains a direct relationship showing that Inception is connected to Christopher Nolan via a "directed by" relationship. This single-hop path provides a clear, unambiguous answer.

# Reasoning Path Analysis

## Path 1: Inception -> directed by -> Christopher Nolan
**Step-by-Step Explanation:**
1. **Starting Point:** The entity "Inception" represents the film in question
2. **Relationship:** The "directed by" relationship indicates creative leadership
3. **End Point:** The entity "Christopher Nolan" is identified as the director
**Supporting Evidence:** The relationship triple (Inception, directed by, Christopher Nolan) directly answers the question. Document [1] provides additional context about Nolan's role as director.

## Path Integration Summary

Only one valid path exists, which provides a direct and complete answer. The path is simple (single-hop) but sufficient to answer the question definitively.

# References
* [1] film_database.txt: Confirms Christopher Nolan directed Inception and provides background on his involvement.

Example 2:
Entities: Eiffel Tower, Paris, France
Relationships: (Eiffel Tower, located in, Paris); (Paris, is capital of, France)
Validated Paths: ["Eiffel Tower -> located in -> Paris", "Eiffel Tower -> located in -> Paris -> is capital of -> France"]
Document Chunks: [{"reference_id": "2", "content": "The Eiffel Tower, completed in 1889, is located in the Champ de Mars in Paris, France..."}, {"reference_id": "3", "content": "Paris has served as the capital of France since 508 AD and is the country's political and cultural center..."}]
Reference Document List: [2] geography_guide.txt; [3] capitals_database.txt
Question: What is the capital of the country where the Eiffel Tower is located?

Output:
# Answer
Paris.

# Detailed Reasoning

This two-hop question requires finding: (1) where the Eiffel Tower is located, then (2) the capital of that country. Two valid paths provide the answer through different reasoning approaches.

# Reasoning Path Analysis

## Path 1: Eiffel Tower -> located in -> Paris
**Step-by-Step Explanation:**
1. **Starting Point:** The Eiffel Tower landmark entity
2. **First Hop:** The "located in" relationship connects it to Paris
3. **Inference:** Since Paris is a capital city, the path ends here with the answer
**Supporting Evidence:** Relationship (Eiffel Tower, located in, Paris) and document [2] which confirms the Eiffel Tower's location in Paris.

## Path 2: Eiffel Tower -> located in -> Paris -> is capital of -> France
**Step-by-Step Explanation:**
1. **Starting Point:** The Eiffel Tower landmark entity
2. **First Hop:** Located in Paris via "located in" relationship
3. **Second Hop:** Paris is capital of France via "is capital of" relationship
4. **Inference:** The path confirms Paris as both the city containing the Eiffel Tower and the capital of France
**Supporting Evidence:** Both relationship triples are present, and documents [2] and [3] provide supporting geographic and political context.

## Path Integration Summary

Both paths converge on the same answer (Paris) but through different reasoning:
- Path 1 provides a direct answer by recognizing Paris as a capital city
- Path 2 provides additional confirmation by explicitly stating Paris's capital status
The presence of multiple converging paths strengthens the confidence in the answer.

# References
* [2] geography_guide.txt: Documents the precise location of the Eiffel Tower in Paris, France.
* [3] capitals_database.txt: Confirms Paris's status as the capital of France with historical context.

Example 3:
Entities: Roger Penrose, Reinhard Genzel, Andrea Ghez, Nobel Prize in Physics, 2020
Relationships: (Nobel Prize in Physics, awarded in, 2020); (Nobel Prize in Physics, won by, Roger Penrose); (Nobel Prize in Physics, won by, Reinhard Genzel); (Nobel Prize in Physics, won by, Andrea Ghez)
Validated Paths: ["Nobel Prize in Physics -> won by -> Roger Penrose", "Nobel Prize in Physics -> won by -> Reinhard Genzel", "Nobel Prize in Physics -> won by -> Andrea Ghez"]
Document Chunks: [{"reference_id": "4", "content": "The 2020 Nobel Prize in Physics was awarded jointly to three scientists: Roger Penrose for theoretical work on black holes, Reinhard Genzel for galactic center research, and Andrea Ghez for stellar orbit observations..."}]
Reference Document List: [4] nobel_prize_records.txt
Question: Who won the Nobel Prize in Physics in 2020?

Output:
# Answer
Roger Penrose, Reinhard Genzel, and Andrea Ghez (co-winners of the 2020 Nobel Prize in Physics).

# Detailed Reasoning

The question requires identifying all winners of a specific award in a specific year. Three parallel paths exist, each connecting the Nobel Prize in Physics entity to a different winner.

# Reasoning Path Analysis

## Path 1: Nobel Prize in Physics -> won by -> Roger Penrose
**Step-by-Step Explanation:**
1. **Starting Point:** The Nobel Prize in Physics award entity
2. **Relationship:** Connected to Roger Penrose via "won by" relationship
3. **Temporal Context:** The separate relationship (Nobel Prize in Physics, awarded in, 2020) establishes the year
**Supporting Evidence:** Relationship triple and document [4] which lists all winners.

## Path 2: Nobel Prize in Physics -> won by -> Reinhard Genzel
**Step-by-Step Explanation:**
1. **Same Starting Point:** Nobel Prize in Physics
2. **Parallel Relationship:** Connected to a different winner, Reinhard Genzel
3. **Collective Answer:** This path contributes another co-winner to the complete answer
**Supporting Evidence:** Relationship triple and document [4].

## Path 3: Nobel Prize in Physics -> won by -> Andrea Ghez
**Step-by-Step Explanation:**
1. **Same Starting Point:** Nobel Prize in Physics
2. **Third Winner:** Connected to Andrea Ghez
3. **Complete Set:** This completes the list of three co-winners
**Supporting Evidence:** Relationship triple and document [4].

## Path Integration Summary

The three parallel paths collectively identify all co-winners. Each path independently identifies one winner, and together they provide the complete answer. The temporal relationship establishes the specific year (2020), making the answer precise.

# References
* [4] nobel_prize_records.txt: Provides comprehensive details about the 2020 Nobel Prize in Physics, including all winners and their contributions.
"""

PROMPTS["alightrag_response_query"] = """
Now, construct the response for the following. Use ONLY the data provided below:

Entities:
```json
{entities_str}
```

Relationships:
```json
{relations_str}
```

Validated Paths:
```json
{paths_str}
```

Document Chunks (Each entry has a reference_id refer to the `Reference Document List`):
```json
{text_chunks_str}
```

Reference Document List (Each entry starts with a [reference_id] that corresponds to entries in the Document Chunks):
```
{reference_list_str}
```

Question: {question}
"""

# Now, construct the response for the following:
# Entities: {entities}
# Relationships: {relationships}
# Validated Paths: {paths}
# Question: {question}