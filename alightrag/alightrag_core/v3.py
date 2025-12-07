from __future__ import annotations

import json
from typing import Callable

from alightrag.base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    QueryParam,
    QueryContextResult,
)
from alightrag.prompt import PROMPTS
from alightrag.utils import (
    logger,
)


#A2
async def _alightrag_build_query_context(
        query: str,
        ll_keywords: str,
        hl_keywords: str,
        knowledge_graph_inst: BaseGraphStorage,
        entities_vdb: BaseVectorStorage,
        relationships_vdb: BaseVectorStorage,
        text_chunks_db: BaseKVStorage,
        query_param: QueryParam,
        chunks_vdb: BaseVectorStorage = None,
        use_model_func: Callable[..., object] = None,
) -> QueryContextResult | None:
    """
    Main query context building function using AlightRAG architecture:
    1. Search -> 2. Truncate -> 3. Merge chunks -> 4. Build LLM context
    1.1. Retrieval
    1.2. Reasoning
    1.3. Reflection

    Returns unified QueryContextResult containing both context and raw_data.
    """
    if not query:
        logger.warning("Query is empty, skipping context building")
        return None

    # Helper function to format entities and relationships for prompts
    def format_for_prompts(entities_list, relations_list):
        """Convert structured entities/relations to prompt-compatible strings."""
        # Format entities: extract entity_name from dictionaries
        entity_names = []
        for entity in entities_list:
            # Entity has: entity_name, entity_type, description, source_id, file_path, timestamp
            name = entity.get("entity_name")
            if name:
                entity_names.append(name)

        entities_str = ", ".join(entity_names) if entity_names else ""

        # Format relationships
        relation_triples = []
        for rel in relations_list:
            # Relationship has: src_id, tgt_id, weight, description, keywords, source_id, file_path, timestamp
            src_name = rel.get("src_id")
            tgt_name = rel.get("tgt_id")
            keywords = rel.get("keywords", "")
            description = rel.get("description", "")

            # Use description as relationship text if available, otherwise use keywords
            # or a generic term
            # rel_text = description[:50] if description else (keywords if keywords else "related to")
            # rel_text = description if description else (keywords if keywords else "related to")
            max_rel_length = 50
            if description:
                rel_text = description[:max_rel_length] + "..." if len(description) > max_rel_length else description
            elif keywords:
                rel_text = keywords[:max_rel_length] + "..." if len(keywords) > max_rel_length else keywords
            else:
                rel_text = "related to"

            if src_name and tgt_name:
                relation_triples.append(f"({src_name}, {rel_text}, {tgt_name})")

        relations_str = "; ".join(relation_triples) if relation_triples else ""

        return entities_str, relations_str

    def reconstruct_search_result(filtered_entities, filtered_relations, original_search_result):
        """
        Reconstruct the search result structure for downstream processing.

        Args:
            filtered_entities: List of entity names from reflection filtering
            filtered_relations: List of relation tuples (src_name, rel_text, tgt_name) from reflection
            original_search_result: Original search result containing full entity/relation objects

        Returns:
            Dictionary with reconstructed search result structure
        """
        formatted_result = {
            "final_entities": [],
            "final_relations": [],
            "vector_chunks": original_search_result.get("vector_chunks", []),
            "chunk_tracking": original_search_result.get("chunk_tracking", {}),
            "query_embedding": original_search_result.get("query_embedding")
        }

        # Build mapping of original entities by name for efficient lookup
        original_entities_map = {}
        for entity in original_search_result.get("original_entities", []):
            name = entity.get("entity_name")
            if name:
                original_entities_map[name] = entity

        # Reconstruct entities, preserving original data when available
        for entity_name in filtered_entities:
            if entity_name in original_entities_map:
                # Use the original entity object with all fields
                formatted_result["final_entities"].append(original_entities_map[entity_name])
            else:
                # Create minimal entity structure for filtered entities
                formatted_result["final_entities"].append({
                    "entity_name": entity_name,
                    "entity_type": "",  # Unknown after filtering
                    "description": "",  # Unknown after filtering
                    "source_id": "",  # Unknown after filtering
                    "file_path": "",  # Unknown after filtering
                    "timestamp": ""  # Unknown after filtering
                })

        # Build mapping of original relations by (src, tgt) tuple
        original_relations_map = {}
        for rel in original_search_result.get("original_relations", []):
            src = rel.get("src_id")
            tgt = rel.get("tgt_id")
            if src and tgt:
                original_relations_map[(src, tgt)] = rel

        # Reconstruct relations, preserving original data when available
        for rel_tuple in filtered_relations:
            if len(rel_tuple) == 3:
                src_name, rel_text, tgt_name = rel_tuple
                rel_key = (src_name, tgt_name)

                if rel_key in original_relations_map:
                    # Use the original relation object with all fields
                    formatted_result["final_relations"].append(original_relations_map[rel_key])
                else:
                    # Create minimal relation structure for filtered relations
                    formatted_result["final_relations"].append({
                        "src_id": src_name,
                        "tgt_id": tgt_name,
                        "weight": 1.0,  # Default weight
                        "description": rel_text,
                        "keywords": rel_text,
                        "source_id": "",  # Unknown after filtering
                        "file_path": "",  # Unknown after filtering
                        "timestamp": ""  # Unknown after filtering
                    })

        return formatted_result


    # Stage 1: Pure search
    user_query = query
    max_iterations = 3  # Default
    current_iteration = 0

    # Initialize with empty search result
    search_result = {
        "final_entities": [],
        "final_relations": [],
        "vector_chunks": [],
        "chunk_tracking": {},
        "query_embedding": None,
        "final_paths": []
    }

    retrieval_query = query
    while current_iteration < max_iterations:
        current_iteration += 1
        logger.debug(f"[AlightRAG] Starting iteration {current_iteration}/{max_iterations}")

        # Phase 1: Retrieval
        logger.debug("[AlightRAG] Retrieval started")
        search_result = await _perform_kg_search(
            retrieval_query,
            ll_keywords,
            hl_keywords,
            knowledge_graph_inst,
            entities_vdb,
            relationships_vdb,
            text_chunks_db,
            query_param,
            chunks_vdb,
        )

        logger.debug(f"[AlightRAG] Retrieval completed: {len(search_result['final_entities'])} entities, "
                     f"{len(search_result['final_relations'])} relations")

        # Store original data before filtering
        search_result["original_entities"] = search_result["final_entities"][:] if search_result[
            "final_entities"] else []
        search_result["original_relations"] = search_result["final_relations"][:] if search_result[
            "final_relations"] else []

        # Format entities and relationships for prompts
        entities_str, relations_str = format_for_prompts(
            search_result["final_entities"],
            search_result["final_relations"]
        )

        # Log the formatted strings for debugging
        logger.debug(f"[AlightRAG] Formatted entities: {entities_str[:100]}...")
        logger.debug(f"[AlightRAG] Formatted relations: {relations_str[:100]}...")

        # Phase 2: Reasoning
        logger.debug("[AlightRAG] Reasoning started")
        try:
            reasoning_prompt = PROMPTS["alightrag_reasoning"].format(
                question=user_query,
                entities=entities_str,
                relationships=relations_str
            )

            reasoning_response = await use_model_func(
                user_query,
                system_prompt=reasoning_prompt,
                history_messages=query_param.conversation_history,
                enable_cot=True,
                stream=query_param.stream,
            )

            # Parse JSON response
            if isinstance(reasoning_response, str):
                path_result = json.loads(reasoning_response)
            else:
                path_result = reasoning_response

            logger.debug(f"[AlightRAG] Reasoning completed: {len(path_result.get('paths', []))} paths found")

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"[AlightRAG] Reasoning failed: {e}")
            path_result = {"paths": [], "explanation": "Reasoning failed"}

        # Phase 3: Reflection
        logger.debug("[AlightRAG] Reflection started")
        try:
            reflection_prompt = PROMPTS["alightrag_reflection"].format(
                question=user_query,
                entities=entities_str,
                relationships=relations_str,
                paths=json.dumps(path_result.get("paths", []))
            )

            reflection_response = await use_model_func(
                user_query,
                system_prompt=reflection_prompt,
                history_messages=query_param.conversation_history,
                enable_cot=True,
                stream=query_param.stream,
            )

            # Parse JSON response
            if isinstance(reflection_response, str):
                validation_result = json.loads(reflection_response)
            else:
                validation_result = reflection_response

            logger.debug(f"[AlightRAG] Reflection completed: "
                         f"{sum(1 for p in validation_result.get('validations', []) if p.get('is_valid'))} valid paths")

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"[AlightRAG] Reflection failed: {e}")
            validation_result = {
                "validations": [],
                "filtered_entities": "",
                "filtered_relationships": "",
                "overall_explanation": "Reflection failed"
            }

        # Update search result with filtered data
        filtered_entities = validation_result.get("filtered_entities", "")
        filtered_relations = validation_result.get("filtered_relationships", "")

        # Convert comma-separated strings back to lists
        search_result["final_entities"] = [e.strip() for e in filtered_entities.split(",")
                                           if e.strip()] if filtered_entities else []

        # Parse relationship triples - these are now in the format (src_name, rel_text, tgt_name)
        search_result["final_relations"] = []
        if filtered_relations:
            for triple_str in filtered_relations.split(";"):
                triple_str = triple_str.strip()
                if triple_str.startswith("(") and triple_str.endswith(")"):
                    triple_str = triple_str[1:-1]  # Remove parentheses
                    parts = [p.strip() for p in triple_str.split(",")]
                    if len(parts) == 3:
                        search_result["final_relations"].append(tuple(parts))

        search_result["final_paths"] = validation_result.get("validations", [])

        # Check if we should continue iterating
        supplementary_questions = validation_result.get("supplementary_questions", [])
        is_sufficient = not supplementary_questions

        if is_sufficient:
            logger.debug("[AlightRAG] Paths are sufficient, stopping iteration")
            break
        elif supplementary_questions and current_iteration < max_iterations:
            # Use first supplementary question for next iteration
            retrieval_query = supplementary_questions[0]
            logger.debug(f"[AlightRAG] Continuing with supplementary question: {retrieval_query}")
        else:
            logger.debug("[AlightRAG] Maximum iterations reached or no supplementary questions")
            break

    # After iteration completes (or breaks early), proceed with remaining stages
    # Check if we have any data to process
    if not search_result["final_entities"] and not search_result["final_relations"]:
        if query_param.mode != "mix":
            return None
        elif not search_result["chunk_tracking"]:
            return None

    # Stage 2: Apply token truncation for LLM efficiency

    # Reconstruct for downstream processing
    formatted_search_result = reconstruct_search_result(
        filtered_entities=search_result["final_entities"],
        filtered_relations=search_result["final_relations"],
        original_search_result=search_result,
    )

    truncation_result = await _apply_token_truncation(
        formatted_search_result,
        query_param,
        text_chunks_db.global_config,
    )


    # Stage 3: Merge chunks using filtered entities/relations
    merged_chunks = await _merge_all_chunks(
        filtered_entities=truncation_result.get("filtered_entities", []),
        filtered_relations=truncation_result.get("filtered_relations", []),
        vector_chunks=search_result.get("vector_chunks", []),
        query=query,  # Use original query for merging
        knowledge_graph_inst=knowledge_graph_inst,
        text_chunks_db=text_chunks_db,
        query_param=query_param,
        chunks_vdb=chunks_vdb,
        chunk_tracking=search_result.get("chunk_tracking", {}),
        query_embedding=search_result.get("query_embedding"),
    )

    if (
            not merged_chunks
            and not truncation_result.get("entities_context")
            and not truncation_result.get("relations_context")
    ):
        return None

    # Stage 4: Build final LLM context
    context, raw_data = await _alightrag_build_context_str(
        entities_context=truncation_result.get("entities_context"),
        relations_context=truncation_result.get("relations_context"),
        # paths_context=truncation_result.get("paths_context", []),  # NEW: Add paths
        paths_context=search_result.get("final_paths", []),  # NEW: Add paths
        merged_chunks=merged_chunks,
        query=query,  # Use original query
        query_param=query_param,
        global_config=text_chunks_db.global_config,
        chunk_tracking=search_result.get("chunk_tracking", {}),
        entity_id_to_original=truncation_result.get("entity_id_to_original", {}),
        relation_id_to_original=truncation_result.get("relation_id_to_original", {}),
    )

    # Add metadata
    hl_keywords_list = hl_keywords.split(", ") if hl_keywords else []
    ll_keywords_list = ll_keywords.split(", ") if ll_keywords else []

    if "metadata" not in raw_data:
        raw_data["metadata"] = {}

    raw_data["metadata"]["keywords"] = {
        "high_level": hl_keywords_list,
        "low_level": ll_keywords_list,
    }
    raw_data["metadata"]["processing_info"] = {
        "total_iterations": current_iteration,
        "total_entities_found": len(search_result.get("final_entities", [])),
        "total_relations_found": len(search_result.get("final_relations", [])),
        "valid_paths_count": sum(1 for p in search_result.get("final_paths", [])
                                 if p.get("is_valid")),
        "entities_after_truncation": len(truncation_result.get("filtered_entities", [])),
        "relations_after_truncation": len(truncation_result.get("filtered_relations", [])),
        "merged_chunks_count": len(merged_chunks),
        "final_chunks_count": len(raw_data.get("data", {}).get("chunks", [])),
    }

    logger.debug(
        f"[AlightRAG] Final context length: {len(context) if context else 0}, "
    )
    logger.debug(
        f"[AlightRAG] Raw data entities: {len(raw_data.get('data', {}).get('entities', []))}, relationships: {len(raw_data.get('data', {}).get('relationships', []))}, paths: {len(raw_data.get('data', {}).get('paths', []))}, chunks: {len(raw_data.get('data', {}).get('chunks', []))}"
    )
    logger.debug(
        f"[AlightRAG] Final iteration times: {current_iteration}"
    )

    return QueryContextResult(context=context, raw_data=raw_data)


async def _alightrag_build_context_str(
        entities_context: list[dict],
        relations_context: list[dict],
        paths_context: list[dict],  # NEW: Add paths_context parameter
        merged_chunks: list[dict],
        query: str,
        query_param: QueryParam,
        global_config: dict[str, str],
        chunk_tracking: dict = None,
        entity_id_to_original: dict = None,
        relation_id_to_original: dict = None,
) -> tuple[str, dict[str, Any]]:
    """
    Build the final LLM context string with token processing.
    This includes dynamic token calculation and final chunk truncation.
    """
    tokenizer = global_config.get("tokenizer")
    if not tokenizer:
        logger.error("Missing tokenizer, cannot build LLM context")
        # Return empty raw data structure when no tokenizer
        empty_raw_data = alightrag_convert_to_user_format(
            [],
            [],
            [],
            [],
            query_param.mode,
        )
        empty_raw_data["status"] = "failure"
        empty_raw_data["message"] = "Missing tokenizer, cannot build LLM context."
        return "", empty_raw_data

    # Get token limits
    max_total_tokens = getattr(
        query_param,
        "max_total_tokens",
        global_config.get("max_total_tokens", DEFAULT_MAX_TOTAL_TOKENS),
    )

    # Get the system prompt template from PROMPTS or global_config
    sys_prompt_template = global_config.get(
        "system_prompt_template", PROMPTS["rag_response"]
    )

    kg_context_template = PROMPTS["alightrag_kg_query_context"]
    user_prompt = query_param.user_prompt if query_param.user_prompt else ""
    response_type = (
        query_param.response_type
        if query_param.response_type
        else "Multiple Paragraphs"
    )

    entities_str = "\n".join(
        json.dumps(entity, ensure_ascii=False) for entity in entities_context
    )
    relations_str = "\n".join(
        json.dumps(relation, ensure_ascii=False) for relation in relations_context
    )

    # NEW: Format paths for context
    paths_str = "\n".join(
        json.dumps(path, ensure_ascii=False) for path in paths_context
    ) if paths_context else ""

    # Calculate preliminary kg context tokens
    pre_kg_context = kg_context_template.format(
        entities_str=entities_str,
        relations_str=relations_str,
        paths_str=paths_str,  # NEW: Include paths
        text_chunks_str="",
        reference_list_str="",
    )
    kg_context_tokens = len(tokenizer.encode(pre_kg_context))

    # Calculate preliminary system prompt tokens
    pre_sys_prompt = sys_prompt_template.format(
        context_data="",  # Empty for overhead calculation
        response_type=response_type,
        user_prompt=user_prompt,
    )
    sys_prompt_tokens = len(tokenizer.encode(pre_sys_prompt))

    # Calculate available tokens for text chunks
    query_tokens = len(tokenizer.encode(query))
    buffer_tokens = 200  # reserved for reference list and safety buffer
    available_chunk_tokens = max_total_tokens - (
            sys_prompt_tokens + kg_context_tokens + query_tokens + buffer_tokens
    )

    logger.debug(
        f"Token allocation - Total: {max_total_tokens}, SysPrompt: {sys_prompt_tokens}, Query: {query_tokens}, KG: {kg_context_tokens}, Buffer: {buffer_tokens}, Available for chunks: {available_chunk_tokens}"
    )

    # Apply token truncation to chunks using the dynamic limit
    truncated_chunks = await process_chunks_unified(
        query=query,
        unique_chunks=merged_chunks,
        query_param=query_param,
        global_config=global_config,
        source_type=query_param.mode,
        chunk_token_limit=available_chunk_tokens,  # Pass dynamic limit
    )

    # Generate reference list from truncated chunks using the new common function
    reference_list, truncated_chunks = generate_reference_list_from_chunks(
        truncated_chunks
    )

    # Rebuild chunks_context with truncated chunks
    chunks_context = []
    for i, chunk in enumerate(truncated_chunks):
        chunks_context.append(
            {
                "reference_id": chunk["reference_id"],
                "content": chunk["content"],
            }
        )

    text_units_str = "\n".join(
        json.dumps(text_unit, ensure_ascii=False) for text_unit in chunks_context
    )
    reference_list_str = "\n".join(
        f"[{ref['reference_id']}] {ref['file_path']}"
        for ref in reference_list
        if ref["reference_id"]
    )

    logger.info(
        f"Final context: {len(entities_context)} entities, {len(relations_context)} relations, {len(paths_context)} paths, {len(chunks_context)} chunks"
    )

    # not necessary to use LLM to generate a response
    if not entities_context and not relations_context and not chunks_context and not paths_context:
        # Return empty raw data structure when no entities/relations
        empty_raw_data = alightrag_convert_to_user_format(
            [],
            [],
            [],
            [],
            query_param.mode,
        )
        empty_raw_data["status"] = "failure"
        empty_raw_data["message"] = "Query returned empty dataset."
        return "", empty_raw_data

    # output chunks tracking infomations
    if truncated_chunks and chunk_tracking:
        chunk_tracking_log = []
        for chunk in truncated_chunks:
            chunk_id = chunk.get("chunk_id")
            if chunk_id and chunk_id in chunk_tracking:
                tracking_info = chunk_tracking[chunk_id]
                source = tracking_info["source"]
                frequency = tracking_info["frequency"]
                order = tracking_info["order"]
                chunk_tracking_log.append(f"{source}{frequency}/{order}")
            else:
                chunk_tracking_log.append("?0/0")

        if chunk_tracking_log:
            logger.info(f"Final chunks S+F/O: {' '.join(chunk_tracking_log)}")

    result = kg_context_template.format(
        entities_str=entities_str,
        relations_str=relations_str,
        paths_str=paths_str,  # NEW: Include paths in context
        text_chunks_str=text_units_str,
        reference_list_str=reference_list_str,
    )
    """
    Knowledge Graph Data (Entity):

    ```json
    {entities_str}
    ```

    Knowledge Graph Data (Relationship):

    ```json
    {relations_str}
    ```

    Knowledge Graph Data (Reasoning Path):

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

    """

    # Always return both context and complete data structure (unified approach)
    logger.debug(
        f"[_build_context_str] Converting to user format: {len(entities_context)} entities, {len(relations_context)} relations, {len(paths_context)} paths, {len(truncated_chunks)} chunks"
    )
    final_data = alightrag_convert_to_user_format(
        entities_context,
        relations_context,
        truncated_chunks,
        reference_list,
        query_param.mode,
        entity_id_to_original,
        relation_id_to_original,
        paths_context,  # NEW: Pass paths to conversion
    )

    logger.debug(
        f"[_build_context_str] Final data after conversion: {len(final_data.get('entities', []))} entities, {len(final_data.get('relationships', []))} relationships, {len(final_data.get('paths', []))} paths, {len(final_data.get('chunks', []))} chunks"
    )
    return result, final_data


def alightrag_convert_to_user_format(
        entities_context: list[dict],
        relations_context: list[dict],
        chunks: list[dict],
        references: list[dict],
        query_mode: str,
        entity_id_to_original: dict = None,
        relation_id_to_original: dict = None,
        paths_context: list[dict] = None,  # NEW: Add paths_context parameter
) -> dict[str, Any]:
    """Convert internal data format to user-friendly format using original database data"""

    # Convert entities format using original data when available
    formatted_entities = []
    for entity in entities_context:
        entity_name = entity.get("entity", "")

        # Try to get original data first
        original_entity = None
        if entity_id_to_original and entity_name in entity_id_to_original:
            original_entity = entity_id_to_original[entity_name]

        if original_entity:
            # Use original database data
            formatted_entities.append(
                {
                    "entity_name": original_entity.get("entity_name", entity_name),
                    "entity_type": original_entity.get("entity_type", "UNKNOWN"),
                    "description": original_entity.get("description", ""),
                    "source_id": original_entity.get("source_id", ""),
                    "file_path": original_entity.get("file_path", "unknown_source"),
                    "created_at": original_entity.get("created_at", ""),
                }
            )
        else:
            # Fallback to LLM context data (for backward compatibility)
            formatted_entities.append(
                {
                    "entity_name": entity_name,
                    "entity_type": entity.get("type", "UNKNOWN"),
                    "description": entity.get("description", ""),
                    "source_id": entity.get("source_id", ""),
                    "file_path": entity.get("file_path", "unknown_source"),
                    "created_at": entity.get("created_at", ""),
                }
            )

    # Convert relationships format using original data when available
    formatted_relationships = []
    for relation in relations_context:
        entity1 = relation.get("entity1", "")
        entity2 = relation.get("entity2", "")
        relation_key = (entity1, entity2)

        # Try to get original data first
        original_relation = None
        if relation_id_to_original and relation_key in relation_id_to_original:
            original_relation = relation_id_to_original[relation_key]

        if original_relation:
            # Use original database data
            formatted_relationships.append(
                {
                    "src_id": original_relation.get("src_id", entity1),
                    "tgt_id": original_relation.get("tgt_id", entity2),
                    "description": original_relation.get("description", ""),
                    "keywords": original_relation.get("keywords", ""),
                    "weight": original_relation.get("weight", 1.0),
                    "source_id": original_relation.get("source_id", ""),
                    "file_path": original_relation.get("file_path", "unknown_source"),
                    "created_at": original_relation.get("created_at", ""),
                }
            )
        else:
            # Fallback to LLM context data (for backward compatibility)
            formatted_relationships.append(
                {
                    "src_id": entity1,
                    "tgt_id": entity2,
                    "description": relation.get("description", ""),
                    "keywords": relation.get("keywords", ""),
                    "weight": relation.get("weight", 1.0),
                    "source_id": relation.get("source_id", ""),
                    "file_path": relation.get("file_path", "unknown_source"),
                    "created_at": relation.get("created_at", ""),
                }
            )

    # NEW: Convert paths format (paths are already validated by AlightRAG)
    formatted_paths = []
    if paths_context:
        for i, path_info in enumerate(paths_context):
            # Paths from AlightRAG have format: {"path": "...", "is_valid": True, "reason": "..."}
            # Or from paths_context: {"path": "...", "reason": "...", "order": i}
            path_str = path_info.get("path", "")
            reason = path_info.get("reason", "")
            is_valid = path_info.get("is_valid", True)  # Default to True for paths_context

            # Only include valid paths with non-empty path strings
            if path_str and is_valid:
                formatted_paths.append({
                    "path": path_str,
                    "reason": reason,
                    "is_valid": True,
                    "order": path_info.get("order", i + 1),
                })

    # Convert chunks format (chunks already contain complete data)
    formatted_chunks = []
    for i, chunk in enumerate(chunks):
        chunk_data = {
            "reference_id": chunk.get("reference_id", ""),
            "content": chunk.get("content", ""),
            "file_path": chunk.get("file_path", "unknown_source"),
            "chunk_id": chunk.get("chunk_id", ""),
        }
        formatted_chunks.append(chunk_data)

    logger.debug(
        f"[convert_to_user_format] Formatted {len(formatted_entities)} entities, "
        f"{len(formatted_relationships)} relationships, "
        f"{len(formatted_paths)} paths, "  # NEW: Log paths count
        f"{len(formatted_chunks)} chunks"
    )

    # Build basic metadata (metadata details will be added by calling functions)
    metadata = {
        "query_mode": query_mode,
        "keywords": {
            "high_level": [],
            "low_level": [],
        },  # Placeholder, will be set by calling functions
        "path_count": len(formatted_paths),  # NEW: Add path count to metadata
    }

    return {
        "status": "success",
        "message": "Query processed successfully",
        "data": {
            "entities": formatted_entities,
            "relationships": formatted_relationships,
            "paths": formatted_paths,  # NEW: Add paths to data
            "chunks": formatted_chunks,
            "references": references,
        },
        "metadata": metadata,
    }