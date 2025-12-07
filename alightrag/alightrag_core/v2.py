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


#A1
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

        # Phase 2: Reasoning
        logger.debug("[AlightRAG] Reasoning started")
        try:
            reasoning_prompt = PROMPTS["alightrag_reasoning"].format(
                question=user_query,
                entities=", ".join(search_result["final_entities"]) if search_result["final_entities"] else "",
                relationships="; ".join([f"({s}, {r}, {t})" for s, r, t in search_result["final_relations"]])
                if search_result["final_relations"] else ""
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
                entities=", ".join(search_result["final_entities"]) if search_result["final_entities"] else "",
                relationships="; ".join([f"({s}, {r}, {t})" for s, r, t in search_result["final_relations"]])
                if search_result["final_relations"] else "",
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
                         f"{sum(1 for p in validation_result.get('validated_paths', []) if p.get('is_valid'))} valid paths")

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"[AlightRAG] Reflection failed: {e}")
            validation_result = {
                "validated_paths": [],
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

        # Parse relationship triples
        search_result["final_relations"] = []
        if filtered_relations:
            for triple_str in filtered_relations.split(";"):
                triple_str = triple_str.strip()
                if triple_str.startswith("(") and triple_str.endswith(")"):
                    triple_str = triple_str[1:-1]  # Remove parentheses
                    parts = [p.strip() for p in triple_str.split(",")]
                    if len(parts) == 3:
                        search_result["final_relations"].append(tuple(parts))

        search_result["final_paths"] = validation_result.get("validated_paths", [])

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
    truncation_result = await _apply_token_truncation(
        search_result,
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
    context, raw_data = await _build_context_str(
        entities_context=truncation_result.get("entities_context"),
        relations_context=truncation_result.get("relations_context"),
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
        "mode": "alightrag"
    }

    logger.debug(
        f"[AlightRAG] Final context length: {len(context) if context else 0}, "
        f"iterations: {current_iteration}"
    )

    return QueryContextResult(context=context, raw_data=raw_data)