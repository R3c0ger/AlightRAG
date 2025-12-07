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

# H
# Now let's update the old _build_query_context to use the new architecture
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
    Main query context building function using the new 4-stage architecture:
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
    # alightrag-insert
    # retrieval call
    logger.debug(
        f"[_alightrag_build_query_context] retrieval started"
    )
    search_result = await _perform_kg_search(
        query,
        ll_keywords,
        hl_keywords,
        knowledge_graph_inst,
        entities_vdb,
        relationships_vdb,
        text_chunks_db,
        query_param,
        chunks_vdb,
    )
    '''
    return {
        "final_entities": final_entities,
        "final_relations": final_relations,
        "vector_chunks": vector_chunks,
        "chunk_tracking": chunk_tracking,
        "query_embedding": query_embedding,
    }
    '''
    logger.debug(f"final_result: {search_result}")
    logger.debug(
        f"[_alightrag_build_query_context] retrieval completed"
    )

    if not search_result["final_entities"] and not search_result["final_relations"]:
        if query_param.mode != "mix":
            return None
        else:
            if not search_result["chunk_tracking"]:
                return None

    # TODO
    user_query = query
    # --------------------------------------------------------------#
    # alightrag iter k
    # it=k
    # while query_param.mode == "alightrag":
    #     # reasoning call
    #     # reflection call
    #     it += 1
    #     if reflection.condition or it >= k:
    #         break
    #     else:
    #         # retrieval call
    # --------------------------------------------------------------#
    # alightrag iter 1
    # reasoning call
    logger.debug(
        f"[_alightrag_build_query_context] reasoning started"
    )
    reasoning_prompt_temp = PROMPTS["alightrag_reasoning"],
    reasoning_prompt = reasoning_prompt_temp.format(
        entities=search_result["final_entities"],
        relationships=search_result["final_relations"],
    )
    path_result = await use_model_func(
        user_query,
        system_prompt=reasoning_prompt,
        history_messages=query_param.conversation_history,
        enable_cot=True,
        stream=query_param.stream,
    )
    '''
    {
      "paths": [
        "Entity1",
        "Entity2 -> rel1 -> Entity3",
        "Entity4 -> rel2 -> Entity5 -> rel3 -> Entity6"
      ],
      "explanation": "Brief explanation of how the paths answer the question."
    }
    '''
    logger.debug(f"path_result: {path_result}")
    logger.debug(
        f"[_alightrag_build_query_context] reasoning completed"
    )
    # reflection call
    logger.debug(
        f"[_alightrag_build_query_context] reflection started"
    )
    reflection_prompt_temp = PROMPTS["alightrag_reflection"]
    reflection_prompt = reflection_prompt_temp.format(
        entities=search_result["final_entities"],
        relationships=search_result["final_relations"],
        paths=path_result["paths"],
    )
    validation_result = await use_model_func(
        user_query,
        system_prompt=reflection_prompt,
        history_messages=query_param.conversation_history,
        enable_cot=True,
        stream=query_param.stream,
    )
    '''
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
    '''
    logger.debug(f"validation_result: {validation_result}")
    logger.debug(
        f"[_alightrag_build_query_context] reflection completed"
    )
    search_result["final_entities"]=validation_result["filtered_entities"]
    search_result["final_relations"]=validation_result["filtered_relationships"]
    search_result["final_paths"]=validation_result["validated_paths"]
    # --------------------------------------------------------------#

    # Stage 2: Apply token truncation for LLM efficiency
    truncation_result = await _apply_token_truncation(
        search_result,
        query_param,
        text_chunks_db.global_config,
    )

    # Stage 3: Merge chunks using filtered entities/relations
    merged_chunks = await _merge_all_chunks(
        filtered_entities=truncation_result["filtered_entities"],
        filtered_relations=truncation_result["filtered_relations"],
        vector_chunks=search_result["vector_chunks"],
        query=query,
        knowledge_graph_inst=knowledge_graph_inst,
        text_chunks_db=text_chunks_db,
        query_param=query_param,
        chunks_vdb=chunks_vdb,
        chunk_tracking=search_result["chunk_tracking"],
        query_embedding=search_result["query_embedding"],
    )

    if (
        not merged_chunks
        and not truncation_result["entities_context"]
        and not truncation_result["relations_context"]
    ):
        return None

    # Stage 4: Build final LLM context with dynamic token processing
    # _build_context_str now always returns tuple[str, dict]
    context, raw_data = await _build_context_str(
        entities_context=truncation_result["entities_context"],
        relations_context=truncation_result["relations_context"],
        merged_chunks=merged_chunks,
        query=query,
        query_param=query_param,
        global_config=text_chunks_db.global_config,
        chunk_tracking=search_result["chunk_tracking"],
        entity_id_to_original=truncation_result["entity_id_to_original"],
        relation_id_to_original=truncation_result["relation_id_to_original"],
    )

    # Convert keywords strings to lists and add complete metadata to raw_data
    hl_keywords_list = hl_keywords.split(", ") if hl_keywords else []
    ll_keywords_list = ll_keywords.split(", ") if ll_keywords else []

    # Add complete metadata to raw_data (preserve existing metadata including query_mode)
    if "metadata" not in raw_data:
        raw_data["metadata"] = {}

    # Update keywords while preserving existing metadata
    raw_data["metadata"]["keywords"] = {
        "high_level": hl_keywords_list,
        "low_level": ll_keywords_list,
    }
    raw_data["metadata"]["processing_info"] = {
        "total_entities_found": len(search_result.get("final_entities", [])),
        "total_relations_found": len(search_result.get("final_relations", [])),
        "valid_paths_count": sum(1 for p in search_result.get("final_paths", [])
                                 if p.get("is_valid")),
        "entities_after_truncation": len(
            truncation_result.get("filtered_entities", [])
        ),
        "relations_after_truncation": len(
            truncation_result.get("filtered_relations", [])
        ),
        "merged_chunks_count": len(merged_chunks),
        "final_chunks_count": len(raw_data.get("data", {}).get("chunks", [])),
    }

    logger.debug(
        f"[_alightrag_build_query_context] Context length: {len(context) if context else 0}"
    )
    logger.debug(
        f"[_alightrag_build_query_context] Raw data entities: {len(raw_data.get('data', {}).get('entities', []))}, "
        f"relationships: {len(raw_data.get('data', {}).get('relationships', []))}, "
        f"chunks: {len(raw_data.get('data', {}).get('chunks', []))}"
        f"iterations: {current_iteration}"
    )

    return QueryContextResult(context=context, raw_data=raw_data)