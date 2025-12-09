import asyncio
import json
import os
import time

from dotenv import load_dotenv

from alightrag import QueryParam
from alightrag.api.lightrag_server import configure_logging
from alightrag.utils import logger
from alightrag_test import initialize_rag, splitline
from dataloader import DatasetLoader, create_dataset_loader


load_dotenv(dotenv_path=".env", override=True)

time_str = time.strftime("%m%d%H%M", time.localtime())
WORKING_DIR = os.getenv("WORKING_DIR", "./experiments/")
WORKING_DIR = WORKING_DIR + time_str
if not os.path.exists(WORKING_DIR):
    os.makedirs(WORKING_DIR, exist_ok=True)


# noinspection PyUnboundLocalVariable
async def eval_alight(data: DatasetLoader):
    contexts = data.contexts
    queries = data.queries
    answers = data.answers

    result_file = os.path.join(WORKING_DIR, "eval_results.jsonl")
    with open(result_file, 'w', encoding='utf-8') as f:
        pass  # Create or clear the file

    try:
        rag = await initialize_rag()

        context_queries = zip(contexts, queries)
        for i, (context, query) in enumerate(context_queries, 1):
            logger.info(f"\n[{i}/{data.num_samples}] Query: \n{query}\n")
            await rag.ainsert(context)

            # --- Mode: hybrid ---
            logger.info(splitline("Query mode: hybrid"))
            start = time.time()
            ans_hybrid = await rag.aquery(query, param=QueryParam(mode="hybrid"))
            time_hybrid = round(time.time() - start, 3)

            # --- Mode: reasoning ---
            logger.info(splitline("Query mode: reasoning"))
            start = time.time()
            ans_reason = await rag.aquery(query, param=QueryParam(mode="alightrag"),
                                          use_reasoning=True, use_reflection=False)
            time_reason = round(time.time() - start, 3)

            # --- Mode: reflection ---
            logger.info(splitline("Query mode: reflection"))
            start = time.time()
            ans_reflect = await rag.aquery(query, param=QueryParam(mode="alightrag"),
                                           use_reasoning=False, use_reflection=True)
            time_reflect = round(time.time() - start, 3)

            # --- Mode: alightrag (full) ---
            logger.info(splitline("Query mode: alightrag"))
            start = time.time()
            ans_alight = await rag.aquery(query, param=QueryParam(mode="alightrag"))
            time_alight = round(time.time() - start, 3)

            report = (
                f"\nQuery: \n{query}"
                f"\nGround Truth Answer: \n{answers[i-1]}"
                f"\n{'-'*40}\n[ LightRAG ] Time: {time_hybrid}s, Answer: \n{ans_hybrid}"
                f"\n{'-'*40}\n[ LightRAG + Reasoning ] Time: {time_reason}s, Answer: \n{ans_reason}"
                f"\n{'-'*40}\n[ LightRAG + Reflection ] Time: {time_reflect}s, Answer: \n{ans_reflect}"
                f"\n{'-'*40}\n[ AlightRAG ] Time: {time_alight}s, Answer: \n{ans_alight}"
                f"\n{'-'*40}\n"
            )
            logger.info(report)
            print(report)

            result_entry = {
                "id": i,
                "question": query,
                "context": context,
                "ground_truth": answers[i - 1],
                "lightrag_answer": ans_hybrid,
                "lightrag_reasoning_answer": ans_reason,
                "lightrag_reflection_answer": ans_reflect,
                "alightrag_answer": ans_alight,
                "time_lightrag": time_hybrid,
                "time_lightrag_reasoning": time_reason,
                "time_lightrag_reflection": time_reflect,
                "time_alightrag": time_alight
            }
            with open(result_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result_entry, ensure_ascii=False) + "\n")
                logger.info(f"Result for sample {i} saved to {result_file}")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        if 'rag' in locals() and rag:
            await rag.finalize_storages()


if __name__ == "__main__":
    configure_logging(WORKING_DIR)
    dataset_root_path = "datasets"
    dataset_name = "2wikimultihop"
    dataset_file = "dev.json"
    dataset_path = os.path.join(dataset_root_path, dataset_name, dataset_file)
    dataloader = create_dataset_loader(dataset_path, num_samples=10)
    asyncio.run(eval_alight(dataloader))
