import asyncio
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

    results = []

    try:
        rag = await initialize_rag()

        ans_hybrid_list = []
        ans_reason_list = []
        ans_reflect_list = []
        ans_alight_list = []

        context_queries = zip(contexts, queries)
        for i, (context, query) in enumerate(context_queries, 1):
            logger.info(f"\n[{i}/{data.num_samples}] Query: \n{query}\n")
            await rag.ainsert(context)

            logger.info(splitline("Query mode: hybrid"))
            ans_hybrid = await rag.aquery(query, param=QueryParam(mode="hybrid"))
            ans_hybrid_list.append(ans_hybrid)

            logger.info(splitline("Query mode: reasoning"))
            ans_reason = await rag.aquery(query, param=QueryParam(mode="alightrag"),
                                          use_reasoning=True, use_reflection=False)
            ans_reason_list.append(ans_reason)

            logger.info(splitline("Query mode: reflection"))
            ans_reflect = await rag.aquery(query, param=QueryParam(mode="alightrag"),
                                           use_reasoning=False, use_reflection=True)
            ans_reflect_list.append(ans_reflect)

            logger.info(splitline("Query mode: alightrag"))
            ans_alight = await rag.aquery(query, param=QueryParam(mode="alightrag"))
            ans_alight_list.append(ans_alight)

            report = (
                f"\nQuery: \n{query}"
                f"\nGround Truth Answer: \n{answers[i-1]}"
                f"\n{'-'*40}\n[ LightRAG ] Answer: \n{ans_hybrid}"
                f"\n{'-'*40}\n[ LightRAG + Reasoning ] Answer: \n{ans_reason}"
                f"\n{'-'*40}\n[ LightRAG + Reflection ] Answer: \n{ans_reflect}"
                f"\n{'-'*40}\n[ AlightRAG ] Answer: \n{ans_alight}"
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
                "alightrag_answer": ans_alight
            }
            results.append(result_entry)

    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        try:
            # Save results to a file
            result_file = os.path.join(WORKING_DIR, "eval_results.jsonl")
            with open(result_file, 'w', encoding='utf-8') as f:
                for entry in results:
                    f.write(f"{entry}\n")
            logger.info(f"Results saved to {result_file}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
        if rag:
            await rag.finalize_storages()


if __name__ == "__main__":
    configure_logging(WORKING_DIR)
    # dataset_path = "datasets/hotpot_dev_distractor_v1.json"
    # dataset_path = "datasets/graphquestions.testing.json"
    dataset_path = "datasets/agriculture.jsonl"
    num_samples = 10
    dataloader = create_dataset_loader(dataset_path, num_samples=num_samples)
    asyncio.run(eval_alight(dataloader))
