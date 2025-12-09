import asyncio
import os
import time

from dotenv import load_dotenv

from alightrag import AlightRAG, QueryParam
from alightrag.api.lightrag_server import configure_logging
from alightrag.kg.shared_storage import initialize_pipeline_status
from alightrag.llm.ds_silicon import siliconflow_embed, llm_model_func
from alightrag.utils import logger, EmbeddingFunc


load_dotenv(dotenv_path=".env", override=True)

time_str = time.strftime("%m%d%H%M", time.localtime())
WORKING_DIR = os.getenv("WORKING_DIR", "./alightrag_test/")
WORKING_DIR = WORKING_DIR + time_str
if not os.path.exists(WORKING_DIR):
    os.makedirs(WORKING_DIR, exist_ok=True)


def splitline(raw_str: str, line_length: int = 25) -> str:
    # "\n=====================\n<raw_str>\n=====================\n"
    return "\n" + "=" * line_length + "\n" + raw_str + "\n" + "=" * line_length + "\n"


async def initialize_rag():
    rag = AlightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=8192,
            func=lambda texts: siliconflow_embed(
                texts=texts,
                client_configs={"timeout": 30.0}),
        ),
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


# noinspection PyUnboundLocalVariable
async def main():
    try:
        rag = await initialize_rag()

        with open("./book.txt", "r", encoding="utf-8") as f:
            await rag.ainsert(f.read())

        query = (
            "What specific ecological project was ultimately funded as a direct, "
            "albeit indirect, consequence of the Kaito Foundation's subsidiary "
            "violating a 2060 conservation act during the procurement of "
            "materials for Centennial City's primary power source?"
        )

        # Perform hybrid search
        logger.info(splitline("Query mode: hybrid"))
        print(await rag.aquery(query, param=QueryParam(mode="hybrid"),))

        # Perform lightrag+reasoning search
        logger.info(splitline("Query mode: reasoning"))
        print(await rag.aquery(query, param=QueryParam(mode="alightrag"),
                               use_reasoning=True, use_reflection=False))

        # Perform lightrag+reflection search
        logger.info(splitline("Query mode: reflection"))
        print(await rag.aquery(query, param=QueryParam(mode="alightrag"),
                               use_reasoning=False, use_reflection=True))

        # Perform alightrag search
        logger.info(splitline("Query mode: alightrag"))
        print(await rag.aquery(query, param=QueryParam(mode="alightrag")))
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        if rag:
            await rag.finalize_storages()


if __name__ == "__main__":
    configure_logging(WORKING_DIR)
    asyncio.run(main())
    logger.info("\nDone!")
