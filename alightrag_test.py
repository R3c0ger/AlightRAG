import os
import asyncio
import logging
import logging.config
from alightrag import AlightRAG, QueryParam
# from alightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from alightrag.llm.azure_openai import azure_openai_complete, azure_openai_embed
from alightrag.kg.shared_storage import initialize_pipeline_status
from alightrag.utils import logger, set_verbose_debug
WORKING_DIR = "./alightrag_test"


def configure_logging():
    """Configure logging for the application"""

    # Reset any existing handlers to ensure clean configuration
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error", "alightrag"]:
        logger_instance = logging.getLogger(logger_name)
        logger_instance.handlers = []
        logger_instance.filters = []

    # Get log directory path from environment variable or use current directory
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(os.path.join(log_dir, "alightrag_demo.log"))

    print(f"\nAlightRAG demo log file: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    # Get log file max size and backup count from environment variables
    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))  # Default 10MB
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))  # Default 5 backups

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(levelname)s: %(message)s",
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_file_path,
                    "maxBytes": log_max_bytes,
                    "backupCount": log_backup_count,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "alightrag": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }
    )

    # Set the logger level to INFO
    logger.setLevel(logging.INFO)
    # Enable verbose debug if needed
    set_verbose_debug(os.getenv("VERBOSE_DEBUG", "false").lower() == "true")


if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def initialize_rag():
    rag = AlightRAG(
        working_dir=WORKING_DIR,
        embedding_func=azure_openai_embed,
        llm_model_func=azure_openai_complete,
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


async def main():
    # Check if OPENAI_API_KEY environment variable exists
    # if not os.getenv("OPENAI_API_KEY"):
    #     print(
    #         "Error: OPENAI_API_KEY environment variable is not set. Please set this variable before running the program."
    #     )
    #     print("You can set the environment variable by running:")
    #     print("  export OPENAI_API_KEY='your-openai-api-key'")
    #     return  # Exit the async function

    try:
        # Clear old data files
        files_to_delete = [
            "graph_chunk_entity_relation.graphml",
            "kv_store_doc_status.json",
            "kv_store_entity_chunks.json",
            "kv_store_full_docs.json",
            "kv_store_full_entities.json",
            "kv_store_full_relations.json",
            "kv_store_llm_response_cache.json",
            "kv_store_relation_chunks.json",
            "kv_store_text_chunks.json",
            "vdb_chunks.json",
            "vdb_entities.json",
            "vdb_relationships.json",
        ]

        for file in files_to_delete:
            file_path = os.path.join(WORKING_DIR, file)
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Deleting old file:: {file_path}")

        # Initialize RAG instance
        rag = await initialize_rag()

        # Test embedding function
        test_text = ["This is a test string for embedding."]
        embedding = await rag.embedding_func(test_text)
        embedding_dim = embedding.shape[1]
        print("\n=======================")
        print("Test embedding function")
        print("========================")
        print(f"Test dict: {test_text}")
        print(f"Detected embedding dimension: {embedding_dim}\n\n")

        with open("./book.txt", "r", encoding="utf-8") as f:
            await rag.ainsert(f.read())

        query="What specific ecological project was ultimately funded as a direct, albeit indirect, consequence of the Kaito Foundation's subsidiary violating a 2060 conservation act during the procurement of materials for Centennial City's primary power source?"

        # # Perform naive search
        # print("\n=====================")
        # print("Query mode: naive")
        # print("=====================")
        # print(
        #     await rag.aquery(
        #         query, param=QueryParam(mode="naive")
        #     )
        # )
        #
        # # Perform local search
        # print("\n=====================")
        # print("Query mode: local")
        # print("=====================")
        # print(
        #     await rag.aquery(
        #         query, param=QueryParam(mode="local")
        #     )
        # )
        #
        # # Perform global search
        # print("\n=====================")
        # print("Query mode: global")
        # print("=====================")
        # print(
        #     await rag.aquery(
        #         query,
        #         param=QueryParam(mode="global"),
        #     )
        # )
        #
        # # Perform hybrid search
        # print("\n=====================")
        # print("Query mode: hybrid")
        # print("=====================")
        # print(
        #     await rag.aquery(
        #         "What specific ecological project was ultimately funded as a direct, albeit indirect, consequence of the Kaito Foundation's subsidiary violating a 2060 conservation act during the procurement of materials for Centennial City's primary power source?",
        #         param=QueryParam(mode="hybrid"),
        #     )
        # )
        #
        # # alightrag-insert TODO
        # # Perform alightrag search
        # print("\n=====================")
        # print("Query mode: alightrag")
        # print("=====================")
        # print(
        #     await rag.aquery(
        #         query,
        #         param=QueryParam(mode="alightrag"),
        #         # below params are default to be true
        #         # use_reasoning=True,
        #         # use_reflection=True,
        #     )
        # )
        #
        # # alightrag-insert TODO
        # # Perform lightrag+reasoning search
        # print("\n=====================")
        # print("Query mode: reasoning")
        # print("=====================")
        # print(
        #     await rag.aquery(
        #         query,
        #         param=QueryParam(mode="alightrag"),
        #         use_reasoning=True,
        #         use_reflection=False,
        #     )
        # )

        # alightrag-insert TODO
        # Perform lightrag+reflection search
        print("\n=====================")
        print("Query mode: reflection")
        print("=====================")
        print(
            await rag.aquery(
                query,
                param=QueryParam(mode="alightrag"),
                use_reasoning=False,
                use_reflection=True,
            )
        )

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if rag:
            await rag.finalize_storages()


if __name__ == "__main__":
    # Configure logging before running the main function
    configure_logging()
    asyncio.run(main())
    print("\nDone!")
