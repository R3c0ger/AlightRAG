import base64
import os
from typing import Any, List
from typing import Dict

import httpx
import numpy as np

from alightrag.llm.openai import openai_complete_if_cache
from alightrag.utils import logger


async def llm_model_func(
        prompt,
        system_prompt=None,
        history_messages=None,
        **kwargs
) -> str:
    return await openai_complete_if_cache(
        os.getenv("LLM_MODEL", "deepseek-chat"),
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("LLM_BINDING_API_KEY"),
        base_url=os.getenv("LLM_BINDING_HOST", "https://api.deepseek.com/v1"),
        **kwargs,
    )


def create_httpx_async_client(
        api_key: str | None = None,
        client_configs: Dict[str, Any] | None = None,
) -> httpx.AsyncClient:
    """Create an httpx AsyncClient with provided configurations."""
    api_key = api_key or os.getenv("EMBEDDING_BINDING_API_KEY")
    if not api_key:
        raise ValueError("SiliconFlow API key required. Set SILICONFLOW_API_KEY or pass api_key=")

    configs = {
        "timeout": 60.0,
        "headers": {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    }
    if client_configs:
        configs.update(client_configs)

    return httpx.AsyncClient(**configs)


async def siliconflow_embed(
        texts: List[str],
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        embedding_dim: int | None = None,
        client_configs: Dict[str, Any] | None = None,
        token_tracker: Any | None = None,
) -> np.ndarray:

    model = model or os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
    base_url = base_url or os.getenv("SILICONFLOW_API_BASE", "https://api.siliconflow.cn/v1")

    async_client = create_httpx_async_client(
        api_key=api_key,
        client_configs=client_configs,
    )

    async with async_client:
        api_params = {
            "model": model,
            "input": texts,
        }

        if embedding_dim is not None:
            # noinspection PyTypeChecker
            api_params["dimensions"] = embedding_dim

        # 发起 API 调用
        try:
            response = await async_client.post(
                f"{base_url.rstrip('/')}/embeddings",
                json=api_params,
                headers={"Authorization": f"Bearer {api_key or os.getenv('EMBEDDING_BINDING_API_KEY')}"},
            )
            response.raise_for_status()  # 抛出 HTTP 错误
            data = response.json()

            # 处理 token 跟踪（如果提供）
            if token_tracker and "usage" in data:
                token_counts = {
                    "prompt_tokens": data.get("usage", {}).get("prompt_tokens", 0),
                    "total_tokens": data.get("usage", {}).get("total_tokens", 0),
                }
                token_tracker.add_usage(token_counts)

            embeddings = [
                np.array(dp["embedding"], dtype=np.float32)
                if isinstance(dp["embedding"], list)
                else np.frombuffer(base64.b64decode(dp["embedding"]), dtype=np.float32)
                for dp in data["data"]
            ]

            return np.array(embeddings)

        except httpx.HTTPStatusError as e:
            logger.error(f"SiliconFlow API HTTP error: {e}")
            raise
        except httpx.ConnectError as e:
            logger.error(f"SiliconFlow API connection error: {e}")
            raise
        except httpx.TimeoutException as e:
            logger.error(f"SiliconFlow API timeout: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in SiliconFlow embed: {e}")
            raise
