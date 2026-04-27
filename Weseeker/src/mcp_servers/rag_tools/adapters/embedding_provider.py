from __future__ import annotations

from langchain_core.embeddings import Embeddings
from openai import OpenAI

from config.settings import RAGSettings


class LMStudioEmbeddings(Embeddings):
    def __init__(self, *, model: str, base_url: str) -> None:
        self._model = model
        self._client = OpenAI(base_url=base_url, api_key="lm-studio")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(model=self._model, input=texts)
        return [list(item.embedding) for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        response = self._client.embeddings.create(model=self._model, input=text)
        return list(response.data[0].embedding)


def create_embedding_model(settings: RAGSettings) -> Embeddings:
    provider = settings.embedding_provider.strip().lower()
    if provider != "lmstudio":
        raise ValueError(f"Step 1 仅支持 lmstudio embedding_provider，当前为: {provider}")
    if not settings.embedding_model:
        raise ValueError("未配置 rag.embedding_model，无法初始化 embedding 模型。")

    return LMStudioEmbeddings(
        model=settings.embedding_model,
        base_url=settings.lmstudio_embedding_base_url,
    )
