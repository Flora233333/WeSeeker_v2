from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

from langchain_core.documents import Document
from openai import OpenAI

from config.settings import RAGSettings, _read_dotenv_value
from mcp_servers.rag_tools.indexing.llm_enhancement_cache import (
    append_cache_record,
    build_cache_key,
)
from mcp_servers.rag_tools.indexing.scanner import FileRecord

_FORBIDDEN_KEYS = {"text", "content", "chunk_text", "parent_text", "child_text"}
_TOP_LEVEL_KEYS = {"parents"}
_PARENT_KEYS = {"parent_no", "title", "summary", "keywords", "children"}
_CHILD_KEYS = {
    "child_no",
    "topic",
    "summary",
    "keywords",
    "entities",
    "aliases",
    "likely_queries",
    "chunk_type",
}
_CHUNK_TYPES = {
    "concept",
    "procedure",
    "comparison",
    "pitfall",
    "code",
    "table",
    "plan",
    "reflection",
    "definition",
    "example",
    "other",
}

_PARENT_TITLE_MAX_CHARS = 40
_PARENT_SUMMARY_MAX_CHARS = 120
_CHILD_TOPIC_MAX_CHARS = 40
_CHILD_SUMMARY_MAX_CHARS = 80
_MAX_PARENT_KEYWORDS = 8
_MAX_CHILD_KEYWORDS = 8
_MAX_CHILD_ENTITIES = 8
_MAX_CHILD_ALIASES = 5
_MAX_CHILD_LIKELY_QUERIES = 5
_MAX_LLM_BATCH_PARENTS = 8
_MAX_LLM_BATCH_CHILDREN = 24
_MAX_LLM_BATCH_CHARS = 30_000

_SYSTEM_PROMPT = """你是一个 RAG 检索增强器。

你的任务是阅读用户提供的“已有 Parent / Child 分块”，为每个 Parent 和 Child 生成检索增强 metadata。
你不是分块器。你不能修改、合并、删除、重排任何 Parent 或 Child。

你可以在内部思考，但最终 content 必须只输出合法 json object。
不要输出 markdown。
不要输出解释。
不要输出代码块。
不要输出自然语言前后缀。

硬性规则：
- 不要改写原文。
- 不要输出 chunk 正文。
- 不要输出 text、content、chunk_text、parent_text、child_text 字段。
- 不要改变 parent_no。
- 不要改变 child_no。
- 不要新增 Parent。
- 不要删除 Parent。
- 不要新增 Child。
- 不要删除 Child。
- 返回的 Parent 数量必须与输入一致。
- 每个 Parent 下返回的 Child 数量必须与输入一致。
- 所有 summary 必须忠实于对应 chunk 原文，不得添加原文没有的信息。
- keywords 必须来自原文术语、原文主题或明确常见同义表达。
- entities 只能包含原文中出现的人名、组织名、模型名、技术名、项目名、文件名、缩写或专有名词。
- aliases 只能包含明确等价、缩写、英文名、中文名或常见翻译。
- likely_queries 必须是用户可能用来查找该 child 的自然语言问题。
- 每个 likely_query 必须能由对应 child 原文直接回答。
- 不要生成过泛的问题，例如“这个文档讲了什么”。
- 不要加入原文无关的热门关键词。
- 如果某个字段无法忠实生成，宁可返回空数组或短字符串，不要编造。
- chunk_type 必须从以下枚举中选择：concept、procedure、comparison、pitfall、code、table、plan、
  reflection、definition、example、other。

长度和数量约束：
- Parent title 目标不超过 24 个字符，绝不能超过 40 个字符。
- Parent summary 目标不超过 90 个中文字符，绝不能超过 120 个字符。
- Parent keywords 最多 8 个。
- Child topic 目标不超过 24 个字符，绝不能超过 40 个字符。
- Child summary 目标不超过 60 个中文字符，绝不能超过 80 个字符。
- Child keywords 最多 8 个。
- Child entities 最多 8 个。
- Child aliases 最多 5 个。
- Child likely_queries 最多 5 个。
- 长度限制按最终 json 字符串的字符数计算。
- title 和 topic 必须是短名词短语，不要写完整句子。
- summary 只写该 chunk 的核心语义关系，不要罗列多个并列技术词。
- 多个技术词、工具名、模型名必须放到 keywords 或 entities，不要塞进 summary。
- 摘要宁可短，不要为了完整覆盖所有细节而超过长度限制。
- 不要为了凑数量而编造内容。

输出 json 格式必须严格符合用户给出的示例。"""


@dataclass(frozen=True)
class LLMEnhancementResult:
    parent_documents: list[Document]
    child_documents: list[Document]
    stats: dict[str, object]


@dataclass(frozen=True)
class _LLMEnhancementBatch:
    parent_offset: int
    parent_documents: list[Document]
    child_documents: list[Document]


def build_initial_llm_enhancement_stats(settings: RAGSettings) -> dict[str, object]:
    config = settings.llm_enhancement
    return {
        "enabled": config.enabled,
        "model": config.model,
        "prompt_version": config.prompt_version,
        "api_calls": 0,
        "cache_hits": 0,
        "cache_writes": 0,
        "enhanced_files": 0,
    }


def merge_llm_enhancement_stats(
    base: dict[str, object],
    update: dict[str, object],
) -> dict[str, object]:
    merged = dict(base)
    for key in ("api_calls", "cache_hits", "cache_writes", "enhanced_files"):
        merged[key] = int(merged.get(key) or 0) + int(update.get(key) or 0)
    return merged


def enhance_file_chunks_with_llm(
    file_record: FileRecord,
    parent_documents: list[Document],
    child_documents: list[Document],
    *,
    settings: RAGSettings,
    kb_name: str,
    cache_records: dict[str, dict[str, object]] | None = None,
    client: OpenAI | None = None,
) -> LLMEnhancementResult:
    config = settings.llm_enhancement
    stats = build_initial_llm_enhancement_stats(settings)
    if not config.enabled:
        return LLMEnhancementResult(parent_documents, child_documents, stats)
    if config.provider.strip().lower() != "deepseek":
        raise ValueError(f"LLM enhancement 仅支持 deepseek provider，当前为: {config.provider}")
    if config.mode != "strict":
        raise ValueError(f"LLM enhancement 当前仅支持 strict mode，当前为: {config.mode}")
    if not parent_documents and not child_documents:
        return LLMEnhancementResult(parent_documents, child_documents, stats)

    cache_key = build_cache_key(
        file_record,
        parent_documents,
        child_documents,
        model=config.model,
        prompt_version=config.prompt_version,
    )
    payload: dict[str, object]
    cache_hit = False
    if config.cache_enabled and cache_records is not None and cache_key in cache_records:
        record = cache_records[cache_key]
        cached_payload = record.get("payload")
        if not isinstance(cached_payload, dict):
            raise ValueError(f"LLM enhancement cache payload 无效: {file_record.path_str}")
        payload = cached_payload
        cache_hit = True
        stats["cache_hits"] = 1
        enhanced_parents, enhanced_children = _apply_payload_with_file_context(
            payload,
            parent_documents,
            child_documents,
            file_record=file_record,
            model=config.model,
            prompt_version=config.prompt_version,
            input_hash=cache_key,
        )
    else:
        payload_parents: list[object] = []
        enhanced_parents = []
        enhanced_children = []
        # 大文件整文件请求会让 DeepSeek 生成超长 JSON；按 Parent 批处理但仍写文件级 cache。
        for batch in _iter_llm_enhancement_batches(parent_documents, child_documents):
            batch_payload, batch_parents, batch_children, api_calls = _enhance_uncached_batch(
                file_record,
                batch.parent_documents,
                batch.child_documents,
                settings=settings,
                client=client,
                model=config.model,
                prompt_version=config.prompt_version,
                input_hash=cache_key,
            )
            stats["api_calls"] = int(stats["api_calls"]) + api_calls
            enhanced_parents.extend(batch_parents)
            enhanced_children.extend(batch_children)
            payload_parents.extend(
                _renumber_payload_parents(batch_payload, parent_offset=batch.parent_offset)
            )
        payload = {"parents": payload_parents}
    stats["enhanced_files"] = 1

    if config.cache_enabled and not cache_hit:
        record = {
            "cache_key": cache_key,
            "file_path": file_record.path_str,
            "model": config.model,
            "prompt_version": config.prompt_version,
            "payload": payload,
        }
        append_cache_record(config.cache_dir, kb_name, record)
        if cache_records is not None:
            cache_records[cache_key] = record
        stats["cache_writes"] = 1

    return LLMEnhancementResult(enhanced_parents, enhanced_children, stats)


def _enhance_uncached_batch(
    file_record: FileRecord,
    parent_documents: list[Document],
    child_documents: list[Document],
    *,
    settings: RAGSettings,
    client: OpenAI | None,
    model: str,
    prompt_version: str,
    input_hash: str,
) -> tuple[dict[str, object], list[Document], list[Document], int]:
    config = settings.llm_enhancement
    chunk_dump = _build_chunk_dump_for_llm(parent_documents, child_documents)
    if len(chunk_dump) > config.max_file_chars:
        raise ValueError(
            "LLM enhancement batch 输入超过 max_file_chars，"
            f"file={file_record.path_str}, chars={len(chunk_dump)}, limit={config.max_file_chars}"
        )

    messages = _build_llm_enhancement_messages(file_record, parent_documents, child_documents)
    total_api_calls = 0
    for validation_attempt in range(config.max_retries + 1):
        payload, api_calls = _call_deepseek_enhancement(messages, settings, client=client)
        total_api_calls += api_calls
        try:
            enhanced_parents, enhanced_children = _apply_payload_with_file_context(
                payload,
                parent_documents,
                child_documents,
                file_record=file_record,
                model=model,
                prompt_version=prompt_version,
                input_hash=input_hash,
            )
        except ValueError as exc:
            if validation_attempt >= config.max_retries:
                raise
            messages = _build_llm_enhancement_retry_messages(
                file_record,
                parent_documents,
                child_documents,
                validation_error=exc,
            )
            continue
        return payload, enhanced_parents, enhanced_children, total_api_calls

    raise RuntimeError(f"LLM enhancement batch 未返回结果: {file_record.path_str}")


def _apply_payload_with_file_context(
    payload: dict[str, object],
    parent_documents: list[Document],
    child_documents: list[Document],
    *,
    file_record: FileRecord,
    model: str,
    prompt_version: str,
    input_hash: str,
) -> tuple[list[Document], list[Document]]:
    try:
        return _apply_enhancement_payload(
            payload,
            parent_documents,
            child_documents,
            model=model,
            prompt_version=prompt_version,
            input_hash=input_hash,
        )
    except ValueError as exc:
        raise ValueError(
            f"LLM enhancement payload 校验失败，file={file_record.path_str}: {exc}"
        ) from exc


def _build_llm_enhancement_messages(
    file_record: FileRecord,
    parent_documents: list[Document],
    child_documents: list[Document],
) -> list[dict[str, str]]:
    file_name = file_record.path.name
    doc_type = _first_doc_type(parent_documents, child_documents)
    chunk_dump = _build_chunk_dump_for_llm(parent_documents, child_documents)
    user_prompt = f"""请为以下已分块文档生成 RAG 检索增强 metadata。

必须输出 json object，格式如下：

{{
  "parents": [
    {{
      "parent_no": 1,
      "title": "简短父块标题",
      "summary": "父块摘要",
      "keywords": ["关键词"],
      "children": [
        {{
          "child_no": 1,
          "topic": "子块主题",
          "summary": "子块摘要",
          "keywords": ["关键词"],
          "entities": ["实体"],
          "aliases": ["同义表达"],
          "likely_queries": ["用户可能查询"],
          "chunk_type": "concept"
        }}
      ]
    }}
  ]
}}

文件元信息：
- file_name: {file_name}
- doc_type: {doc_type}
- parent_count: {len(parent_documents)}
- child_count: {len(child_documents)}

已分块内容如下：

{chunk_dump}
"""
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _build_llm_enhancement_retry_messages(
    file_record: FileRecord,
    parent_documents: list[Document],
    child_documents: list[Document],
    *,
    validation_error: ValueError,
) -> list[dict[str, str]]:
    messages = _build_llm_enhancement_messages(
        file_record,
        parent_documents,
        child_documents,
    )
    messages.append(
        {
            "role": "user",
            "content": (
                "上一次输出未通过本地校验，必须重新输出完整 json object。"
                f"校验错误：{validation_error}\n"
                "只修正格式、长度和字段，不要改变 parent_no、child_no、Parent/Child 数量。"
                "title/topic 必须更短；summary 不要罗列技术词，技术词放 keywords。"
            ),
        }
    )
    return messages


def _call_deepseek_enhancement(
    messages: list[dict[str, str]],
    settings: RAGSettings,
    *,
    client: OpenAI | None = None,
) -> tuple[dict[str, object], int]:
    config = settings.llm_enhancement
    api_key = os.environ.get(config.api_key_env) or _read_dotenv_value(config.api_key_env)
    if not api_key:
        raise ValueError(f"未找到 {config.api_key_env}，无法调用 DeepSeek LLM enhancement。")

    effective_client = client or OpenAI(api_key=api_key, base_url=config.base_url)
    last_error: Exception | None = None
    for attempt in range(config.max_retries + 1):
        try:
            kwargs = {
                "model": config.model,
                "messages": messages,
                "response_format": {"type": "json_object"},
                "stream": False,
                "extra_body": {
                    "thinking": {"type": "enabled" if config.thinking_enabled else "disabled"}
                },
            }
            if config.thinking_enabled:
                kwargs["reasoning_effort"] = config.reasoning_effort
            response = effective_client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            return _parse_llm_response(content), attempt + 1
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= config.max_retries:
                break
            time.sleep(0.5 * (2**attempt))

    raise RuntimeError(f"DeepSeek LLM enhancement 调用失败: {last_error}") from last_error


def _parse_llm_response(content: str | None) -> dict[str, object]:
    if not content or not content.strip():
        raise ValueError("DeepSeek LLM enhancement 返回空 content。")
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("DeepSeek LLM enhancement 返回内容不是合法 JSON。") from exc
    if not isinstance(payload, dict):
        raise ValueError("DeepSeek LLM enhancement 顶层结果必须是 JSON object。")
    return payload


def _validate_enhancement_payload(
    payload: dict[str, object],
    parent_documents: list[Document],
    child_documents: list[Document],
) -> None:
    _reject_forbidden_keys(payload)
    _ensure_exact_keys(payload, _TOP_LEVEL_KEYS, "top-level")
    parents_payload = payload.get("parents")
    if not isinstance(parents_payload, list):
        raise ValueError("LLM enhancement payload.parents 必须是 list。")
    if len(parents_payload) != len(parent_documents):
        raise ValueError(
            "LLM enhancement parent 数量不匹配，"
            f"expected={len(parent_documents)}, actual={len(parents_payload)}"
        )

    children_by_parent = _group_children_by_parent(parent_documents, child_documents)
    for parent_index, (parent_payload, parent_document) in enumerate(
        zip(parents_payload, parent_documents, strict=True),
        start=1,
    ):
        if not isinstance(parent_payload, dict):
            raise ValueError(f"LLM enhancement parent #{parent_index} 必须是 object。")
        _ensure_exact_keys(parent_payload, _PARENT_KEYS, f"parent #{parent_index}")
        if parent_payload["parent_no"] != parent_index:
            raise ValueError(f"LLM enhancement parent_no 不匹配: parent #{parent_index}")
        _ensure_text(
            parent_payload["title"],
            _PARENT_TITLE_MAX_CHARS,
            f"parent #{parent_index} title",
        )
        _ensure_text(
            parent_payload["summary"],
            _PARENT_SUMMARY_MAX_CHARS,
            f"parent #{parent_index} summary",
        )
        _ensure_text_list(parent_payload["keywords"], _MAX_PARENT_KEYWORDS, "parent keywords")

        expected_children = children_by_parent[str(parent_document.metadata.get("parent_id") or "")]
        children_payload = parent_payload["children"]
        if not isinstance(children_payload, list):
            raise ValueError(f"LLM enhancement parent #{parent_index}.children 必须是 list。")
        if len(children_payload) != len(expected_children):
            raise ValueError(
                "LLM enhancement child 数量不匹配，"
                f"parent={parent_index}, expected={len(expected_children)}, "
                f"actual={len(children_payload)}"
            )
        for child_index, child_payload in enumerate(children_payload, start=1):
            if not isinstance(child_payload, dict):
                raise ValueError(
                    f"LLM enhancement parent #{parent_index} child #{child_index} 必须是 object。"
                )
            _validate_child_payload(parent_index, child_index, child_payload)


def _apply_enhancement_payload(
    payload: dict[str, object],
    parent_documents: list[Document],
    child_documents: list[Document],
    *,
    model: str,
    prompt_version: str,
    input_hash: str,
) -> tuple[list[Document], list[Document]]:
    _validate_enhancement_payload(payload, parent_documents, child_documents)
    children_by_parent = _group_children_by_parent(parent_documents, child_documents)
    parents_payload = payload["parents"]

    enhanced_parents: list[Document] = []
    enhanced_children: list[Document] = []
    for parent_payload, parent_document in zip(parents_payload, parent_documents, strict=True):
        parent_payload = _as_dict(parent_payload, "parent payload")
        parent_metadata = dict(parent_document.metadata)
        parent_metadata.update(
            {
                "llm_parent_title": str(parent_payload["title"]),
                "llm_parent_summary": str(parent_payload["summary"]),
                "llm_parent_keywords_text": _join_terms(parent_payload["keywords"]),
                "llm_enhancement_model": model,
                "llm_enhancement_prompt_version": prompt_version,
                "llm_enhancement_input_hash": input_hash,
            }
        )
        enhanced_parents.append(
            Document(page_content=parent_document.page_content, metadata=parent_metadata)
        )

        parent_id = str(parent_document.metadata.get("parent_id") or "")
        for child_payload, child_document in zip(
            _as_list(parent_payload["children"], "children"),
            children_by_parent[parent_id],
            strict=True,
        ):
            child_payload = _as_dict(child_payload, "child payload")
            child_metadata = dict(child_document.metadata)
            child_metadata.update(
                {
                    "llm_child_topic": str(child_payload["topic"]),
                    "llm_child_summary": str(child_payload["summary"]),
                    "llm_keywords_text": _join_terms(child_payload["keywords"]),
                    "llm_entities_text": _join_terms(child_payload["entities"]),
                    "llm_aliases_text": _join_terms(child_payload["aliases"]),
                    "llm_likely_queries_text": _join_queries(child_payload["likely_queries"]),
                    "llm_chunk_type": str(child_payload["chunk_type"]),
                    "llm_enhancement_model": model,
                    "llm_enhancement_prompt_version": prompt_version,
                    "llm_enhancement_input_hash": input_hash,
                }
            )
            enhanced_children.append(
                Document(page_content=child_document.page_content, metadata=child_metadata)
            )

    return enhanced_parents, enhanced_children


def _build_llm_enhanced_embedding_text(document: Document) -> str:
    metadata = document.metadata
    prefix_parts = []
    if metadata.get("llm_child_topic"):
        prefix_parts.append(f"主题：{metadata['llm_child_topic']}")
    if metadata.get("llm_child_summary"):
        prefix_parts.append(f"摘要：{metadata['llm_child_summary']}")
    if metadata.get("llm_keywords_text"):
        prefix_parts.append(f"关键词：{metadata['llm_keywords_text']}")
    if metadata.get("llm_entities_text"):
        prefix_parts.append(f"实体：{metadata['llm_entities_text']}")
    if metadata.get("llm_aliases_text"):
        prefix_parts.append(f"别名：{metadata['llm_aliases_text']}")
    if metadata.get("llm_likely_queries_text"):
        prefix_parts.append(f"可能问题：\n{metadata['llm_likely_queries_text']}")
    if not prefix_parts:
        return document.page_content
    return "\n".join(prefix_parts) + f"\n\n原文：\n{document.page_content}"


def _build_chunk_dump_for_llm(
    parent_documents: list[Document],
    child_documents: list[Document],
) -> str:
    children_by_parent = _group_children_by_parent(parent_documents, child_documents)
    lines: list[str] = []
    for parent_index, parent in enumerate(parent_documents, start=1):
        parent_id = str(parent.metadata.get("parent_id") or "")
        lines.extend(
            [
                f"Parent {parent_index}",
                f"heading_path: {parent.metadata.get('heading_path') or ''}",
                f"parent_chars: {len(parent.page_content)}",
                "",
            ]
        )
        for child_index, child in enumerate(children_by_parent[parent_id], start=1):
            lines.extend(
                [
                    f"Child {child_index}",
                    f"child_id: {child.metadata.get('child_id') or ''}",
                    f"chars: {len(child.page_content)}",
                    "text:",
                    child.page_content.strip(),
                    "",
                ]
            )
    return "\n".join(lines).strip()


def _iter_llm_enhancement_batches(
    parent_documents: list[Document],
    child_documents: list[Document],
) -> list[_LLMEnhancementBatch]:
    children_by_parent = _group_children_by_parent(parent_documents, child_documents)
    batches: list[_LLMEnhancementBatch] = []
    current_parents: list[Document] = []
    current_children: list[Document] = []
    current_offset = 0

    for parent_index, parent in enumerate(parent_documents):
        parent_id = str(parent.metadata.get("parent_id") or "")
        parent_children = children_by_parent[parent_id]
        candidate_parents = [*current_parents, parent]
        candidate_children = [*current_children, *parent_children]
        candidate_chars = len(_build_chunk_dump_for_llm(candidate_parents, candidate_children))
        should_split = (
            current_parents
            and (
                len(candidate_parents) > _MAX_LLM_BATCH_PARENTS
                or len(candidate_children) > _MAX_LLM_BATCH_CHILDREN
                or candidate_chars > _MAX_LLM_BATCH_CHARS
            )
        )
        if should_split:
            batches.append(
                _LLMEnhancementBatch(
                    parent_offset=current_offset,
                    parent_documents=current_parents,
                    child_documents=current_children,
                )
            )
            current_offset = parent_index
            current_parents = [parent]
            current_children = list(parent_children)
            continue
        current_parents = candidate_parents
        current_children = candidate_children

    if current_parents:
        batches.append(
            _LLMEnhancementBatch(
                parent_offset=current_offset,
                parent_documents=current_parents,
                child_documents=current_children,
            )
        )
    return batches


def _renumber_payload_parents(
    payload: dict[str, object],
    *,
    parent_offset: int,
) -> list[object]:
    parents = payload.get("parents")
    if not isinstance(parents, list):
        raise ValueError("LLM enhancement payload.parents 必须是 list。")
    renumbered: list[object] = []
    for parent in parents:
        if not isinstance(parent, dict):
            raise ValueError("LLM enhancement parent payload 必须是 object。")
        copied = dict(parent)
        parent_no = copied.get("parent_no")
        if not isinstance(parent_no, int):
            raise ValueError("LLM enhancement parent_no 必须是 int。")
        copied["parent_no"] = parent_offset + parent_no
        renumbered.append(copied)
    return renumbered


def _group_children_by_parent(
    parent_documents: list[Document],
    child_documents: list[Document],
) -> dict[str, list[Document]]:
    parent_ids = [str(document.metadata.get("parent_id") or "") for document in parent_documents]
    grouped = {parent_id: [] for parent_id in parent_ids}
    for child in child_documents:
        parent_id = str(child.metadata.get("parent_id") or "")
        if parent_id not in grouped:
            raise ValueError(f"Child 引用了不存在的 parent_id: {parent_id}")
        grouped[parent_id].append(child)
    return grouped


def _validate_child_payload(
    parent_index: int,
    child_index: int,
    child_payload: dict[str, object],
) -> None:
    _ensure_exact_keys(child_payload, _CHILD_KEYS, f"parent #{parent_index} child #{child_index}")
    if child_payload["child_no"] != child_index:
        raise ValueError(
            f"LLM enhancement child_no 不匹配: parent #{parent_index} child #{child_index}"
        )
    child_label = f"parent #{parent_index} child #{child_index}"
    _ensure_text(child_payload["topic"], _CHILD_TOPIC_MAX_CHARS, f"{child_label} topic")
    _ensure_text(child_payload["summary"], _CHILD_SUMMARY_MAX_CHARS, f"{child_label} summary")
    _ensure_text_list(child_payload["keywords"], _MAX_CHILD_KEYWORDS, "child keywords")
    _ensure_text_list(child_payload["entities"], _MAX_CHILD_ENTITIES, "child entities")
    _ensure_text_list(child_payload["aliases"], _MAX_CHILD_ALIASES, "child aliases")
    _ensure_text_list(
        child_payload["likely_queries"],
        _MAX_CHILD_LIKELY_QUERIES,
        "child likely_queries",
    )
    chunk_type = child_payload["chunk_type"]
    if not isinstance(chunk_type, str) or chunk_type not in _CHUNK_TYPES:
        raise ValueError(f"LLM enhancement chunk_type 无效: {chunk_type}")


def _ensure_exact_keys(
    payload: dict[str, object],
    expected: set[str],
    label: str,
) -> None:
    actual = set(payload)
    unknown = actual - expected
    missing = expected - actual
    if unknown:
        raise ValueError(f"LLM enhancement {label} 存在未知字段: {sorted(unknown)}")
    if missing:
        raise ValueError(f"LLM enhancement {label} 缺少字段: {sorted(missing)}")


def _reject_forbidden_keys(value: object) -> None:
    if isinstance(value, dict):
        forbidden = set(value).intersection(_FORBIDDEN_KEYS)
        if forbidden:
            raise ValueError(f"LLM enhancement 禁止输出正文相关字段: {sorted(forbidden)}")
        for child in value.values():
            _reject_forbidden_keys(child)
    elif isinstance(value, list):
        for item in value:
            _reject_forbidden_keys(item)


def _ensure_text(value: object, max_chars: int, label: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"LLM enhancement {label} 必须是字符串。")
    if len(value) > max_chars:
        raise ValueError(f"LLM enhancement {label} 超过长度限制: {len(value)} > {max_chars}")


def _ensure_text_list(value: object, max_items: int, label: str) -> None:
    if not isinstance(value, list):
        raise ValueError(f"LLM enhancement {label} 必须是 list。")
    if len(value) > max_items:
        raise ValueError(f"LLM enhancement {label} 数量过多: {len(value)} > {max_items}")
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"LLM enhancement {label} 只能包含字符串。")


def _join_terms(value: object) -> str:
    items = [item.strip() for item in _as_list(value, "terms") if isinstance(item, str)]
    return "、".join(item for item in items if item)


def _join_queries(value: object) -> str:
    items = [item.strip() for item in _as_list(value, "queries") if isinstance(item, str)]
    return "\n".join(f"- {item}" for item in items if item)


def _as_dict(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"LLM enhancement {label} 必须是 object。")
    return value


def _as_list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"LLM enhancement {label} 必须是 list。")
    return value


def _first_doc_type(parent_documents: list[Document], child_documents: list[Document]) -> str:
    for document in [*parent_documents, *child_documents]:
        doc_type = str(document.metadata.get("doc_type") or "")
        if doc_type:
            return doc_type
    return ""
