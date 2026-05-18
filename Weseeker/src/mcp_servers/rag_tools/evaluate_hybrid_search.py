"""Evaluate Vector, BM25 and Hybrid retrieval on an existing KB.

This is a local analysis script. It does not rebuild indexes and does not define
the future MCP API contract.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document

from config.settings import get_settings
from mcp_servers.rag_tools.adapters import BatchedEmbedder, create_embedding_model
from mcp_servers.rag_tools.indexing.persist.bm25_store import BM25Store
from mcp_servers.rag_tools.indexing.persist.chroma_store import ChromaChildStore
from mcp_servers.rag_tools.indexing.persist.doc_store import ParentDocStore
from mcp_servers.rag_tools.retrieval import (
    BM25Retriever,
    HybridRetriever,
    ParentPromoter,
    VectorRetriever,
)
from mcp_servers.rag_tools.retrieval.query_profile import thresholds_for_query

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")


TOP_K = 5
CANDIDATE_K = 30
HYBRID_MIN_VECTOR_SCORE = 0.45
HYBRID_MIN_BM25_SCORE = 4.0
PARENT_MIN_VECTOR_ONLY_SCORE = 0.50
PARENT_MIN_BM25_ONLY_SCORE = 5.0
MODES = (
    "vector_raw",
    "vector_thresholded",
    "bm25_raw",
    "bm25_thresholded",
    "hybrid_raw",
    "hybrid_thresholded",
    "hybrid_query_aware",
)
MODE_TITLES = {
    "vector_raw": "Vector Raw",
    "vector_thresholded": "Vector Thresholded",
    "bm25_raw": "BM25 Raw",
    "bm25_thresholded": "BM25 Thresholded",
    "hybrid_raw": "Hybrid Raw",
    "hybrid_thresholded": "Hybrid Thresholded",
    "hybrid_query_aware": "Hybrid Query-Aware",
}
THRESHOLD_PAIRS = (
    ("vector_raw", "vector_thresholded"),
    ("bm25_raw", "bm25_thresholded"),
    ("hybrid_raw", "hybrid_thresholded"),
    ("hybrid_thresholded", "hybrid_query_aware"),
)


@dataclass(frozen=True)
class QueryCase:
    query: str
    category: str
    expected_files: tuple[str, ...]
    expected_heading_terms: tuple[str, ...] = ()
    notes: str = ""
    negative: bool = False


@dataclass(frozen=True)
class RetrievalRun:
    mode: str
    query: str
    latency_ms: float
    documents: list[Document]


@dataclass(frozen=True)
class QueryEvaluation:
    case: QueryCase
    runs: dict[str, RetrievalRun]


QUERY_CASES: tuple[QueryCase, ...] = (
    QueryCase(
        query="我之前记的那个 LangGraph interrupt 和 checkpointer 的坑在哪里？",
        category="自然语言 + 英文技术词",
        expected_files=("LangChain的坑.md",),
        expected_heading_terms=("interrupt",),
        notes="用户不会只输入关键词，通常会混入“之前记的”“坑在哪里”这类自然表达。",
    ),
    QueryCase(
        query="DeepSeek 返回 reasoning_content 那个问题我记在哪篇笔记里了？",
        category="精确字段 + 问句",
        expected_files=("LangChain的坑.md",),
        expected_heading_terms=("reasoning", "reasoning_content"),
        notes="精确字段对 BM25 有利，检验 Hybrid 是否保留关键词优势。",
    ),
    QueryCase(
        query="提示词工程里面那个要求模型只输出 JSON 的地方",
        category="短语义块",
        expected_files=("LangChain.md",),
        expected_heading_terms=("提示词工程", "JSON"),
        notes="短 child `只输出 JSON` 是当前分块保护规则的代表样例。",
    ),
    QueryCase(
        query="我想找风电叶片 4K 6K 分块切片和坐标回投那段方案",
        category="中英数字混合 + PDF 技术点",
        expected_files=("风电机组故障预警技术方案-视觉检测.pdf",),
        expected_heading_terms=("分块切片", "坐标回投", "3.1.4"),
        notes="BM25-only 历史上 Top1 会偏到同文件研究目标，Hybrid 应更容易推到精确章节。",
    ),
    QueryCase(
        query="风电叶片视觉检测里面小样本数据增强和裂缝粘贴怎么做？",
        category="PDF 语义技术点",
        expected_files=("风电机组故障预警技术方案-视觉检测.pdf",),
        expected_heading_terms=("小样本", "数据增强", "Poisson", "裂缝"),
        notes="检验 PDF 动态标题结构恢复后的内容级召回。",
    ),
    QueryCase(
        query="LoRA 为什么能省显存，低秩适配器那块训练原理在哪？",
        category="语义改写 + 技术主题",
        expected_files=("LLM微调+低精度训练.md", "LLM微调踩坑.md"),
        expected_heading_terms=("LoRA", "Lora", "低精度", "显存"),
        notes="用户可能不用原文标题，而是描述概念关系。",
    ),
    QueryCase(
        query="我当时记录的 Lora 训练踩坑和 SFTTrainer 问题",
        category="精确技术词 + 主题词",
        expected_files=("LLM微调踩坑.md", "LLM微调+低精度训练.md"),
        expected_heading_terms=("SFTTrainer", "Lora", "LoRA"),
        notes="同时包含精确类名和自然语言“踩坑”。",
    ),
    QueryCase(
        query="RAG 里面说召回、重排、评估这些技术点的那篇总结",
        category="主题概括",
        expected_files=("RAG技术点.md", "RAG学习.md"),
        expected_heading_terms=("召回", "重排", "评估", "RAG"),
        notes="偏内容主题，检验向量语义召回和 Hybrid 覆盖。",
    ),
    QueryCase(
        query="我目前的 Agent 学习计划接下来应该怎么安排？",
        category="标题/文件名型自然问法",
        expected_files=("目前 Agent 学习计划.md",),
        expected_heading_terms=("Agent", "计划"),
        notes="文件名和标题信息较强，BM25 通常应表现稳定。",
    ),
    QueryCase(
        query="LangChain 有哪些坑，特别是输出 reasoning 的问题",
        category="泛词 + 精确约束",
        expected_files=("LangChain的坑.md",),
        expected_heading_terms=("reasoning", "reasoning_content", "LangChain"),
        notes="泛词“坑”会引入噪音，精确约束应帮助排序。",
    ),
    QueryCase(
        query="研究生阶段我写过哪些应对策略和心态调整？",
        category="生活类语义查询",
        expected_files=("研究生阶段应对策略.md", "我对读研生涯的再认识.docx"),
        expected_heading_terms=("研究生", "读研", "应对策略"),
        notes="非技术内容，检验用户真实自然表达下的主题召回。",
    ),
    QueryCase(
        query="找一下我写的朋友分级制度，哪些朋友应该深交？",
        category="生活类标题 + 内容",
        expected_files=("朋友分级体系.md", "朋友分级制度.md"),
        expected_heading_terms=("朋友", "分级", "深交"),
        notes="相近文件名较多，检验 TopK 覆盖。",
    ),
    QueryCase(
        query="婚后家庭事务里面关于家务和经济分工的想法",
        category="生活类内容定位",
        expected_files=("对家庭婚后事务的看法.md",),
        expected_heading_terms=("家庭", "婚后", "家务", "经济"),
        notes="自然语言内容定位，文件名只有部分线索。",
    ),
    QueryCase(
        query="就业计划里有没有提到违约金或者实习协议风险？",
        category="单关键词 + 场景约束",
        expected_files=("实习就业计划（2025.12.4）.md",),
        expected_heading_terms=("违约金", "实习", "协议", "就业"),
        notes="单关键词容易误召回，额外场景词应帮助过滤。",
    ),
    QueryCase(
        query="李沐 NLP 学习里 Attention 计算那块笔记",
        category="人名/课程 + 英文术语",
        expected_files=("LM note.md",),
        expected_heading_terms=("Attention", "注意力", "李沐", "NLP"),
        notes="BM25 和向量都可能命中，观察排序差异。",
    ),
    QueryCase(
        query="强化学习里面 PPO、GRPO、奖励模型这些训练方法总结",
        category="多技术词主题",
        expected_files=("LM的RL训练方法学习.md", "LM的RL训练总览.md"),
        expected_heading_terms=("PPO", "GRPO", "奖励", "RL"),
        notes="多关键词主题查询，检验 TopK 覆盖。",
    ),
    QueryCase(
        query="我对恋爱相亲前需要注意什么写过一篇东西",
        category="模糊文件名 + 内容",
        expected_files=("打算恋爱or相亲前的注意事项.md",),
        expected_heading_terms=("恋爱", "相亲", "注意"),
        notes="用户回忆式表达，不完全等于文件名。",
    ),
    QueryCase(
        query="违约金",
        category="短关键词",
        expected_files=("实习就业计划（2025.12.4）.md",),
        expected_heading_terms=("违约金", "实习", "就业"),
        notes="短 query 用来观察 BM25 4.0 阈值是否误伤低分但有效命中。",
    ),
    QueryCase(
        query="Attention 计算",
        category="短技术词",
        expected_files=("LM note.md",),
        expected_heading_terms=("Attention", "注意力", "计算"),
        notes="短技术 query 用来观察向量和 BM25 阈值是否仍保留技术命中。",
    ),
    QueryCase(
        query="SFTTrainer",
        category="短精确 API 名",
        expected_files=("LLM微调踩坑.md",),
        expected_heading_terms=("SFTTrainer",),
        notes="短精确 API 名通常应由 BM25 保底召回。",
    ),
    QueryCase(
        query="火星土豆烤鸭",
        category="短负例",
        expected_files=(),
        notes="短负例用于观察阈值能否过滤随机词项和向量最近邻噪音。",
        negative=True,
    ),
    QueryCase(
        query="火星土豆烤鸭这种完全不相关的东西",
        category="负例",
        expected_files=(),
        notes="负例应尽量无命中，Hybrid 当前有低相关 vector-only gate。",
        negative=True,
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="评估 Vector/BM25/Hybrid 在真实 KB 上的命中率")
    parser.add_argument("--kb-name", default="test_kb_notes")
    parser.add_argument(
        "--output",
        default="storage/debug/hybrid_vs_bm25_vector_evaluation_test_kb_notes_2026-04-28.md",
    )
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--candidate-k", type=int, default=CANDIDATE_K)
    args = parser.parse_args()

    settings = get_settings()
    kb_name = args.kb_name
    top_k = max(1, args.top_k)
    candidate_k = max(top_k, args.candidate_k)
    output_path = Path(args.output)

    embedding_model = create_embedding_model(settings.rag)
    embedder = BatchedEmbedder(
        embedding_model,
        batch_size=settings.rag.embedding_batch_size,
        max_retries=settings.rag.embedding_max_retries,
    )
    chroma_store = ChromaChildStore(settings.rag.chroma_persist_dir, kb_name)
    bm25_store = BM25Store(settings.rag.bm25_dir, kb_name)
    doc_store = ParentDocStore(settings.rag.docstore_dir, kb_name)

    vector_retriever = VectorRetriever(chroma_store, embedder)
    bm25_retriever = BM25Retriever(bm25_store)
    hybrid_raw_retriever = HybridRetriever(
        vector_retriever,
        bm25_retriever,
        min_vector_score=None,
        min_bm25_score=None,
        query_aware_thresholds=False,
    )
    hybrid_thresholded_retriever = HybridRetriever(
        vector_retriever,
        bm25_retriever,
        min_vector_score=HYBRID_MIN_VECTOR_SCORE,
        min_bm25_score=HYBRID_MIN_BM25_SCORE,
        query_aware_thresholds=False,
    )
    hybrid_query_aware_retriever = HybridRetriever(
        vector_retriever,
        bm25_retriever,
        min_vector_score=HYBRID_MIN_VECTOR_SCORE,
        min_bm25_score=HYBRID_MIN_BM25_SCORE,
        query_aware_thresholds=True,
    )
    parent_raw_promoter = ParentPromoter(
        doc_store,
        min_vector_only_score=0.0,
        min_bm25_only_score=0.0,
        min_vector_score=0.0,
        min_bm25_score=0.0,
    )
    parent_thresholded_promoter = ParentPromoter(
        doc_store,
        min_vector_only_score=PARENT_MIN_VECTOR_ONLY_SCORE,
        min_bm25_only_score=PARENT_MIN_BM25_ONLY_SCORE,
        min_vector_score=HYBRID_MIN_VECTOR_SCORE,
        min_bm25_score=HYBRID_MIN_BM25_SCORE,
    )

    evaluations: list[QueryEvaluation] = []
    for case in QUERY_CASES:
        query = case.query
        vector_raw_run = _time_run(
            "vector_raw",
            query,
            lambda query=query: vector_retriever.search(query, top_k=top_k),
        )
        vector_thresholded_run = _time_run(
            "vector_thresholded",
            query,
            lambda query=query: vector_retriever.search(
                query,
                top_k=top_k,
                min_score=HYBRID_MIN_VECTOR_SCORE,
            ),
        )
        bm25_raw_run = _time_run(
            "bm25_raw",
            query,
            lambda query=query: bm25_retriever.search(query, top_k=top_k),
        )
        bm25_thresholded_run = _time_run(
            "bm25_thresholded",
            query,
            lambda query=query: bm25_retriever.search(
                query,
                top_k=top_k,
                min_score=HYBRID_MIN_BM25_SCORE,
            ),
        )
        hybrid_raw_run = _run_hybrid(
            "hybrid_raw",
            query,
            hybrid_raw_retriever,
            parent_raw_promoter,
            top_k=top_k,
            candidate_k=candidate_k,
        )
        hybrid_thresholded_run = _run_hybrid(
            "hybrid_thresholded",
            query,
            hybrid_thresholded_retriever,
            parent_thresholded_promoter,
            top_k=top_k,
            candidate_k=candidate_k,
        )
        hybrid_query_aware_run = _run_hybrid(
            "hybrid_query_aware",
            query,
            hybrid_query_aware_retriever,
            parent_thresholded_promoter,
            top_k=top_k,
            candidate_k=candidate_k,
            query_aware_parent_thresholds=True,
        )
        evaluations.append(
            QueryEvaluation(
                case=case,
                runs={
                    "vector_raw": vector_raw_run,
                    "vector_thresholded": vector_thresholded_run,
                    "bm25_raw": bm25_raw_run,
                    "bm25_thresholded": bm25_thresholded_run,
                    "hybrid_raw": hybrid_raw_run,
                    "hybrid_thresholded": hybrid_thresholded_run,
                    "hybrid_query_aware": hybrid_query_aware_run,
                },
            )
        )
        print(f"evaluated: {case.query}")

    report = build_report(evaluations, kb_name=kb_name, top_k=top_k, candidate_k=candidate_k)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"report written: {output_path}")


def _time_run(mode: str, query: str, callback) -> RetrievalRun:
    started_at = time.perf_counter()
    documents = callback()
    return RetrievalRun(
        mode=mode,
        query=query,
        latency_ms=(time.perf_counter() - started_at) * 1000,
        documents=documents,
    )


def _run_hybrid(
    mode: str,
    query: str,
    hybrid_retriever: HybridRetriever,
    parent_promoter: ParentPromoter,
    *,
    top_k: int,
    candidate_k: int,
    query_aware_parent_thresholds: bool = False,
) -> RetrievalRun:
    child_run = _time_run(
        mode,
        query,
        lambda: hybrid_retriever.search(query, top_k=candidate_k, candidate_k=candidate_k),
    )
    promote_started_at = time.perf_counter()
    if query_aware_parent_thresholds:
        thresholds = thresholds_for_query(
            query,
            vector_min_score=HYBRID_MIN_VECTOR_SCORE,
            bm25_child_min_score=HYBRID_MIN_BM25_SCORE,
            vector_only_parent_min_score=PARENT_MIN_VECTOR_ONLY_SCORE,
            bm25_only_parent_min_score=PARENT_MIN_BM25_ONLY_SCORE,
            vector_parent_min_score=HYBRID_MIN_VECTOR_SCORE,
            bm25_parent_min_score=HYBRID_MIN_BM25_SCORE,
        )
        documents = parent_promoter.promote(
            child_run.documents,
            top_k=top_k,
            min_vector_only_score=thresholds.vector_only_parent_min_score,
            min_bm25_only_score=thresholds.bm25_only_parent_min_score,
            min_vector_score=thresholds.vector_parent_min_score,
            min_bm25_score=thresholds.bm25_parent_min_score,
        )
    else:
        documents = parent_promoter.promote(child_run.documents, top_k=top_k)
    latency_ms = child_run.latency_ms + ((time.perf_counter() - promote_started_at) * 1000)
    return RetrievalRun(mode=mode, query=query, latency_ms=latency_ms, documents=documents)


def build_report(
    evaluations: list[QueryEvaluation],
    *,
    kb_name: str,
    top_k: int,
    candidate_k: int,
) -> str:
    modes = MODES
    lines = [
        f"# Hybrid vs BM25 vs Vector 检索评估报告 - {kb_name}",
        "",
        "> 验证日期：2026-04-28",
        f"> 验证对象：`{kb_name}`",
        "> 验证方式：复用已有 Chroma / BM25 / ParentDocStore，不重建 KB。",
        "> Query 设计：模拟用户真实聊天表达，而不是只输入标准关键词。",
        "",
        "---",
        "",
        "## 1. 评估范围",
        "",
        "本轮只评估检索层命中率和查询延迟，并对比 raw / fixed threshold / query-aware 行为：",
        "",
        "- `vector_raw`：`VectorRetriever` 原始 TopK，返回 child-level hit。",
        "- `vector_thresholded`：`VectorRetriever(min_score=0.45)`。",
        "- `bm25_raw`：`BM25Retriever` 原始 TopK，返回 child-level hit。",
        "- `bm25_thresholded`：`BM25Retriever(min_score=4.0)`。",
        "- `hybrid_raw`：不启用 child / parent 阈值的 Hybrid + ParentPromote。",
        (
            "- `hybrid_thresholded`：启用阈值的 Hybrid + ParentPromote，"
            "用于保留固定阈值对照。"
        ),
        (
            "- `hybrid_query_aware`：自然语言 query 使用固定阈值，短 query 放宽 BM25 "
            "阈值并增加 token overlap gate，作为当前建议的候选路径。"
        ),
        (
            "- 固定阈值：vector child `>= 0.45`，BM25 child `>= 4.0`，"
            "vector-only parent `>= 0.50`，BM25-only parent `>= 5.0`。"
        ),
        "- 短 query query-aware 阈值：BM25 child 不设 `min_score`，BM25 parent `>= 2.0`。",
        "",
        "本轮不做：",
        "",
        "- 不重建 KB。",
        "- 不修改 Parent / Child 分块边界。",
        "- 不接 MCP / Agent。",
        "- 不做 rerank。",
        "",
        "---",
        "",
        "## 2. 评估指标",
        "",
        "命中判定基于人工定义的期望文件名和章节关键词。",
        "",
        "- `Hit@1`：Top1 是否命中期望文件，且章节/正文包含期望关键词之一。",
        "- `Hit@3`：Top3 内是否命中。",
        "- `Hit@5`：Top5 内是否命中。",
        "- `MRR@5`：Top5 内首个命中的倒数排名；无命中为 0。",
        "- `Negative Clean`：负例 query 是否无结果。",
        "- `Duplicate Parent`：Top5 内同一 `parent_id` 重复次数，数值越低越好。",
        "- `Latency`：单进程 warm path 下每条 query 的检索耗时，包含 query embedding。",
        "",
    ]

    lines.extend(_build_summary_table(evaluations, modes, top_k))
    lines.extend(_build_threshold_impact_table(evaluations, top_k))
    lines.extend(_build_category_table(evaluations, modes))
    lines.extend(_build_query_table(evaluations, modes))
    lines.extend(_build_case_details(evaluations, modes))
    lines.extend(_build_conclusions(evaluations, modes))
    lines.extend(
        [
            "---",
            "",
            "## 9. 运行参数",
            "",
            f"- `top_k`: `{top_k}`",
            f"- `candidate_k`: `{candidate_k}`",
            f"- `hybrid_min_vector_score`: `{HYBRID_MIN_VECTOR_SCORE}`",
            f"- `hybrid_min_bm25_score`: `{HYBRID_MIN_BM25_SCORE}`",
            f"- `parent_min_vector_only_score`: `{PARENT_MIN_VECTOR_ONLY_SCORE}`",
            f"- `parent_min_bm25_only_score`: `{PARENT_MIN_BM25_ONLY_SCORE}`",
            f"- query 数：`{len(evaluations)}`",
            "- 报告生成脚本：`src/mcp_servers/rag_tools/evaluate_hybrid_search.py`",
            "",
        ]
    )
    return "\n".join(lines)


def _build_summary_table(
    evaluations: list[QueryEvaluation], modes: tuple[str, ...], top_k: int
) -> list[str]:
    positive = [evaluation for evaluation in evaluations if not evaluation.case.negative]
    negative = [evaluation for evaluation in evaluations if evaluation.case.negative]
    lines = ["## 3. 总体命中率", ""]
    lines.append("| 模式 | Hit@1 | Hit@3 | Hit@5 | MRR@5 | 负例干净率 | "
                 "平均重复 parent | 平均延迟 ms | P95 延迟 ms |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for mode in modes:
        hit1 = _ratio(_hit_at(positive, mode, 1), len(positive))
        hit3 = _ratio(_hit_at(positive, mode, min(3, top_k)), len(positive))
        hit5 = _ratio(_hit_at(positive, mode, top_k), len(positive))
        mrr = _avg([_reciprocal_rank(evaluation, mode, top_k) for evaluation in positive])
        negative_clean = _ratio(
            sum(1 for evaluation in negative if not evaluation.runs[mode].documents),
            len(negative),
        )
        duplicate_parent_avg = _avg(
            [
                _duplicate_parent_count(evaluation.runs[mode].documents[:top_k])
                for evaluation in evaluations
            ]
        )
        latencies = [evaluation.runs[mode].latency_ms for evaluation in evaluations]
        lines.append(
            "| {mode} | {hit1:.1%} | {hit3:.1%} | {hit5:.1%} | {mrr:.3f} | "
            "{negative_clean:.1%} | {dup:.2f} | {avg_latency:.1f} | {p95:.1f} |".format(
                mode=MODE_TITLES[mode],
                hit1=hit1,
                hit3=hit3,
                hit5=hit5,
                mrr=mrr,
                negative_clean=negative_clean,
                dup=duplicate_parent_avg,
                avg_latency=_avg(latencies),
                p95=_p95(latencies),
            )
        )
    lines.extend(["", "---", ""])
    return lines


def _build_threshold_impact_table(evaluations: list[QueryEvaluation], top_k: int) -> list[str]:
    positive = [evaluation for evaluation in evaluations if not evaluation.case.negative]
    negative = [evaluation for evaluation in evaluations if evaluation.case.negative]
    lines = ["## 4. 阈值影响", ""]
    lines.append(
        "| 对比 | Hit@1 变化 | Hit@5 变化 | MRR@5 变化 | 负例干净率变化 | 被阈值清空的 Query |"
    )
    lines.append("|---|---:|---:|---:|---:|---|")
    for raw_mode, thresholded_mode in THRESHOLD_PAIRS:
        raw_hit1 = _ratio(_hit_at(positive, raw_mode, 1), len(positive))
        new_hit1 = _ratio(_hit_at(positive, thresholded_mode, 1), len(positive))
        raw_hit5 = _ratio(_hit_at(positive, raw_mode, top_k), len(positive))
        new_hit5 = _ratio(_hit_at(positive, thresholded_mode, top_k), len(positive))
        raw_mrr = _avg([_reciprocal_rank(evaluation, raw_mode, top_k) for evaluation in positive])
        new_mrr = _avg(
            [_reciprocal_rank(evaluation, thresholded_mode, top_k) for evaluation in positive]
        )
        raw_negative = _negative_clean_rate(negative, raw_mode)
        new_negative = _negative_clean_rate(negative, thresholded_mode)
        emptied = [
            evaluation.case.query
            for evaluation in evaluations
            if evaluation.runs[raw_mode].documents
            and not evaluation.runs[thresholded_mode].documents
        ]
        lines.append(
            "| {raw} -> {new} | {hit1:+.1%} | {hit5:+.1%} | {mrr:+.3f} | "
            "{negative:+.1%} | {emptied} |".format(
                raw=MODE_TITLES[raw_mode],
                new=MODE_TITLES[thresholded_mode],
                hit1=new_hit1 - raw_hit1,
                hit5=new_hit5 - raw_hit5,
                mrr=new_mrr - raw_mrr,
                negative=new_negative - raw_negative,
                emptied=_escape_table("；".join(emptied) or "无"),
            )
        )
    lines.extend(["", "---", ""])
    return lines


def _build_category_table(evaluations: list[QueryEvaluation], modes: tuple[str, ...]) -> list[str]:
    categories = sorted(
        {evaluation.case.category for evaluation in evaluations if not evaluation.case.negative}
    )
    lines = ["## 5. 分类命中率", ""]
    header = "| 类别 | Query 数 | " + " | ".join(
        f"{MODE_TITLES[mode]} Hit@5" for mode in modes
    ) + " | 观察 |"
    lines.append(header)
    lines.append("|---|---:" + "|---:" * len(modes) + "|---|")
    for category in categories:
        group = [
            evaluation
            for evaluation in evaluations
            if evaluation.case.category == category and not evaluation.case.negative
        ]
        observations = _category_observation(group, modes)
        values = " | ".join(
            f"{_ratio(_hit_at(group, mode, TOP_K), len(group)):.1%}"
            for mode in modes
        )
        lines.append(
            "| {category} | {count} | {values} | {obs} |".format(
                category=_escape_table(category),
                count=len(group),
                values=values,
                obs=_escape_table(observations),
            )
        )
    lines.extend(["", "---", ""])
    return lines


def _build_query_table(evaluations: list[QueryEvaluation], modes: tuple[str, ...]) -> list[str]:
    lines = ["## 6. 逐 Query 对比", ""]
    header = "| # | 用户式 Query | 类别 | 期望文件 | " + " | ".join(
        MODE_TITLES[mode] for mode in modes
    ) + " | 最佳观察 |"
    lines.append(header)
    lines.append("|---:|---|---|---" + "|---" * len(modes) + "|---|")
    for index, evaluation in enumerate(evaluations, start=1):
        mode_cells = []
        for mode in modes:
            rank = _first_hit_rank(evaluation, mode, TOP_K)
            if evaluation.runs[mode].documents:
                top1 = _result_label(evaluation.runs[mode].documents[0])
            else:
                top1 = "无结果"
            if evaluation.case.negative:
                status = "干净" if not evaluation.runs[mode].documents else "误召回"
            else:
                status = f"Hit@{rank}" if rank else "Miss"
            mode_cells.append(f"{status}: {top1}")
        lines.append(
            "| {index} | {query} | {category} | {expected} | {mode_cells} | {note} |".format(
                index=index,
                query=_escape_table(evaluation.case.query),
                category=_escape_table(evaluation.case.category),
                expected=_escape_table(" / ".join(evaluation.case.expected_files) or "应无结果"),
                mode_cells=" | ".join(_escape_table(cell) for cell in mode_cells),
                note=_escape_table(_best_observation(evaluation, modes)),
            )
        )
    lines.extend(["", "---", ""])
    return lines


def _build_case_details(evaluations: list[QueryEvaluation], modes: tuple[str, ...]) -> list[str]:
    lines = ["## 7. 典型案例", ""]
    selected = _select_interesting_cases(evaluations, modes)
    for evaluation in selected:
        lines.extend([f"### {evaluation.case.query}", "", f"类别：{evaluation.case.category}", ""])
        if evaluation.case.notes:
            lines.extend([f"人工预期：{evaluation.case.notes}", ""])
        for mode in modes:
            run = evaluation.runs[mode]
            rank = _first_hit_rank(evaluation, mode, TOP_K)
            if evaluation.case.negative:
                hit_text = "干净无结果" if not run.documents else "误召回"
            else:
                hit_text = f"Hit@{rank}" if rank else "Miss"
            lines.extend(
                [
                    (
                        f"- `{MODE_TITLES[mode]}`：{hit_text}，耗时 `{run.latency_ms:.1f} ms`，"
                        f"Top3：{_topn_labels(run.documents, 3)}"
                    ),
                ]
            )
        lines.append("")
    lines.extend(["---", ""])
    return lines


def _build_conclusions(evaluations: list[QueryEvaluation], modes: tuple[str, ...]) -> list[str]:
    positive = [evaluation for evaluation in evaluations if not evaluation.case.negative]
    lines = ["## 8. 分析结论", ""]
    hit5 = {mode: _ratio(_hit_at(positive, mode, TOP_K), len(positive)) for mode in modes}
    hit1 = {mode: _ratio(_hit_at(positive, mode, 1), len(positive)) for mode in modes}
    fastest_mode = min(
        modes,
        key=lambda mode: _avg([evaluation.runs[mode].latency_ms for evaluation in evaluations]),
    )
    best_hit5_mode = max(modes, key=lambda mode: hit5[mode])
    best_hit1_mode = max(modes, key=lambda mode: hit1[mode])
    lines.extend(
        [
            (
                f"- Top5 召回最好的模式：`{MODE_TITLES[best_hit5_mode]}`，"
                f"Hit@5 = `{hit5[best_hit5_mode]:.1%}`。"
            ),
            (
                f"- Top1 排序最好的模式：`{MODE_TITLES[best_hit1_mode]}`，"
                f"Hit@1 = `{hit1[best_hit1_mode]:.1%}`。"
            ),
            f"- 平均查询最快的模式：`{MODE_TITLES[fastest_mode]}`。",
            "- `thresholded` 列用于观察阈值影响；`raw` 列保留原始最近邻 / 词项召回基线。",
            "- BM25 对精确字段、英文技术 token、标题型 query 通常更敏感。",
            (
                "- Vector 对语义改写和自然语言表达更有帮助，"
                "但可能在负例或弱相关 query 上返回最近邻噪音。"
            ),
            (
                "- 阈值接入后，Hybrid Thresholded 对短负例和自然语言负例均返回无结果；"
                "raw 基线仍会误召回。"
            ),
            (
                "- Hybrid 的核心收益不是让每个单例都变好，而是在同一个排序中保留 "
                "BM25 的词项召回和 Vector 的语义召回，并通过 Parent Promote 减少同 "
                "parent 多 child 重复展示。"
            ),
            (
                "- 当前 Hybrid 仍没有 rerank；泛词 query 或同文件多章节竞争时，"
                "TopK 内可能有相关但不够精确的结果。"
            ),
            (
                "- 如果后续要继续提升 Top1，优先考虑 lightweight rerank 或 "
                "exact phrase boost，而不是继续只调 BM25/Vector 单路。"
            ),
            "",
        ]
    )
    return lines


def _select_interesting_cases(
    evaluations: list[QueryEvaluation], modes: tuple[str, ...]
) -> list[QueryEvaluation]:
    selected: list[QueryEvaluation] = []
    for evaluation in evaluations:
        ranks = {mode: _first_hit_rank(evaluation, mode, TOP_K) for mode in modes}
        if evaluation.case.negative or len(set(ranks.values())) > 1:
            selected.append(evaluation)
    if len(selected) < 6:
        selected.extend(evaluation for evaluation in evaluations if evaluation not in selected)
    return selected[:8]


def _hit_at(evaluations: list[QueryEvaluation], mode: str, k: int) -> int:
    return sum(1 for evaluation in evaluations if _first_hit_rank(evaluation, mode, k))


def _first_hit_rank(evaluation: QueryEvaluation, mode: str, k: int) -> int | None:
    if evaluation.case.negative:
        return None
    for rank, document in enumerate(evaluation.runs[mode].documents[:k], start=1):
        if _matches_case(document, evaluation.case):
            return rank
    return None


def _reciprocal_rank(evaluation: QueryEvaluation, mode: str, k: int) -> float:
    rank = _first_hit_rank(evaluation, mode, k)
    return 0.0 if rank is None else 1.0 / rank


def _matches_case(document: Document, case: QueryCase) -> bool:
    metadata = document.metadata
    file_name = str(metadata.get("file_name") or "")
    file_match = any(expected in file_name for expected in case.expected_files)
    if not file_match:
        return False
    if not case.expected_heading_terms:
        return True
    haystack = " ".join(
        [
            str(metadata.get("heading_path") or ""),
            str(metadata.get("section_title") or ""),
            document.page_content,
            " ".join(str(snippet) for snippet in metadata.get("matched_child_snippets", [])),
            str(metadata.get("best_child_snippet") or ""),
        ]
    ).casefold()
    return any(term.casefold() in haystack for term in case.expected_heading_terms)


def _duplicate_parent_count(documents: list[Document]) -> int:
    parent_ids = [str(document.metadata.get("parent_id") or "") for document in documents]
    parent_ids = [parent_id for parent_id in parent_ids if parent_id]
    return len(parent_ids) - len(set(parent_ids))


def _result_label(document: Document) -> str:
    metadata = document.metadata
    file_name = str(metadata.get("file_name") or "<无文件名>")
    heading = str(metadata.get("heading_path") or "<无章节>")
    return f"{file_name} / {heading}"


def _topn_labels(documents: list[Document], n: int) -> str:
    if not documents:
        return "无结果"
    return "；".join(_result_label(document) for document in documents[:n])


def _best_observation(evaluation: QueryEvaluation, modes: tuple[str, ...]) -> str:
    if evaluation.case.negative:
        clean_modes = [mode for mode in modes if not evaluation.runs[mode].documents]
        if not clean_modes:
            return "全部模式均误召回"
        return "干净无结果：" + ", ".join(MODE_TITLES[mode] for mode in clean_modes)

    ranks = {mode: _first_hit_rank(evaluation, mode, TOP_K) for mode in modes}
    hit_modes = [mode for mode, rank in ranks.items() if rank]
    if not hit_modes:
        return "全部模式均未命中人工期望"
    best_rank = min(rank for rank in ranks.values() if rank is not None)
    best_modes = [mode for mode, rank in ranks.items() if rank == best_rank]
    return f"最佳 {', '.join(MODE_TITLES[mode] for mode in best_modes)} @ {best_rank}"


def _category_observation(group: list[QueryEvaluation], modes: tuple[str, ...]) -> str:
    if not group:
        return "无"
    hit5 = {mode: _ratio(_hit_at(group, mode, TOP_K), len(group)) for mode in modes}
    best = max(modes, key=lambda mode: hit5[mode])
    if len({hit5[mode] for mode in modes}) == 1:
        return "各模式 Top5 持平"
    return f"{MODE_TITLES[best]} Top5 更稳"


def _ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _negative_clean_rate(evaluations: list[QueryEvaluation], mode: str) -> float:
    return _ratio(
        sum(1 for evaluation in evaluations if not evaluation.runs[mode].documents),
        len(evaluations),
    )


def _avg(values: list[float]) -> float:
    return 0.0 if not values else statistics.fmean(values)


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    return sorted(values)[min(len(values) - 1, int(len(values) * 0.95))]


def _escape_table(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    main()
