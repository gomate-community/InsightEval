"""
Benchmark Insight Evaluation Program
=====================================
使用 LLM 评估 survey 中各章节的洞察力质量。
针对 cites 不为空的章节，通过大模型评估其 synthesis、critical、abstraction 三个维度。

Usage:
    # 评估所有 surveys
    python benckmark/evaluate_insight.py --surveys_path benckmark/SurGE/surveys.json --corpus_path benckmark/SurGE/corpus.parquet --output_dir benckmark/results

    # 评估单个 survey 样例
    python benckmark/evaluate_insight.py --surveys_path benckmark/SurGE/sutvey_sample.json --single --corpus_path benckmark/SurGE/corpus.parquet --output_dir benckmark/results
"""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from loguru import logger
from openai import AsyncOpenAI

# ============================================================
# 1. Prompt Template
# ============================================================

INSIGHT_EVAL_SYSTEM_PROMPT = """You are an expert academic survey analyst specializing in evaluating the insight quality of scholarly survey papers.
Your task is to evaluate how insightfully a survey section integrates and analyzes the cited papers."""

INSIGHT_EVAL_USER_PROMPT = """Evaluate the insight quality of the following survey section.

## Survey Information
- **Survey Title**: {survey_title}
- **Survey Abstract**: {survey_abstract}

## Section Being Evaluated
- **Section Path**: {section_path}
- **Section Title**: {section_title}

### Section Content
{section_content}

### Cited Papers in This Section
{cited_papers_info}

## Task
Evaluate this survey section's insight quality on 3 dimensions (1.0-5.0):

1. **synthesis**: How well does the section synthesize and integrate information from cited papers? Does it connect ideas across multiple sources to create a coherent narrative?
   - 1.0: No synthesis, just lists or paraphrases individual papers
   - 3.0: Basic connections between sources, some integration
   - 5.0: Novel framework combining multiple ideas into new understanding

2. **critical**: Does the section show critical analysis? Does it evaluate, compare, or identify limitations of the cited works rather than just summarizing them?
   - 1.0: No critical perspective, pure description
   - 3.0: Identifies some gaps, comparisons or problems
   - 5.0: Deep evaluative analysis with nuanced critique

3. **abstraction**: Does the section generalize beyond specific papers to identify broader patterns, principles, or frameworks?
   - 1.0: Purely concrete, describes specific systems only
   - 3.0: Some pattern identification across works
   - 5.0: Meta-level insights, identifies overarching principles

Also determine:
- **type**: Is this section primarily `descriptive` (factual summary of systems/methods), `comparative` (comparing different approaches), or `analytical` (providing insights, identifying trends, critical analysis)?
- **insight_level**: Based on the average of the three scores: `low` (avg < 2.0), `medium` (avg 2.0-3.5), `high` (avg > 3.5)

## Output Format
Return ONLY valid JSON (no markdown fences, no extra text):
{{
    "type": "descriptive|comparative|analytical",
    "scores": {{"synthesis": X.X, "critical": X.X, "abstraction": X.X}},
    "insight_level": "low|medium|high",
    "analysis": "brief explanation of the evaluation (2-3 sentences)"
}}"""


# ============================================================
# 2. LLM Client (lightweight, reuses same config as LLMEngine)
# ============================================================

class LLMClient:
    """轻量级 LLM 客户端，复用项目中相同的 API 配置"""

    def __init__(
        self,
        base_url: str = "https://aicloud.oneainexus.cn:30013/inference/aicloud-yanqiang/qwen3-32b-server/v1",
        model_name: str = "Qwen/Qwen3-32B",
        api_key: str = "dummy_key",
        timeout: int = 600,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature

        http_client = httpx.AsyncClient(verify=False)
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            http_client=http_client,
        )

    async def chat(self, system_prompt: str, user_prompt: str) -> str:
        """发送请求并返回文本响应"""
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            extra_body={
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        return response.choices[0].message.content


# ============================================================
# 3. Corpus Loader
# ============================================================

def load_corpus(corpus_path: str) -> Dict[int, Dict[str, str]]:
    """
    加载 corpus 文件，返回 doc_id -> {Title, Abstract} 的字典。
    支持 .parquet 和 .json 两种格式，优先使用 parquet（更快）。
    """
    corpus_path = Path(corpus_path)

    if corpus_path.suffix == ".parquet":
        import pandas as pd
        logger.info(f"从 parquet 加载 corpus: {corpus_path}")
        df = pd.read_parquet(corpus_path)
        corpus_dict = {}
        for _, row in df.iterrows():
            doc_id = int(row["doc_id"])
            corpus_dict[doc_id] = {
                "Title": str(row.get("Title", "")),
                "Abstract": str(row.get("Abstract", "")),
            }
        return corpus_dict
    else:
        logger.info(f"从 JSON 加载 corpus: {corpus_path} (可能较慢...)")
        with open(corpus_path, "r", encoding="utf-8") as f:
            corpus_list = json.load(f)
        corpus_dict = {}
        for item in corpus_list:
            doc_id = int(item["doc_id"])
            corpus_dict[doc_id] = {
                "Title": item.get("Title", ""),
                "Abstract": item.get("Abstract", ""),
            }
        return corpus_dict


# ============================================================
# 4. Survey Processing Utilities
# ============================================================

def load_surveys(surveys_path: str, single: bool = False) -> List[Dict[str, Any]]:
    """加载 surveys 数据"""
    with open(surveys_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if single:
        # 单个 survey（如 sutvey_sample.json）
        if isinstance(data, dict):
            return [data]
        else:
            return [data[0]]
    else:
        # surveys.json 是一个列表
        if isinstance(data, list):
            return data
        else:
            return [data]


def get_sections_with_cites(survey: Dict[str, Any]) -> List[Dict[str, Any]]:
    """从 survey 的 structure 中提取 cites 不为空的章节"""
    sections = []
    for section in survey.get("structure", []):
        if section.get("cites") and len(section["cites"]) > 0:
            sections.append(section)
    return sections


def build_section_path(section: Dict[str, Any]) -> str:
    """从 prefix_titles 构建章节路径字符串，如: 'Title > Section > Subsection'"""
    prefix_titles = section.get("prefix_titles", [])
    return " > ".join([f"[{level}] {title}" for level, title in prefix_titles])


def build_cited_papers_info(
    cites: List[int], corpus: Dict[int, Dict[str, str]]
) -> str:
    """构建引用论文的信息文本"""
    papers_info_parts = []
    for i, doc_id in enumerate(cites, 1):
        paper = corpus.get(doc_id)
        if paper:
            papers_info_parts.append(
                f"**Paper {i}** (doc_id: {doc_id}):\n"
                f"  - Title: {paper['Title']}\n"
                f"  - Abstract: {paper['Abstract'][:500]}{'...' if len(paper['Abstract']) > 500 else ''}"
            )
        else:
            papers_info_parts.append(
                f"**Paper {i}** (doc_id: {doc_id}):\n  - [Not found in corpus]"
            )
    return "\n\n".join(papers_info_parts)


def repair_json_output(content: str) -> str:
    """
    Repair and normalize JSON output.

    Args:
        content (str): String content that may contain JSON

    Returns:
        str: Repaired JSON string, or original content if not JSON
    """
    content = content.strip()
    if content.startswith(("{", "[")) or "```json" in content or "```ts" in content:
        try:
            # If content is wrapped in ```json code block, extract the JSON part
            if content.startswith("```json"):
                content = content.removeprefix("```json")

            if content.startswith("```ts"):
                content = content.removeprefix("```ts")

            if content.endswith("```"):
                content = content.removesuffix("```")

            # Try to repair and parse JSON
            repaired_content = json.loads(content)
            return json.dumps(repaired_content, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"JSON repair failed: {e}")
    return content

def parse_llm_response(response_text: str) -> Optional[Dict[str, Any]]:
    """解析 LLM 返回的 JSON 响应"""
    # text = response_text.strip()
    #
    # # 去掉 markdown 代码块包裹
    # if text.startswith("```"):
    #     lines = text.split("\n")
    #     # 去掉第一行 (```json) 和最后一行 (```)
    #     lines = [l for l in lines if not l.strip().startswith("```")]
    #     text = "\n".join(lines)
    text=repair_json_output(response_text)
    try:
        result = json.loads(text)
        # 验证必要字段
        if "scores" in result and "insight_level" in result:
            return result
        else:
            logger.warning(f"LLM 响应缺少必要字段: {list(result.keys())}")
            return result
    except json.JSONDecodeError as e:
        logger.error(f"JSON 解析失败: {e}\n原始文本: {text[:200]}...")
        return None



# ============================================================
# 5. Main Evaluation Logic
# ============================================================

async def evaluate_section(
    llm: LLMClient,
    survey: Dict[str, Any],
    section: Dict[str, Any],
    corpus: Dict[int, Dict[str, str]],
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    """评估单个章节的洞察力"""
    async with semaphore:
        survey_title = survey.get("survey_title", "Unknown")
        survey_abstract = survey.get("abstract", "")
        section_title = section.get("title", "")
        section_path = build_section_path(section)
        section_content = section.get("content", "")
        cites = section.get("cites", [])
        cited_papers_info = build_cited_papers_info(cites, corpus)

        # 截断过长的内容以防超出 token 限制
        if len(section_content) > 8000:
            section_content = section_content[:8000] + "\n... [content truncated]"
        if len(survey_abstract) > 1000:
            survey_abstract = survey_abstract[:1000] + "..."

        user_prompt = INSIGHT_EVAL_USER_PROMPT.format(
            survey_title=survey_title,
            survey_abstract=survey_abstract,
            section_path=section_path,
            section_title=section_title,
            section_content=section_content,
            cited_papers_info=cited_papers_info,
        )

        try:
            logger.info(
                f"  评估章节: [{section.get('level', '')}] {section_title} "
                f"(cites: {len(cites)} 篇)"
            )
            response_text = await llm.chat(INSIGHT_EVAL_SYSTEM_PROMPT, user_prompt)
            insight_result = parse_llm_response(response_text)

            if insight_result is None:
                insight_result = {
                    "error": "Failed to parse LLM response",
                    "raw_response": response_text[:500],
                }
        except Exception as e:
            logger.error(f"  评估失败: {section_title} - {str(e)}")
            insight_result = {"error": str(e)}

        # 构建输出：原始 section 结构 + insight_result
        result = dict(section)
        result["insight_result"] = insight_result
        return result


async def evaluate_survey(
    llm: LLMClient,
    survey: Dict[str, Any],
    corpus: Dict[int, Dict[str, str]],
    output_dir: Path,
    semaphore: asyncio.Semaphore,
) -> None:
    """评估单个 survey 的所有有引用的章节"""
    survey_id = survey.get("survey_id", 0)
    survey_title = survey.get("survey_title", "Unknown")
    output_path = output_dir / f"survey_{survey_id}.jsonl"

    # 断点续传：如果已有结果文件，则跳过
    if output_path.exists():
        logger.info(f"⏭️  跳过 survey_{survey_id}: {survey_title} (已有结果)")
        return

    sections_with_cites = get_sections_with_cites(survey)
    if not sections_with_cites:
        logger.info(f"⏭️  跳过 survey_{survey_id}: {survey_title} (无引用章节)")
        return

    logger.info(
        f"📝 评估 survey_{survey_id}: {survey_title} "
        f"({len(sections_with_cites)} 个有引用章节)"
    )

    # 并发评估各章节
    tasks = [
        evaluate_section(llm, survey, section, corpus, semaphore)
        for section in sections_with_cites
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 写入 JSONL 文件
    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"  章节评估异常: {result}")
                continue
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    logger.info(
        f"✅ survey_{survey_id} 评估完成，"
        f"结果已保存至 {output_path} ({len(results)} 条)"
    )


async def main(args):
    """主流程"""
    start_time = time.time()

    # 1. 加载 corpus
    logger.info("=" * 60)
    logger.info("📚 加载 corpus...")
    corpus = load_corpus(args.corpus_path)
    logger.info(f"   corpus 加载完成，共 {len(corpus)} 篇论文")

    # 2. 加载 surveys
    logger.info("📖 加载 surveys...")
    surveys = load_surveys(args.surveys_path, single=args.single)
    logger.info(f"   surveys 加载完成，共 {len(surveys)} 篇 survey")

    # 3. 初始化 LLM 客户端
    llm = LLMClient(
        base_url=args.base_url,
        model_name=args.model_name,
        api_key=args.api_key,
        temperature=args.temperature,
    )
    logger.info(f"🤖 LLM 客户端初始化: {args.model_name}")

    # 4. 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 5. 统计信息
    total_sections = sum(
        len(get_sections_with_cites(s)) for s in surveys
    )
    logger.info(f"📊 待评估: {len(surveys)} 篇 survey, {total_sections} 个有引用章节")
    logger.info("=" * 60)

    # 6. 并发控制
    semaphore = asyncio.Semaphore(args.concurrency)

    # 7. 逐个评估 survey
    for i, survey in enumerate(surveys):
        logger.info(f"\n--- [{i+1}/{len(surveys)}] ---")
        await evaluate_survey(llm, survey, corpus, output_dir, semaphore)

    elapsed = time.time() - start_time
    logger.info(f"\n🎉 全部评估完成! 耗时: {elapsed:.1f}s")


# ============================================================
# 6. CLI Entry Point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark Insight Evaluation - 使用 LLM 评估 survey 章节洞察力"
    )
    parser.add_argument(
        "--surveys_path",
        type=str,
        default="benckmark/SurGE/surveys.json",
        help="surveys 数据路径 (surveys.json 或 sutvey_sample.json)",
    )
    parser.add_argument(
        "--corpus_path",
        type=str,
        default="benckmark/SurGE/corpus.parquet",
        help="corpus 数据路径 (.parquet 或 .json)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="benckmark/results",
        help="评估结果输出目录",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="单篇 survey 模式（如 sutvey_sample.json）",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="并发请求数 (默认: 3)",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="https://aicloud.oneainexus.cn:30013/inference/aicloud-yanqiang/qwen3-32b-server/v1",
        help="LLM API base URL",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-32B",
        help="模型名称",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="dummy_key",
        help="API 密钥",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="生成温度 (默认: 0.7)",
    )

    args = parser.parse_args()
    asyncio.run(main(args))

# >python benckmark/evaluate_insight.py --surveys_path benckmark/SurGE/surveys.json --corpus_path benckmark/SurGE/corpus.parquet --output_dir benckmark/results/surge --concurrency 3