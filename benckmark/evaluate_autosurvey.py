"""
Autosurvey Baseline Evaluation Script
======================================
使用 LLM 评估 Autosurvey baseline 生成的 survey 各章节洞察力质量。
Autosurvey 的数据格式：每个文件夹中包含一个 JSON 文件，包含：
  - "survey": markdown 格式的 survey 全文
  - "references": 引用编号 -> arXiv ID 的映射

Usage:
    python benckmark/evaluate_autosurvey.py \
        --input_dir benckmark/SurGE/baselines/Autosurvey/output \
        --output_dir benckmark/results/autosurvey \
        --cache_path benckmark/paper_info_cache.json

    # 指定评估某些文件夹
    python benckmark/evaluate_autosurvey.py \
        --input_dir benckmark/SurGE/baselines/Autosurvey/output \
        --folders 0 1 2
"""

import argparse
import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from loguru import logger
from openai import AsyncOpenAI

# 复用 evaluate_insight.py 中的 prompt 与工具
from evaluate_insight import (
    INSIGHT_EVAL_SYSTEM_PROMPT,
    INSIGHT_EVAL_USER_PROMPT,
    LLMClient,
    parse_llm_response,
)


# ============================================================
# 1. arXiv Paper Info Fetcher (with caching)
# ============================================================

class PaperInfoCache:
    """本地缓存 arXiv 论文信息，避免重复 API 调用"""

    def __init__(self, cache_path: str = "paper_info_cache.json"):
        self.cache_path = Path(cache_path)
        self.cache: Dict[str, Dict[str, str]] = {}
        self._load()

    def _load(self):
        if self.cache_path.exists():
            with open(self.cache_path, "r", encoding="utf-8") as f:
                self.cache = json.load(f)
            logger.info(f"📦 从缓存加载了 {len(self.cache)} 条论文信息")

    def save(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)

    def get(self, arxiv_id: str) -> Optional[Dict[str, str]]:
        return self.cache.get(arxiv_id)

    def set(self, arxiv_id: str, info: Dict[str, str]):
        self.cache[arxiv_id] = info


def fetch_paper_info(arxiv_id: str) -> Dict[str, str]:
    """
    通过 ArxivAPIWrapper 获取论文信息。
    与 get_paper_info.py 使用相同的方法。
    """
    try:
        from langchain_community.utilities import ArxivAPIWrapper
        wrapper = ArxivAPIWrapper()
        results = wrapper._fetch_results(arxiv_id)
        for result in results:
            return {
                "Title": result.title,
                "Abstract": result.summary,
            }
        return {"Title": "", "Abstract": ""}
    except Exception as e:
        logger.warning(f"获取论文 {arxiv_id} 信息失败: {e}")
        return {"Title": "", "Abstract": ""}


async def fetch_papers_for_references(
    references: Dict[str, str],
    cache: PaperInfoCache,
    delay: float = 1.0,
) -> Dict[str, Dict[str, str]]:
    """
    批量获取 references 中所有论文的信息。
    使用缓存 + 延迟来避免 API 限流。
    返回: arxiv_id -> {Title, Abstract}
    """
    paper_info_map: Dict[str, Dict[str, str]] = {}
    fetch_count = 0

    for cite_num, arxiv_id in references.items():
        cached = cache.get(arxiv_id)
        if cached is not None:
            paper_info_map[arxiv_id] = cached
            continue

        # 需要从 API 获取
        logger.debug(f"  从 arXiv 获取: {arxiv_id}")
        info = await asyncio.to_thread(fetch_paper_info, arxiv_id)
        cache.set(arxiv_id, info)
        paper_info_map[arxiv_id] = info
        fetch_count += 1

        # 每获取一篇延迟一下，避免 API 限流
        if delay > 0:
            await asyncio.sleep(delay)

    if fetch_count > 0:
        cache.save()
        logger.info(f"  新获取 {fetch_count} 篇论文信息，已更新缓存")

    return paper_info_map


# ============================================================
# 2. Markdown Section Parser
# ============================================================

def parse_markdown_sections(markdown_text: str) -> List[Dict[str, Any]]:
    """
    解析 markdown 文本，按标题拆分为章节列表。
    返回: [{level, title, content, cites, section_path}, ...]
    """
    # 匹配 markdown 标题行: ## Title, ### Title, etc.
    heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)

    headings = list(heading_pattern.finditer(markdown_text))
    if not headings:
        return []

    sections = []
    # 用于构建层级路径
    title_stack: List[Tuple[int, str]] = []

    for i, match in enumerate(headings):
        level = len(match.group(1))  # '#' 的数量
        title = match.group(2).strip()

        # 获取章节内容（从当前标题到下一个标题之间）
        start = match.end()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(markdown_text)
        content = markdown_text[start:end].strip()

        # 跳过空内容的章节
        if not content:
            continue

        # 更新标题栈，构建层级路径
        while title_stack and title_stack[-1][0] >= level:
            title_stack.pop()
        title_stack.append((level, title))

        section_path = " > ".join(
            [f"[H{lvl}] {ttl}" for lvl, ttl in title_stack]
        )

        # 提取引用编号 [1], [2], [3,4], [1, 5] 等
        cites = extract_citations(content)

        sections.append({
            "level": level,
            "title": title,
            "content": content,
            "cites": cites,
            "section_path": section_path,
        })

    return sections


def extract_citations(text: str) -> List[str]:
    """
    从文本中提取所有引用编号。
    支持 [1], [2], [3,4], [1, 5, 10] 等格式。
    返回去重后的引用编号列表（字符串）。
    """
    cite_ids = set()
    # 匹配 [数字] 或 [数字, 数字, ...] 格式
    pattern = re.compile(r'\[(\d+(?:\s*,\s*\d+)*)\]')
    for match in pattern.finditer(text):
        nums = match.group(1).split(",")
        for num in nums:
            num = num.strip()
            if num:
                cite_ids.add(num)
    return sorted(cite_ids, key=lambda x: int(x))


# ============================================================
# 3. Survey Loading & Processing
# ============================================================

def load_autosurvey_folders(
    input_dir: str, folders: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    加载 Autosurvey output 目录中的所有 survey 数据。
    返回: [{folder_id, survey_title, survey_text, references, json_path}, ...]
    """
    input_path = Path(input_dir)
    surveys = []

    if folders is None:
        # 自动发现所有数字编号的子目录
        folder_candidates = sorted(
            [d for d in input_path.iterdir() if d.is_dir() and d.name.isdigit()],
            key=lambda d: int(d.name),
        )
    else:
        folder_candidates = [input_path / f for f in folders]

    for folder in folder_candidates:
        if not folder.exists():
            logger.warning(f"文件夹不存在: {folder}")
            continue

        # 查找 JSON 文件
        json_files = list(folder.glob("*.json"))
        if not json_files:
            logger.warning(f"文件夹 {folder.name} 中没有 JSON 文件")
            continue

        json_path = json_files[0]  # 取第一个 JSON 文件
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"读取 {json_path} 失败: {e}")
            continue

        survey_text = data.get("survey", "")
        references = data.get("references", {})

        # 从 survey markdown 的第一个 H1 标题提取 survey title
        title_match = re.search(r'^#\s+(.+)$', survey_text, re.MULTILINE)
        survey_title = title_match.group(1).strip() if title_match else json_path.stem

        surveys.append({
            "folder_id": folder.name,
            "survey_title": survey_title,
            "survey_text": survey_text,
            "references": references,
            "json_path": str(json_path),
        })

    return surveys


def build_cited_papers_info_autosurvey(
    cite_ids: List[str],
    references: Dict[str, str],
    paper_info_map: Dict[str, Dict[str, str]],
) -> str:
    """构建引用论文的信息文本（Autosurvey 版）"""
    papers_info_parts = []
    for cite_id in cite_ids:
        arxiv_id = references.get(cite_id)
        if not arxiv_id:
            papers_info_parts.append(
                f"**Paper [{cite_id}]**:\n  - [Reference ID not found in references mapping]"
            )
            continue

        info = paper_info_map.get(arxiv_id, {})
        title = info.get("Title", "")
        abstract = info.get("Abstract", "")

        if title or abstract:
            abstract_display = abstract[:500] + "..." if len(abstract) > 500 else abstract
            papers_info_parts.append(
                f"**Paper [{cite_id}]** (arXiv: {arxiv_id}):\n"
                f"  - Title: {title}\n"
                f"  - Abstract: {abstract_display}"
            )
        else:
            papers_info_parts.append(
                f"**Paper [{cite_id}]** (arXiv: {arxiv_id}):\n"
                f"  - [Paper info not available]"
            )

    return "\n\n".join(papers_info_parts)


# ============================================================
# 4. Main Evaluation Logic
# ============================================================

async def evaluate_autosurvey_section(
    llm: LLMClient,
    survey_title: str,
    section: Dict[str, Any],
    references: Dict[str, str],
    paper_info_map: Dict[str, Dict[str, str]],
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    """评估 Autosurvey 中的单个章节"""
    async with semaphore:
        section_title = section.get("title", "")
        section_path = section.get("section_path", "")
        section_content = section.get("content", "")
        cite_ids = section.get("cites", [])

        cited_papers_info = build_cited_papers_info_autosurvey(
            cite_ids, references, paper_info_map
        )

        # 截断过长内容
        if len(section_content) > 8000:
            section_content = section_content[:8000] + "\n... [content truncated]"

        # Autosurvey 没有独立 abstract，使用空字符串
        user_prompt = INSIGHT_EVAL_USER_PROMPT.format(
            survey_title=survey_title,
            survey_abstract="(Abstract not available for baseline-generated survey)",
            section_path=section_path,
            section_title=section_title,
            section_content=section_content,
            cited_papers_info=cited_papers_info,
        )

        try:
            logger.info(
                f"  评估章节: [H{section.get('level', '')}] {section_title} "
                f"(cites: {len(cite_ids)} 篇)"
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

        result = dict(section)
        result["insight_result"] = insight_result
        return result


async def evaluate_one_survey(
    llm: LLMClient,
    survey: Dict[str, Any],
    paper_info_map: Dict[str, Dict[str, str]],
    output_dir: Path,
    semaphore: asyncio.Semaphore,
) -> None:
    """评估单个 Autosurvey survey 的所有有引用章节"""
    folder_id = survey["folder_id"]
    survey_title = survey["survey_title"]
    references = survey["references"]
    output_path = output_dir / f"survey_{folder_id}.jsonl"

    # 断点续传
    if output_path.exists():
        logger.info(f"⏭️  跳过 survey_{folder_id}: {survey_title} (已有结果)")
        return

    # 解析 markdown sections
    sections = parse_markdown_sections(survey["survey_text"])
    sections_with_cites = [s for s in sections if s.get("cites")]

    if not sections_with_cites:
        logger.info(f"⏭️  跳过 survey_{folder_id}: {survey_title} (无引用章节)")
        return

    logger.info(
        f"📝 评估 survey_{folder_id}: {survey_title} "
        f"(共 {len(sections)} 个章节, {len(sections_with_cites)} 个有引用章节)"
    )

    # 并发评估各章节
    tasks = [
        evaluate_autosurvey_section(
            llm, survey_title, section, references, paper_info_map, semaphore
        )
        for section in sections_with_cites
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 写入 JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"  章节评估异常: {result}")
                continue
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    logger.info(
        f"✅ survey_{folder_id} 评估完成，"
        f"结果已保存至 {output_path} ({len(results)} 条)"
    )


async def main(args):
    """主流程"""
    start_time = time.time()

    # 1. 加载 Autosurvey 数据
    logger.info("=" * 60)
    logger.info("📂 加载 Autosurvey 数据...")
    folders = args.folders if args.folders else None
    surveys = load_autosurvey_folders(args.input_dir, folders)
    logger.info(f"   共加载 {len(surveys)} 篇 Autosurvey survey")

    if not surveys:
        logger.error("未找到任何 survey 数据，请检查 input_dir 路径")
        return

    # 2. 收集所有唯一的 arXiv ID 并获取论文信息
    logger.info("📚 获取引用论文信息 (arXiv API)...")
    cache = PaperInfoCache(args.cache_path)

    # 合并所有 references
    all_references: Dict[str, str] = {}
    for survey in surveys:
        for cite_num, arxiv_id in survey["references"].items():
            all_references[cite_num] = arxiv_id

    unique_arxiv_ids = set(all_references.values())
    logger.info(f"   共 {len(unique_arxiv_ids)} 个唯一 arXiv ID")

    # 构建 arxiv_id -> info 的映射
    # 为了复用，这里对每个 survey 的 references 进行批量获取
    all_paper_info: Dict[str, Dict[str, str]] = {}
    for survey in surveys:
        paper_info_map = await fetch_papers_for_references(
            survey["references"], cache, delay=args.api_delay
        )
        all_paper_info.update(paper_info_map)

    cached_count = sum(1 for aid in unique_arxiv_ids if cache.get(aid) is not None)
    logger.info(f"   论文信息获取完成: {cached_count}/{len(unique_arxiv_ids)} 已缓存")

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
    total_sections = 0
    for survey in surveys:
        sections = parse_markdown_sections(survey["survey_text"])
        total_sections += len([s for s in sections if s.get("cites")])
    logger.info(f"📊 待评估: {len(surveys)} 篇 survey, {total_sections} 个有引用章节")
    logger.info("=" * 60)

    # 6. 并发控制
    semaphore = asyncio.Semaphore(args.concurrency)

    # 7. 逐个评估
    for i, survey in enumerate(surveys):
        logger.info(f"\n--- [{i+1}/{len(surveys)}] ---")
        await evaluate_one_survey(llm, survey, all_paper_info, output_dir, semaphore)

    elapsed = time.time() - start_time
    logger.info(f"\n🎉 全部评估完成! 耗时: {elapsed:.1f}s")


# ============================================================
# 5. CLI Entry Point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Autosurvey Baseline Evaluation - 使用 LLM 评估 Autosurvey 生成的 survey 章节洞察力"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="benckmark/SurGE/baselines/Autosurvey/output",
        help="Autosurvey 输出目录路径",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="benckmark/results/autosurvey",
        help="评估结果输出目录",
    )
    parser.add_argument(
        "--cache_path",
        type=str,
        default="benckmark/paper_info_cache.json",
        help="arXiv 论文信息缓存路径",
    )
    parser.add_argument(
        "--folders",
        nargs="*",
        default=None,
        help="指定要评估的文件夹编号 (如 0 1 2)，不指定则评估全部",
    )
    parser.add_argument(
        "--api_delay",
        type=float,
        default=1.0,
        help="arXiv API 调用间隔时间/秒 (默认: 1.0)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="LLM 并发请求数 (默认: 3)",
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



# python benckmark/evaluate_autosurvey.py --folders