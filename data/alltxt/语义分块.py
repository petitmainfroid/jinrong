import os
import re
import json
import itertools
from loguru import logger
from fastbm25 import fastbm25
from typing import List, Dict, Tuple, Optional
# LangChain 核心组件
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 向量数据库与模型
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Reranker (使用 sentence_transformers 原生调用，比 LangChain 封装更灵活)
from sentence_transformers import CrossEncoder

# 本地配置
import config as cfg


class AnnualReportChunker:
    """年报结构化分块器"""

    def __init__(self):
        # 年报常见章节标题模式（按优先级排序）
        self.section_patterns = [
            # 第X节 格式（最明确）
            r'^第[一二三四五六七八九十百]+节\s+\S.*$',
            r'^第\d+\s*节\s+\S.*$',

            # Markdown标题格式
            r'^#+\s+\S.*$',  # # 标题

            # 一级标题：中文数字 + 、
            r'^[一二三四五六七八九十百]+[、．.]\s+\S.*$',  # 确保标题后有内容

            # 目录/重要提示/释义（单独成行）
            r'^重要提示\s*$',
            r'^目录\s*$',
            r'^释义\s*$',

            # 常见章节名称（完整匹配）
            r'^公司简介\s*$',
            r'^会计数据\s*$',
            r'^财务报告\s*$',
            r'^董事会报告\s*$',
            r'^监事会报告\s*$',
            r'^重要事项\s*$',
            r'^股本变动\s*$',
            r'^股东信息\s*$',
            r'^公司债券\s*$',
            r'^财务报表\s*$',
        ]

        # 子章节标题模式（用于识别小节）
        self.subsection_patterns = [
            r'^#+\s+\S.*$',  # Markdown标题
            r'^[（(][一二三四五六七八九十]+[)）]\s*.+',
            r'^[（(]\d+[)）]\s*.+',
            r'^[①②③④⑤⑥⑦⑧⑨⑩]\s*.+',
        ]

    def is_section_title(self, line: str) -> Tuple[bool, str]:
        """
        判断一行是否是章节标题
        返回: (是否是标题, 标题级别: 'main'/'sub'/'none')
        """
        line = line.strip()

        # 检查是否是一级章节标题
        for pattern in self.section_patterns:
            if re.match(pattern, line):
                return True, 'main'

        # 检查是否是子章节标题
        for pattern in self.subsection_patterns:
            if re.match(pattern, line):
                return True, 'sub'

        return False, 'none'

    def clean_line(self, line: str) -> str:
        """清理行内容"""
        line = line.strip()
        # 移除多余空格
        line = re.sub(r'\s+', ' ', line)
        return line

    def is_markdown_metadata(self, line: str) -> bool:
        """判断是否是Markdown元数据"""
        line = line.strip()
        return line.startswith('---') or line.startswith('```') or line.startswith('|--')

    def extract_sections_from_text(self, text: str) -> List[Dict]:
        """
        从纯文本中提取章节
        返回: [{'title': 章节标题, 'level': 级别, 'content': 内容, 'lines': 行号列表}]
        """
        lines = text.split('\n')
        sections = []
        current_section = None
        line_number = 0

        for line in lines:
            line_number += 1
            line = self.clean_line(line)

            # 跳过空行和Markdown元数据
            if not line or self.is_markdown_metadata(line):
                continue

            # 检查是否是章节标题
            is_title, level = self.is_section_title(line)

            if is_title:
                # 保存上一个章节
                if current_section:
                    sections.append(current_section)

                # 创建新章节
                current_section = {
                    'title': line,
                    'level': level,
                    'content': '',
                    'lines': [line_number],
                    'char_start': len(text[:text.index(line)]) if line in text else 0
                }
            else:
                # 添加到当前章节
                if current_section:
                    current_section['content'] += line + '\n'
                    current_section['lines'].append(line_number)
                else:
                    # 文档开头的内容（在第一个章节之前的）
                    current_section = {
                        'title': '文档开头',
                        'level': 'main',
                        'content': line + '\n',
                        'lines': [line_number],
                        'char_start': 0
                    }

        # 保存最后一个章节
        if current_section:
            sections.append(current_section)

        return sections

    def merge_small_sections(self, sections: List[Dict], min_chars: int = 200) -> List[Dict]:
        """
        合并过小的章节到相邻章节
        min_chars: 最小字符数，小于此值的章节会被合并
        """
        if not sections:
            return sections

        merged = []
        i = 0

        while i < len(sections):
            current = sections[i]

            # 如果当前章节太小且不是第一个，尝试合并到前一个章节
            if len(current['content']) < min_chars and merged:
                merged[-1]['content'] += '\n\n' + current['content']
                merged[-1]['title'] += ' + ' + current['title']
                merged[-1]['lines'].extend(current['lines'])
            else:
                merged.append(current)

            i += 1

        return merged

    def chunk_by_sections(
            self,
            text: str,
            min_chars: int = 100,
            max_chars: int = 3000,
            merge_small: bool = True
    ) -> List[Dict]:
        """
        按章节进行分块

        参数:
            text: 输入文本
            min_chars: 最小字符数（用于合并小章节）
            max_chars: 最大字符数（超过此大小的章节会进一步分割）
            merge_small: 是否合并小章节

        返回:
            章节块列表
        """
        # 提取章节
        sections = self.extract_sections_from_text(text)

        if merge_small:
            sections = self.merge_small_sections(sections, min_chars)

        # 对过大的章节进行分割
        final_chunks = []
        for section in sections:
            if len(section['content']) <= max_chars:
                final_chunks.append(section)
            else:
                # 分割大章节
                sub_chunks = self.split_large_section(section, max_chars)
                final_chunks.extend(sub_chunks)

        return final_chunks

    def split_large_section(self, section: Dict, max_chars: int) -> List[Dict]:
        """
        将过大的章节分割成多个块
        策略：
        1. 先按子章节分割（如果有）
        2. 再按段落分割
        3. 避免截断表格（检测连续的短行）
        """
        content = section['content']
        title = section['title']
        level = section['level']

        # 检查是否有Markdown标题作为子章节
        lines = content.split('\n')
        has_subheadings = any(re.match(r'^#+\s+', line.strip()) for line in lines)

        if has_subheadings:
            return self._split_by_markdown_headings(section, max_chars)

        # 没有子章节，按智能段落分割
        return self._split_by_smart_paragraphs(title, content, level, max_chars)

    def _split_by_markdown_headings(self, section: Dict, max_chars: int) -> List[Dict]:
        """按Markdown标题分割大章节"""
        content = section['content']
        title = section['title']
        level = section['level']

        chunks = []
        current_content = ''
        current_heading = title
        chunk_num = 1

        lines = content.split('\n')
        for line in lines:
            # 检查是否是Markdown标题
            if re.match(r'^#+\s+', line.strip()):
                # 保存当前块
                if current_content.strip():
                    chunks.append({
                        'title': current_heading,
                        'content': current_content.strip(),
                        'level': level
                    })
                    chunk_num += 1

                # 开始新块
                current_heading = f"{title} - {line.strip()}"
                current_content = line + '\n'
            else:
                current_content += line + '\n'

        # 保存最后一块
        if current_content.strip():
            chunks.append({
                'title': current_heading,
                'content': current_content.strip(),
                'level': level
            })

        return chunks

    def _split_by_smart_paragraphs(self, title: str, content: str, level: str, max_chars: int) -> List[Dict]:
        """
        智能按段落分割，避免截断表格
        表格特征：连续的短行（通常<50字符）
        """
        lines = content.split('\n')
        chunks = []
        current_chunk_lines = []
        current_size = 0
        chunk_num = 1

        i = 0
        while i < len(lines):
            line = lines[i]
            line_size = len(line)

            # 检测Markdown表格
            is_markdown_table = line.strip().startswith('|')

            # 检测是否是表格（连续短行或Markdown表格）
            is_table = False
            if i + 3 < len(lines):  # 至少4行
                next_lines_short = all(len(lines[j]) < 80 for j in range(i, min(i + 4, len(lines))))
                next_lines_table = all(lines[j].strip().startswith('|') for j in range(i, min(i + 2, len(lines))))
                is_table = next_lines_short or next_lines_table

            # 如果加上这一行会超限
            if current_size + line_size > max_chars and current_chunk_lines:
                # 如果是表格，尽量保持表格完整
                if is_table or is_markdown_table:
                    # 找到表格结束位置
                    table_end = i
                    while table_end < len(lines) and (
                            len(lines[table_end]) < 80 or lines[table_end].strip().startswith('|')):
                        table_end += 1

                    # 如果整个表格放得下，一起放入当前块
                    table_size = sum(len(lines[j]) for j in range(i, table_end))
                    if current_size + table_size <= max_chars * 1.2:  # 允许超限20%
                        # 一起放入
                        for j in range(i, table_end):
                            current_chunk_lines.append(lines[j])
                            current_size += len(lines[j])
                        i = table_end
                        continue

                # 保存当前块
                chunks.append({
                    'title': f"{title} ({chunk_num})" if chunk_num > 1 else title,
                    'content': '\n'.join(current_chunk_lines),
                    'level': level
                })
                chunk_num += 1
                current_chunk_lines = []
                current_size = 0

            # 添加到当前块
            current_chunk_lines.append(line)
            current_size += line_size
            i += 1

        # 保存最后一块
        if current_chunk_lines:
            chunks.append({
                'title': f"{title} ({chunk_num})" if chunk_num > 1 else title,
                'content': '\n'.join(current_chunk_lines),
                'level': level
            })

        return chunks

    def chunk_by_sections_with_sliding_window(
            self,
            text: str,
            section_max_chars: int = 2000,
            sliding_window_size: int = 1000,
            sliding_overlap: int = 200,
            merge_small: bool = True
    ) -> List[Dict]:
        """
        混合分块策略：先结构化分块，大章节使用滑窗

        参数:
            text: 输入文本
            section_max_chars: 章节最大字符数，超过则使用滑窗
            sliding_window_size: 滑窗大小
            sliding_overlap: 滑窗重叠大小
            merge_small: 是否合并小章节

        返回:
            分块列表
        """
        # 1. 先进行结构化分块
        sections = self.extract_sections_from_text(text)

        if merge_small:
            sections = self.merge_small_sections(sections, min_chars=100)

        # 2. 对每个章节判断是否需要滑窗
        final_chunks = []
        for section in sections:
            content_len = len(section['content'])

            if content_len <= section_max_chars:
                # 小章节，直接保留
                final_chunks.append(section)
            else:
                # 大章节，使用滑窗分块
                logger.info(f'章节 "{section["title"][:30]}..." 大小 {content_len} 字符，使用滑窗分块')

                sliding_chunks = self._sliding_window_by_char(
                    title=section['title'],
                    content=section['content'],
                    level=section['level'],
                    chunk_size=sliding_window_size,
                    overlap=sliding_overlap
                )

                final_chunks.extend(sliding_chunks)

        return final_chunks

    def _sliding_window_by_char(
            self,
            title: str,
            content: str,
            level: str,
            chunk_size: int,
            overlap: int
    ) -> List[Dict]:
        """
        按字符滑窗分块
        优先在句子/段落边界切分
        """
        chunks = []
        start = 0
        content_len = len(content)
        chunk_num = 1

        while start < content_len:
            # 计算窗口结束位置
            end = min(start + chunk_size, content_len)

            # 如果不是最后一块，尝试在句子边界切分
            if end < content_len:
                # 优先找段落边界（\n\n）
                paragraph_boundary = content.rfind('\n\n', start, end)
                if paragraph_boundary > start + chunk_size * 0.7:  # 至少保留70%
                    end = paragraph_boundary + 2
                else:
                    # 其次找句子边界（句号）
                    sentence_boundary = content.rfind('。', start, end)
                    if sentence_boundary > start + chunk_size * 0.7:
                        end = sentence_boundary + 1
                    else:
                        # 最后找换行
                        line_boundary = content.rfind('\n', start, end)
                        if line_boundary > start + chunk_size * 0.7:
                            end = line_boundary + 1

            # 提取窗口内容
            chunk_content = content[start:end].strip()

            if chunk_content:
                chunks.append({
                    'title': f"{title} (滑动{chunk_num})" if chunk_num > 1 else title,
                    'content': chunk_content,
                    'level': level,
                    'char_range': [start, end],
                    'overlap': overlap if chunk_num > 1 else 0
                })
                chunk_num += 1

            # 移动窗口（保留重叠）
            start = end - overlap if end < content_len else content_len

        return chunks


def load_md_file(file_path: str) -> List[str]:
    """
    加载Markdown文件

    参数:
        file_path: Markdown文件路径

    返回:
        内容列表，每个元素为一页（这里将整个文件作为一页处理）
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        logger.info(f"成功加载Markdown文件: {file_path} (大小: {len(content)} 字符)")
        return [content]  # 返回列表格式以保持接口一致性

    except FileNotFoundError:
        logger.error(f"文件未找到: {file_path}")
        return []
    except Exception as e:
        logger.error(f"读取Markdown文件失败: {e}")
        return []


def find_md_files(data_path: str) -> List[str]:
    """
    在指定目录下查找所有Markdown文件

    参数:
        data_path: 数据目录路径

    返回:
        Markdown文件路径列表
    """
    md_files = []

    # 检查是否为文件
    if os.path.isfile(data_path) and data_path.endswith('.md'):
        return [data_path]

    # 检查是否为目录
    if os.path.isdir(data_path):
        for file_name in os.listdir(data_path):
            if file_name.endswith('.md'):
                file_path = os.path.join(data_path, file_name)
                md_files.append(file_path)

    logger.info(f"在 {data_path} 中找到 {len(md_files)} 个Markdown文件")
    return md_files


def chunk_md_by_sections(file_path: str) -> List[Dict]:
    """
    对Markdown文件按章节进行分块

    参数:
        file_path: Markdown文件路径

    返回:
        章节块列表
    """
    chunker = AnnualReportChunker()
    pages = load_md_file(file_path)

    if not pages:
        logger.warning(f'未加载到页面内容: {file_path}')
        return []

    # 合并所有页面文本
    full_text = '\n\n'.join(pages)

    # 按章节分块
    chunks = chunker.chunk_by_sections(
        full_text,
        min_chars=100,
        max_chars=3000,
        merge_small=True
    )

    return chunks


def clean_text(text):
    """清理文本"""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def get_text_chunks_from_md(file_path: str, save_json: bool = True) -> List[str]:
    """
    从Markdown文件获取文本分块

    参数:
        file_path: Markdown文件路径
        save_json: 是否保存为JSON文件

    返回:
        格式化后的文本块列表
    """
    pages = load_md_file(file_path)
    if not pages:
        return []

    full_text = "\n\n".join(pages)
    chunker = AnnualReportChunker()

    # 执行结构化分块
    structured_chunks = chunker.chunk_by_sections(
        full_text,
        min_chars=200,
        max_chars=800,
        merge_small=True
    )

    # --- 保存 JSON 文件 ---
    if save_json:
        # 从文件路径生成JSON文件名
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        json_filename = f"{base_name}_chunks.json"

        # 保存到同目录
        output_dir = os.path.dirname(file_path) or '.'
        output_path = os.path.join(output_dir, json_filename)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(structured_chunks, f, ensure_ascii=False, indent=4)
        logger.info(f"分块 JSON 已保存至: {output_path}")

    # 格式转换供检索使用
    final_text_list = []
    for item in structured_chunks:
        title = item.get('title', '未知章节')
        content = item.get('content', '').strip()
        formatted_text = f"【章节：{title}】\n{content}"
        final_text_list.append(formatted_text)

    logger.info(f"结构化分块完成，共生成 {len(final_text_list)} 个切片")
    return final_text_list


# ================= 1. 核心算法: RRF 融合 =================
def rrf_fusion(list1, list2, k=60):
    """
    Reciprocal Rank Fusion (RRF) 算法
    将两路检索结果（BM25 和 向量）的排名进行融合
    """
    fusion_scores = {}

    # 建立内容到索引的映射，防止重复内容处理
    # list1, list2 格式: [(content, score), (content, score)...]

    # 处理第一路 (BM25)
    for rank, (content, _) in enumerate(list1):
        if content not in fusion_scores: fusion_scores[content] = 0
        fusion_scores[content] += 1 / (rank + k)

    # 处理第二路 (Vector)
    for rank, (content, _) in enumerate(list2):
        if content not in fusion_scores: fusion_scores[content] = 0
        fusion_scores[content] += 1 / (rank + k)

    # 按融合分数从高到低排序
    # 返回格式: [(content, fusion_score), ...]
    sorted_results = sorted(fusion_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results


def hybrid_search_md(
        query: str,
        md_file_path: str,
        embedding_model: HuggingFaceEmbeddings,
        reranker_model: CrossEncoder
):
    """
    对Markdown文件进行混合检索

    参数:
        query: 查询文本
        md_file_path: Markdown文件路径
        embedding_model: 嵌入模型
        reranker_model: 重排序模型

    返回:
        检索结果列表
    """
    import torch
    import gc

    # 0. 垃圾回收，腾出空间
    gc.collect()
    torch.cuda.empty_cache()

    logger.info(f"正在处理文件: {md_file_path}")

    # --- Step 1: 获取切片 ---
    chunks = get_text_chunks_from_md(md_file_path)
    if not chunks:
        logger.error("文件内容为空")
        return []

    print(f"文本共切分为 {len(chunks)} 个片段")

    # --- Step 2: BM25 召回 ---
    print(">>> [1/4] 正在执行 BM25 关键词检索...")
    bm25_model = fastbm25(chunks)
    bm25_res_raw = bm25_model.top_k_sentence(query, k=50)
    bm25_list = [(res[0], res[2]) for res in bm25_res_raw]

    # --- Step 3: Vector 召回 ---
    print(">>> [2/4] 正在构建向量索引并检索...")
    try:
        docs = [Document(page_content=t) for t in chunks]
        vector_store = FAISS.from_documents(docs, embedding_model)
        vector_res_raw = vector_store.similarity_search_with_score(query, k=50)
        vector_list = [(doc.page_content, score) for doc, score in vector_res_raw]
    except Exception as e:
        print(f"❌ 向量检索步骤出错: {e}")
        return []

    # --- Step 4: RRF 融合 ---
    print(">>> [3/4] 正在执行 RRF 融合...")
    fusion_results = rrf_fusion(bm25_list, vector_list)

    # 取前 50 个候选
    candidate_texts = [item[0] for item in fusion_results[:50]]
    if not candidate_texts:
        return []

    # --- Step 5: Rerank 重排序 ---
    print(">>> [4/4] 正在使用 BGE-Reranker 进行精排 (CPU模式)...")

    try:
        # 构造输入对
        rerank_pairs = [[query, doc_text] for doc_text in candidate_texts]

        # 🔥【关键修改】添加 batch_size=1 和进度条
        # CPU 计算能力有限，必须由 batch_size=1 来保证内存安全
        scores = reranker_model.predict(
            rerank_pairs,
            batch_size=1,  # 最安全的设置
            show_progress_bar=True,  # 显示进度条
            num_workers=0  # 防止多进程死锁
        )

        # 排序
        scored_results = list(zip(candidate_texts, scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)

        final_top_3 = scored_results[:3]

        formatted_results = []
        for i, (text, score) in enumerate(final_top_3):
            block = f"【Rank {i + 1} | Rerank得分: {score:.4f}】\n{text}"
            formatted_results.append(block)

        return formatted_results

    except Exception as e:
        print(f"❌ Rerank 阶段发生错误: {e}")
        # 打印详细堆栈以便调试
        import traceback
        traceback.print_exc()
        return []


def batch_process_md_files(data_path: str = None):
    """
    批量处理Markdown文件

    参数:
        data_path: 数据目录路径，默认为当前目录
    """
    # 设置数据路径
    if data_path is None:
        data_path = '.'  # 当前目录

    # 查找所有Markdown文件
    md_files = find_md_files(data_path)

    if not md_files:
        print(f"在 {data_path} 中未找到Markdown文件")
        return

    print(f"找到 {len(md_files)} 个Markdown文件:")
    for i, file_path in enumerate(md_files, 1):
        print(f"  {i}. {os.path.basename(file_path)}")

    # 加载模型
    print("\n" + "=" * 50)
    print("正在加载模型...")

    # 检查是否有GPU可用
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-zh-v1.5",
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True, 'batch_size': 8}
    )

    # 根据设备设置reranker
    reranker_device = 'cpu'  # 为了稳定性，reranker通常用CPU
    reranker = CrossEncoder(model_name="BAAI/bge-reranker-large", device=reranker_device)
    print("模型加载完成！")

    # 测试问题
    queries = [
        "公司营收",
        "年度营业收入、净利润是多少？",
        "利润分配预案",
        "产量、销量数据",
        "发展战略",
        "风险因素"
    ]

    # 处理每个文件
    for md_file in md_files:
        print("\n" + "★" * 30)
        print(f"处理文件: {os.path.basename(md_file)}")
        print("★" * 30 + "\n")

        # 先进行分块处理
        print(f"正在对 {os.path.basename(md_file)} 进行分块处理...")
        chunks = get_text_chunks_from_md(md_file, save_json=True)

        if not chunks:
            print(f"  ⚠️  {os.path.basename(md_file)} 分块失败，跳过")
            continue

        print(f"  生成 {len(chunks)} 个分块")

        # 执行检索测试
        for i, query in enumerate(queries[:3]):  # 只测试前3个问题
            print(f"\n[问题 {i + 1}] >>> {query}")

            results = hybrid_search_md(query, md_file, embeddings, reranker)

            print(f"\n--- 问题 {i + 1} 的 Top-3 检索结果 ---")
            if results:
                for res in results:
                    print(res)
                    print("-" * 30)
            else:
                print("未找到相关结果。")

            print("\n" + "=" * 60)


# ================= 主程序入口 =================
# ================= 优化后的主程序入口 =================

if __name__ == '__main__':
    # 1. 指定测试文件路径 (可以是单个文件，也可以是目录)
    test_md_path = './'  # 替换为你的实际文件名

    # 2. 定义你想提问的指定金融问题
    financial_queries = [
        "2023年度公司的营业收入、净利润及其同比增长率是多少？",
        "公司主要的经营风险有哪些？请列举至少三个。",
        "公司本年度的利润分配预案或股利分配政策是什么？",
        "研发投入占营业收入的比重是多少？",
        "前五大客户的销售额占比情况如何？"
    ]

    # 3. 初始化模型 (只加载一次，避免内存浪费)
    print("\n" + "=" * 20 + " 初始化模型 " + "=" * 20)
    import torch

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-zh-v1.5",
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )

    reranker = CrossEncoder(
        model_name="BAAI/bge-reranker-large",
        device='cpu'  # Reranker 在 CPU 上运行通常更稳定，若显存充足可改为 device
    )
    print(f"模型加载成功，使用设备: {device}")

    # 4. 执行召回与提问
    if os.path.exists(test_md_path):
        print(f"\n开始分析文件: {os.path.basename(test_md_path)}")

        for idx, query in enumerate(financial_queries):
            print(f"\n🔍 [问题 {idx + 1}] {query}")

            # 调用混合检索函数
            # 注意：该函数内部会自动完成：分块 -> BM25 -> Vector -> RRF -> Rerank
            results = hybrid_search_md(
                query=query,
                md_file_path=test_md_path,
                embedding_model=embeddings,
                reranker_model=reranker
            )

            # 5. 格式化输出召回的内容
            if results:
                print(f"✅ 召回成功，最相关的 Top-{len(results)} 个上下文片段如下：")
                for res in results:
                    print(res)
                    print("-" * 50)
            else:
                print("❌ 未能召回相关内容，请检查分块策略或文件格式。")
    else:
        print(f"错误：找不到文件 {test_md_path}")

    print("\n" + "=" * 20 + " 测试流程结束 " + "=" * 20)