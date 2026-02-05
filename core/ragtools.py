import os
import sys
import pickle
import re
from typing import List, Dict, Optional
import torch

# 引入核心库
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from fastbm25 import fastbm25
from sentence_transformers import CrossEncoder

# 🔥🔥🔥 新增：导入统一配置文件 🔥🔥🔥
try:
    import config
except ImportError:
    # 兜底：如果找不到config，尝试把当前目录加入path再导入
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import config

class MoutaiRAGEngine:
    """
    RAG 核心引擎类：负责加载模型、索引和执行搜索。
    """

    def __init__(self):
        # 🔥 修改点 1：直接从 config 读取路径，不再手动拼接
        self.index_dir = config.FAISS_INDEX_PATH
        self.docs_path = config.DOCS_INFO_PATH

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🚀 [Engine] 初始化中... 设备: {self.device}")

        self.embeddings = None
        self.reranker = None
        self.vector_store = None
        self.bm25 = None
        self.all_documents = []
        self.is_ready = False

        self._load_resources()

    def _load_resources(self):
        try:
            # 1. 加载 Embedding
            print("📦 Loading Embeddings (BGE-Large)...")
            self.embeddings = HuggingFaceBgeEmbeddings(
                model_name="BAAI/bge-large-zh-v1.5",
                model_kwargs={'device': self.device},
                encode_kwargs={'normalize_embeddings': True}
            )

            # 2. 加载 Reranker
            print("📦 Loading Reranker (BGE-Reranker)...")
            self.reranker = CrossEncoder(model_name_or_path="BAAI/bge-reranker-large", device=self.device)

            # 3. 检查并加载索引
            print(f"📂 Loading Indices from: {self.index_dir}")
            if not os.path.exists(self.index_dir):
                raise FileNotFoundError(f"找不到索引目录: {self.index_dir}\n请先运行 build_rag_index.py 构建索引！")

            self.vector_store = FAISS.load_local(
                self.index_dir,
                self.embeddings,
                allow_dangerous_deserialization=True
            )

            # 4. 加载原始文档 docs.pkl
            if not os.path.exists(self.docs_path):
                raise FileNotFoundError(f"关键文件缺失: {self.docs_path}")

            print(f"📂 Loading Documents from: {self.docs_path}")
            with open(self.docs_path, 'rb') as f:
                self.all_documents = pickle.load(f)

            # 5. 重建 BM25
            print("🔄 Rebuilding BM25 Index...")
            corpus_texts = [doc.page_content for doc in self.all_documents]
            self.bm25 = fastbm25(corpus_texts)

            self.is_ready = True
            print(f"✅ RAG 引擎加载完成! 包含 {len(self.all_documents)} 条数据。")

        except Exception as e:
            print(f"❌ RAG 引擎加载失败: {e}")
            # print("💡 提示: 如果是第一次运行，请确保 data/ 目录下有 docs.pkl 和 faiss_index 文件夹")
            self.is_ready = False

    def _clean_text(self, text: str) -> str:
        """
        🧹 文本清洗器：修复 PDF 解析问题，但保护财务数字格式
        """
        if not text:
            return ""

        # 1. 先保护财务数字格式 (如 "14,769,360.50" 或 "1,234.56")
        protected_numbers = []

        def protect_num(match):
            protected_numbers.append(match.group(0))
            return f"__NUM_{len(protected_numbers) - 1}__"

        text = re.sub(r'\d{1,3}(,\d{3})+(\.\d+)?', protect_num, text)

        # 2. 把所有换行符替换成空格
        text = text.replace('\n', ' ').replace('\r', ' ')

        # 3. 修复被空格打断的中文词汇
        text = re.sub(r'([\u4e00-\u9fa5])\s+([\u4e00-\u9fa5])', r'\1\2', text)

        # 4. 合并空格
        text = re.sub(r'\s+', ' ', text)

        # 5. 恢复保护的数字
        for i, num in enumerate(protected_numbers):
            text = text.replace(f"__NUM_{i}__", num)

        return text.strip()

    def _extract_financial_highlights(self, text: str) -> List[str]:
        """🔍 提取财务数据亮点"""
        highlights = []
        patterns = [
            r'(营业收入|营收|利润总额|净利润|发生额).*?([\d,]+\.?\d*)\s*(万元|亿元|元)',
            r'([\d,]+\.?\d*)\s*(万元|亿元|元).*?(营业收入|营收|利润总额|净利润|发生额)',
            r'(?:增长|下降|同比|较上年).*?(\d+\.?\d*)%\s*(?:增长|下降|同比)?',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                clean_match = ''.join([m for m in match if m]).strip()
                if clean_match and clean_match not in highlights:
                    highlights.append(clean_match)

        return highlights[:3]

    def _rrf_fusion(self, list1, list2, k=60):
        """RRF 融合算法"""
        fusion_scores = {}
        for rank, (content, _) in enumerate(list1):
            if content not in fusion_scores: fusion_scores[content] = 0
            fusion_scores[content] += 1 / (rank + k)
        for rank, (content, _) in enumerate(list2):
            if content not in fusion_scores: fusion_scores[content] = 0
            fusion_scores[content] += 1 / (rank + k)
        return sorted(fusion_scores.items(), key=lambda x: x[1], reverse=True)

    def search(self, query: str, top_k: int = None) -> str:
        """执行混合检索 + 重排序"""
        if not self.is_ready:
            return "❌ 错误: 本地财务报表引擎未就绪。"
        
        # 🔥 修改点 2：默认使用 config 中的 Top K
        if top_k is None:
            top_k = config.RAG_TOP_K if hasattr(config, 'RAG_TOP_K') else 50

        try:
            # Step 1: BM25 + Vector 检索 (粗排)
            bm25_res = self.bm25.top_k_sentence(query, k=top_k)
            bm25_list = [(res[0], res[2]) for res in bm25_res]
            vector_res = self.vector_store.similarity_search_with_score(query, k=top_k)
            vector_list = [(doc.page_content, score) for doc, score in vector_res]

            # Step 2: RRF 融合
            fusion_results = self._rrf_fusion(bm25_list, vector_list)
            candidate_texts = [item[0] for item in fusion_results[:top_k]]

            if not candidate_texts:
                return "未找到相关本地财务信息。"

            # Step 3: Rerank 重排序 (精排)
            rerank_pairs = [[query, text] for text in candidate_texts]
            scores = self.reranker.predict(rerank_pairs, batch_size=4, show_progress_bar=False)

            scored_results = sorted(zip(candidate_texts, scores), key=lambda x: x[1], reverse=True)

            # Step 4: 组装上下文
            content_map = {doc.page_content: doc.metadata for doc in self.all_documents}
            final_output = ["以下是从茅台历史年报中检索到的相关内容："]

            # 只取前 5 个最相关的结果
            for i, (text, score) in enumerate(scored_results[:5]):
                meta = content_map.get(text, {})
                source = meta.get('source', '未知年报')
                cleaned_text = self._clean_text(text)
                highlights = self._extract_financial_highlights(cleaned_text)
                highlight_str = f" [关键数据: {' | '.join(highlights)}]" if highlights else ""

                final_output.append(f"资料[{i + 1}] 来源: {source}{highlight_str}\n内容: {cleaned_text}\n")

            return "\n".join(final_output)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"检索过程出错: {str(e)}"