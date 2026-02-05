import asyncio
import concurrent.futures
from typing import Dict, Any, List
import config
# ==================== 1. 导入两个核心工具 ====================
from core.skill_loader import SkillLoader

# 工具一：Web 搜索工具 (来自 core/tools.py)
from core.web_search_tool import SimpleWebSearchTool 

# 工具二：RAG 引擎工具 (来自 ragtools.py)
# 假设 ragtools.py 在项目根目录，如果移到了 core，请改为 from core.rag_engine import MoutaiRAGEngine

from core.ragtools import MoutaiRAGEngine


class InformationCollectionAgent:
    def __init__(self, enable_parallel: bool = True):
        # 初始化统一加载器
        self.loader = SkillLoader()
        
        # ========== 初始化两大工具 ==========
        
        # 1. Web 工具 (轻量级，直接初始化)
        self.web = SimpleWebSearchTool(api_key=config.TAVILY_API_KEY)
        
        # 2. RAG 工具 (重量级，先占位，稍后异步加载)
        # 我们不在这里直接 MoutaiRAGEngine()，因为加载模型会卡住主线程好几秒
        self.rag_engine = None 
        self.rag_ready = False
        
        # 线程池：专门用来跑 RAG 这种重活
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    async def _ensure_rag_loaded(self):
        """
        [工具二激活]：异步加载 RAG 引擎
        """
        if self.rag_ready: return True

        print("⏳ [Collector] 正在后台启动 RAG 引擎 (Loading Models)...")
        loop = asyncio.get_event_loop()

        try:
            # 把 RAG 的初始化放到线程池里去跑，不卡顿
            self.rag_engine = await loop.run_in_executor(
                self._executor,
                lambda: MoutaiRAGEngine()
            )
            
            if self.rag_engine and getattr(self.rag_engine, 'is_ready', True):
                self.rag_ready = True
                print("✅ [Collector] RAG 引擎就绪!")
                return True
        except Exception as e:
            print(f"❌ [Collector] RAG 启动失败: {e}")
            return False

    async def execute(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """执行搜集任务"""
        # 确保工具二 (RAG) 已就绪
        await self._ensure_rag_loaded()
        
        results = {}
        tasks = plan.get("required_info", [])
        
        print(f"\n🚀 [Collector] 执行 {len(tasks)} 个搜集任务...")

        for req in tasks:
            desc = req["desc"]
            source_pref = req.get("source", "rag") 
            print(f"\n👉 [Task] {desc}")

            rag_data = ""
            is_valid = False

            # ==================== 使用工具二: RAG ====================
            if "rag" in source_pref and self.rag_ready:
                print(f"   🔍 [RAG] 检索本地知识库...")
                loop = asyncio.get_event_loop()
                # 在线程池中调用 rag_engine.search
                rag_data = await loop.run_in_executor(
                    self._executor, 
                    lambda: self.rag_engine.search(desc, top_k=50)
                )

                # 评估数据质量
                if rag_data and "❌" not in rag_data and "未找到" not in rag_data:
                    eval_res = await self.loader.execute_skill("info_evaluator.md", {
                        "query": desc, 
                        "content": rag_data
                    })
                    
                    if eval_res.get("is_sufficient"):
                        print(f"   ✅ [RAG] 命中有效数据")
                        results[desc] = {"data": rag_data, "source": "RAG"}
                        is_valid = True
                    else:
                        print(f"   ⚠️ [RAG] 数据无效: {eval_res.get('reason')}")
                else:
                    print(f"   💨 [RAG] 未找到相关信息")

            # ==================== 使用工具一: Web ====================
            if not is_valid:
                print(f"   🌐 [Web] 启动联网搜索...")
                # 直接调用 web.search
                web_data = await self.web.search(desc)
                
                if web_data:
                    results[desc] = {"data": web_data, "source": "Web"}
                else:
                    results[desc] = {"data": "未找到", "source": "Failed"}

        return {"validated_data": results}