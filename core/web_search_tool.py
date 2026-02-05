import os
from tavily import TavilyClient
from typing import Optional


class SimpleWebSearchTool:
    """
    基于 Tavily 的深度网络搜索工具。
    """

    def __init__(self, api_key: Optional[str] = None):
        # 优先使用传入的 Key，否则尝试从环境变量读取
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")

        if not self.api_key:
            raise ValueError("❌ 未找到 Tavily API Key。请传入 api_key 参数或设置环境变量 'TAVILY_API_KEY'")

        self.client = TavilyClient(api_key=self.api_key)

    async def search(self, query: str) -> str:
        """
        执行异步搜索并返回格式化后的上下文。
        """
        print(f"      📡 [Tavily] 深度搜索中: {query}...")
        try:
            # search_depth="advanced" 会进行深度抓取，这对找具体数字非常关键
            # include_answer=True 让 Tavily 尝试直接生成简短答案
            response = self.client.search(
                query=query,
                search_depth="advanced",
                max_results=3,
                include_answer=True
            )

            # 拼接上下文
            context = []

            # 1. 如果有直接生成的答案，先加上
            if response.get("answer"):
                context.append(f"【AI总结答案】: {response['answer']}")

            # 2. 拼接搜索结果内容
            for result in response.get("results", []):
                content = result.get("content", "")
                url = result.get("url", "未知链接")

                # 过滤掉太短的无意义内容
                if len(content) > 50:
                    context.append(f"来源: {url}\n内容: {content}")

            if not context:
                return "未找到相关且有价值的信息。"

            return "\n\n".join(context)

        except Exception as e:
            error_msg = f"❌ Tavily Search Error: {str(e)}"
            print(error_msg)
            return "网络搜索失败，请稍后重试。"


# ==================== 测试代码 ====================
if __name__ == "__main__":
    import asyncio


    # 简单的本地测试
    async def test():
        # 这里替换你的 Key 用于测试，或者设置环境变量
        tool = SimpleWebSearchTool(api_key="tvly-dev-YOUR_KEY_HERE")
        result = await tool.search("2024年茅台集团的营收是多少？")
        print("\n🔎 搜索结果:\n", result)


    asyncio.run(test())