import json
from typing import Optional, Dict, Any
from core.skill_loader import SkillLoader

class FinancialQueryRewriter:
    def __init__(self):
        # 1. 初始化统一加载器 (它会自动从 config.py 读取 Key)
        self.loader = SkillLoader()

    async def rewrite(self, user_query: str, user_profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        执行语义改写
        :param user_query: 用户原始输入
        :param user_profile: 用户画像 (可选，例如 {"risk": "保守型"})
        :return: 改写后的 JSON 结构
        """
        
        # 如果没有传入画像，给一个默认空字典，或者保留你代码里的默认值
        if user_profile is None:
            user_profile = {"risk": "保守型", "investment_experience": "3年"}

        # 2. 准备 Prompt 变量
        # 这里的 key 必须对应 semantic_rewrite.md 里的 {query}
        # 如果你的 md 里没有 {user_profile}，传入了也没关系，SkillLoader 会忽略多余的
        inputs = {
            "query": user_query,
            "user_profile": json.dumps(user_profile, ensure_ascii=False)
        }

        # 3. 调用 Skill (注意后缀是 .md)
        print(f"🔄 [Rewriter] 正在改写: {user_query}")
        result = await self.loader.execute_skill("semantic_rewrite.md", inputs)
        
        # 简单的错误处理
        if "error" in result:
            print(f"⚠️ 改写失败: {result['error']}")
            return {}

        return result