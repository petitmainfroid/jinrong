import json
from dataclasses import dataclass, field
from typing import Optional, Dict
from core.skill_loader import SkillLoader

@dataclass
class ChaseResult:
    """追问结果的数据结构"""
    can_proceed: bool
    action: str
    question: Optional[str] = None
    options: list = field(default_factory=list)

class ChaserAgent:
    def __init__(self, max_chase_rounds=3):
        # 1. 初始化统一加载器 (自动读取 config.py)
        self.loader = SkillLoader()
        self.max_chase_rounds = max_chase_rounds

    async def check_and_chase(self, context_data: Dict) -> ChaseResult:
        """
        执行完整性审查
        :param context_data: 包含 original_query, rewritten_query, filled_slots 的字典
        """
        print(f"\n[Chaser] 🔍 完整性审查...")

        # 2. 准备 Prompt 变量 (对应 .md 文件中的占位符)
        input_vars = {
            "original_query": context_data.get("original_query", ""),
            # 注意：Leader 传进来时可能叫 rewritten_query 或 rewrite_result.step5_rewritten_query
            # 这里做个兼容处理，或者在 Leader 那边统一下
            "rewritten_query": context_data.get("rewritten_query", ""), 
            "intent": context_data.get("filled_slots", {}).get("intent", "unknown"),
            # 将字典转为 JSON 字符串，方便 Prompt 阅读
            "current_slots_json": json.dumps(context_data.get("filled_slots", {}), ensure_ascii=False)
        }

        # 3. 调用 Skill (使用 Markdown 格式)
        result = await self.loader.execute_skill("chaser_integrity_check.md", input_vars)

        # 4. 解析结果
        # LLM 返回的是 JSON，SkillLoader 已经帮我们 parse 好了
        if result.get("is_sufficient"):
            print(f"✅ [Chaser] 信息完整，放行")
            return ChaseResult(can_proceed=True, action="proceed")
        else:
            print(f"🛑 [Chaser] 信息缺失: {result.get('reason')}")
            return ChaseResult(
                can_proceed=False,
                action="chase",
                question=result.get("suggested_question"),
                options=result.get("suggested_options", [])
            )

    async def integrate_user_answer(self, old_context: Dict, user_answer: str) -> Dict:
        """
        处理用户的补充回答
        (简单版：直接拼接到 rewritten_query 后面，让 LLM 自己去理解)
        """
        # 在更复杂的版本中，这里应该再调一个 Skill (如 slot_filling.md) 来提取实体
        # 这里为了演示，采用"追加上下文"的方式
        
        print(f"🔄 [Chaser] 合并用户补充信息...")
        
        # 简单追加，这样再次改写或规划时，LLM 就能看到补充信息了
        # 注意：这里修改的是内存里的 context，不会改动原始 query
        old_context["rewritten_query"] += f" (补充说明: {user_answer})"
        
        # 也可以尝试直接塞进 slots 里（取决于你的下游 Planner 怎么用）
        # slots = old_context.get("filled_slots", {})
        # slots["user_supplement"] = user_answer
        # old_context["filled_slots"] = slots
        
        return old_context