import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from core.skill_loader import SkillLoader

# ==================== 数据结构 ====================
# 这里保留简单的数据结构，如果多个文件共用，建议提到 core/types.py (可选)
@dataclass
class CheckResult:
    verdict: str
    score: float
    missing: List[Dict]
    caveats: str

@dataclass
class SummarizerResult:
    status: str
    report: Optional[Dict] = None
    caveats: Optional[str] = None
    missing: Optional[List[Dict]] = None


# ==================== Agent 类 ====================
class SummarizerAgent:
    def __init__(self, strict_mode: bool = False):
        """
        初始化总结者 Agent
        :param strict_mode: 严格模式开关 (True=任何缺失都报错, False=允许部分缺失)
        """
        # 1. 使用统一加载器 (自动读取 config)
        self.loader = SkillLoader()
        self.strict_mode = strict_mode
        
        # 技能文件名
        self.skill_check = "summarizer_check.md"
        self.skill_synth = "summarizer_synthesis.md"

    async def execute(self, context_data: Dict[str, Any]) -> SummarizerResult:
        """
        执行总结任务
        :param context_data: 包含 query, plan, collected_data 的字典
        """
        print(f"\n[Summarizer] 🤖 开始总结 (模式: {'Strict' if self.strict_mode else 'Loose'})...")

        # --- Step 1: 质量审查 ---
        check_res = await self._run_check(context_data)
        print(f"   📊 评分: {check_res.score} ({check_res.verdict})")

        # --- Step 2: 决策逻辑 ---
        should_rework = False
        final_caveats = "无"

        if self.strict_mode:
            # 严格模式：只要不是 sufficient 就返工
            if check_res.verdict != "sufficient":
                should_rework = True
        else:
            # 宽松模式：只有 insufficient (严重不足) 才返工
            if check_res.verdict == "insufficient":
                should_rework = True
            elif check_res.verdict == "partial":
                final_caveats = check_res.caveats or "部分数据缺失，结果仅供参考"
                print(f"   ⚠️ 触发宽松放行，附加声明: {final_caveats}")

        # --- Step 3: 执行动作 ---
        if should_rework:
            print(f"   🛑 决定: 请求返工 (Missing: {len(check_res.missing)} items)")
            return SummarizerResult(
                status="fail",
                missing=check_res.missing
            )
        else:
            print(f"   ✅ 决定: 生成报告")
            report = await self._run_synthesis(context_data, final_caveats)
            return SummarizerResult(
                status="success",
                report=report,
                caveats=final_caveats
            )

    async def _run_check(self, context_data: Dict) -> CheckResult:
        """调用审查 Skill"""
        # 准备数据，注意字段名要对应 summarizer_check.md
        inputs = {
            "required_info": json.dumps(context_data.get("plan", {}).get("required_info", []), ensure_ascii=False),
            "collected_data": json.dumps(context_data.get("collected_data", {}), ensure_ascii=False)
        }

        # 执行 Skill
        res = await self.loader.execute_skill(self.skill_check, inputs)

        return CheckResult(
            verdict=res.get("sufficiency_verdict", "insufficient"),
            score=float(res.get("sufficiency_score", 0.0)),
            missing=res.get("missing_critical_items", []),
            caveats=res.get("caveats")
        )

    async def _run_synthesis(self, context_data: Dict, caveats: str) -> Dict:
        """调用撰写 Skill"""
        # 准备数据，对应 summarizer_synthesis.md
        inputs = {
            "user_query": context_data.get("user_query", ""),
            "caveats": caveats,
            # 将字典转为 JSON 字符串，防止 Prompt 格式乱掉
            "validated_data": json.dumps(context_data.get("collected_data", {}), ensure_ascii=False)
        }

        # 执行 Skill
        report_json = await self.loader.execute_skill(self.skill_synth, inputs)
        
        return report_json