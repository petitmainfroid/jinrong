import json
import asyncio
import os
import sys
from typing import Dict, List, Optional, TypedDict, Any
from enum import Enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from openai import AsyncOpenAI

# ==================== 0. 导入真实项目组件 ====================
# 确保能找到 core 和 agents 包
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from core.skill_loader import SkillLoader
from agents.rewriter import FinancialQueryRewriter
from agents.collector import InformationCollectionAgent
from agents.summarizer import SummarizerAgent
from agents.chaser import ChaserAgent

# ==================== 1. 基础类型定义 ====================

class ActionType(Enum):
    """动作空间 - Agent 可以执行的操作"""
    REWRITE = "rewrite"  # 改写查询
    PLAN = "plan"  # 制定计划
    CHASE = "chase"
    SEARCH_DB = "search_db"  # 查 RAG
    SEARCH_WEB = "search_web"  # 查网络
    SUMMARIZE = "summarize"  # 总结报告
    FINISH = "finish"  # 结束任务


@dataclass
class Action:
    """模型输出的动作决策"""
    type: ActionType
    parameters: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""  # 思维链 (CoT)


@dataclass
class Observation:
    """环境反馈"""
    success: bool
    data: Any
    cost: float = 0.0  # 用于 RL 奖励计算
    error_msg: Optional[str] = None


class AgentState(TypedDict):
    """全局状态 (用于模型决策的上下文)"""
    query: str
    context: Dict[str, Any]  # 累积的知识
    history: List[Dict[str, Any]]  # 动作历史
    step_count: int
    accumulated_reward: float


# ==================== 2. 真实 LLM 客户端 (用于决策) ====================

class OpenAILLM:
    """
    用于 Leader 进行决策的 LLM
    """

    def __init__(self, api_key: str, base_url: str, model_name: str):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    async def generate(self, prompt: str, temperature: float = 0.1) -> str:
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system",
                     "content": "你是一个专业的Agent决策模型。请根据上下文选择下一步动作，并严格输出合法的 JSON 格式。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"❌ 决策模型调用失败: {e}")
            return json.dumps({
                "action": "finish",
                "parameters": {},
                "reason": f"API Error: {str(e)}"
            })


# ==================== 3. 真实 Skill 包装层 ====================

class BaseSkill(ABC):
    def __init__(self, name: str): self.name = name

    @abstractmethod
    async def execute(self, params: Dict, state: AgentState) -> Observation: pass


class ChaseSkill(BaseSkill):
    """
    [真实] 调用 agents.chaser.ChaserAgent 进行完整性检查
    """

    def __init__(self):
        super().__init__("chaser")
        self.agent = ChaserAgent()

    async def execute(self, params: Dict, state: AgentState) -> Observation:
        print(f"   🔍 [Skill: Chase] 正在执行信息完整性审查...")

        ctx = state.get("context", {})

        # 构造 Chaser 需要的上下文
        # 对应 chaser.py 中 check_and_chase 的参数要求
        chaser_input = {
            "original_query": state["query"],
            "rewritten_query": ctx.get("rewritten_query", state["query"]),
            "filled_slots": {
                "intent": ctx.get("intent", {}).get("intent_name", "unknown"),
                "entities": ctx.get("entities", [])
            }
        }

        try:
            res = await self.agent.check_and_chase(chaser_input)

            if res.can_proceed:
                return Observation(
                    success=True,
                    data={"integrity_ok": True},
                    cost=0.01
                )
            else:
                # ✅ 修复：即使需要追问，success 也必须是 True！
                # 这样 _update_state 才会把 suggested_question 存入 context
                return Observation(
                    success=True,  # <--- 改成 True
                    data={
                        "integrity_ok": False,
                        "suggested_question": res.question,
                        "suggested_options": res.options,
                        "is_wait_user": True
                    },
                    cost=0.01
                )
        except Exception as e:
            # 只有程序崩溃报错时，才返回 False
            return Observation(success=False, data=None, error_msg=str(e))

class RewriteSkill(BaseSkill):
    """
    [真实] 调用 agents.rewriter.FinancialQueryRewriter
    """

    def __init__(self):
        super().__init__("rewriter")
        self.agent = FinancialQueryRewriter()

    async def execute(self, params: Dict, state: AgentState) -> Observation:
        print(f"   ⚙️ [Skill: Rewrite] 正在调用改写模型...")
        query = params.get("query", state["query"])

        try:
            # 真实调用
            res = await self.agent.rewrite(query)

            # 解析结果
            rewritten = res.get("step5_rewritten_query", query)
            entities = res.get("step2_entities", [])
            intent = res.get("step1_intent", {})

            # 返回 Observation
            return Observation(
                success=True,
                data={
                    "rewritten_query": rewritten,
                    "entities": entities,
                    "intent": intent
                },
                cost=0.01
            )
        except Exception as e:
            return Observation(success=False, data=None, error_msg=str(e))


class PlanningSkill(BaseSkill):
    """
    [真实] 调用 skills/leader_planning.md
    """

    def __init__(self):
        super().__init__("planner")
        self.loader = SkillLoader()

    async def execute(self, params: Dict, state: AgentState) -> Observation:
        print(f"   ⚙️ [Skill: Plan] 正在制定计划...")

        ctx = state.get("context", {})
        # 从上下文获取必要信息
        rewritten_query = ctx.get("rewritten_query", state["query"])
        entities_json = json.dumps(ctx.get("entities", []), ensure_ascii=False)

        inputs = {
            "rewritten_query": rewritten_query,
            "entities": entities_json
        }

        try:
            # 真实调用 Prompt
            plan = await self.loader.execute_skill("leader_planning.md", inputs)
            return Observation(success=True, data={"plan": plan}, cost=0.02)
        except Exception as e:
            return Observation(success=False, data=None, error_msg=str(e))


class SearchSkill(BaseSkill):
    """
    [真实] 调用 agents.collector.InformationCollectionAgent
    """

    def __init__(self):
        super().__init__("collector")
        # 初始化搜集者 (包含 RAG 和 Web 工具)
        self.agent = InformationCollectionAgent()

    async def execute(self, params: Dict, state: AgentState) -> Observation:
        print(f"   ⚙️ [Skill: Search] 正在执行搜集任务...")

        ctx = state.get("context", {})
        plan = ctx.get("plan", {})

        if not plan or "required_info" not in plan:
            return Observation(success=False, data=None, error_msg="没有找到有效的计划 (Plan)")

        if params.get("force_web"):
            print("      ⚠️ [指令] 强制使用 Web 搜索")
            new_plan = {"required_info": []}
            for item in plan["required_info"]:
                new_item = item.copy()
                new_item["source"] = "web_only"
                new_plan["required_info"].append(new_item)
            plan = new_plan

        try:
            # 真实调用搜集
            res = await self.agent.execute(plan)

            # ==========================================
            # 【新增代码】RAG 结果输出区域 - 开始
            # ==========================================

            # 方式 1: 直接打印原始返回结构（调试用）
            print(f"   📦 [Debug] Agent 完整返回: {res}")

            # 方式 2: 如果 Agent 返回区分了来源，单独提取 RAG 结果
            # 假设返回结构包含 source 标记或分字段存储
            all_results = res.get("validated_data", {})

            # 方案 A: 如果 validated_data 是按 source 分组的字典
            rag_results = {}
            web_results = {}

            for key, value in all_results.items():
                # 假设每个结果项有 _source 标记，或根据查询内容判断
                if isinstance(value, dict) and value.get("_source") == "rag":
                    rag_results[key] = value
                    print(f"   📚 [RAG 结果] {key}: {value.get('content', value)[:200]}...")  # 截断显示
                elif isinstance(value, dict) and value.get("_source") == "web":
                    web_results[key] = value
                else:
                    # 无法区分时，默认归入 RAG（或根据 plan 的 source 判断）
                    rag_results[key] = value

            # 方案 B: 如果 Agent 返回了详细的 chunks/context
            raw_rag_contexts = res.get("rag_contexts", [])  # 原始检索到的文档块
            if raw_rag_contexts:
                print(f"   📄 [RAG 原始文档块] 共检索到 {len(raw_rag_contexts)} 个片段:")
                for idx, chunk in enumerate(raw_rag_contexts[:3], 1):  # 只显示前3个
                    print(f"      [{idx}] 来源: {chunk.get('source', 'unknown')}")
                    print(f"          内容: {chunk.get('text', '')[:150]}...")
                    print(f"          相似度: {chunk.get('score', 'N/A')}")

            # ==========================================
            # 【新增代码】RAG 结果输出区域 - 结束
            # ==========================================

            validated_data = res.get("validated_data", {})

            # 方式 3: 将 RAG 明细加入返回数据，供上层使用
            enhanced_data = {
                "collected_data": validated_data,
                "rag_details": {
                    "rag_only_results": rag_results,  # 仅 RAG 的结果
                    "web_only_results": web_results,  # 仅 Web 的结果
                    "raw_contexts": raw_rag_contexts,  # 原始引用文档
                    "sources_breakdown": {  # 统计信息
                        "rag_count": len(rag_results),
                        "web_count": len(web_results),
                        "total": len(validated_data)
                    }
                }
            }

            success = len(validated_data) > 0
            return Observation(
                success=success,
                data=enhanced_data,  # 改为返回增强后的数据
                cost=0.05
            )

        except Exception as e:
            import traceback
            print(f"   ❌ [RAG 错误] {traceback.format_exc()}")  # 打印详细错误堆栈
            return Observation(success=False, data=None, error_msg=str(e))


class SummarizeSkill(BaseSkill):
    """
    [真实] 调用 agents.summarizer.SummarizerAgent
    """

    def __init__(self):
        super().__init__("summarizer")
        self.agent = SummarizerAgent(strict_mode=False)

    async def execute(self, params: Dict, state: AgentState) -> Observation:
        print(f"   ⚙️ [Skill: Summarize] 正在生成报告...")

        ctx = state.get("context", {})

        # 组装 Summarizer 需要的上下文
        summary_ctx = {
            "user_query": state["query"],
            "plan": ctx.get("plan", {}),
            "collected_data": ctx.get("collected_data", {})
        }

        try:
            # 真实调用
            res = await self.agent.execute(summary_ctx)

            if res.status == "success":
                return Observation(
                    success=True,
                    data={
                        "report": res.report.get("executive_summary"),
                        "is_complete": True
                    },
                    cost=0.02
                )
            else:
                return Observation(
                    success=False,
                    data={"missing": res.missing},
                    error_msg=f"信息不足: {res.missing}",
                    cost=0.01
                )
        except Exception as e:
            return Observation(success=False, data=None, error_msg=str(e))


# ==================== 4. 策略层 (Policy) ====================

class LLMPolicy:
    """基于 LLM 的决策策略"""

    def __init__(self, llm: OpenAILLM):
        self.llm = llm
        self.action_history = []

    async def select_action(self, state: AgentState, available_actions: List[ActionType]) -> Action:
        prompt = self._build_prompt(state, available_actions)
        response_str = await self.llm.generate(prompt)
        print(f"   🧠 [LLM原始响应] {response_str}")
        action = self._parse_response(response_str, available_actions)
        print(f"   🧠 [思维链] {action.reason}")
        # 记录决策数据
        self.action_history.append({
            "state_snapshot": json.dumps(state["context"], ensure_ascii=False)[:500] + "...",
            "prompt": prompt,
            "action_label": action.type.value,
            "reason": action.reason
        })

        return action

    def _build_prompt(self, state: AgentState, available_actions: List[ActionType]) -> str:
        # 只取最近 3 步历史，减少 Token
        history_str = json.dumps([
            {"step": h["step"], "action": h["action"], "success": h["success"]}
            for h in state["history"][-3:]
        ], ensure_ascii=False)

        # 简化上下文显示，防止 Prompt 过长
        ctx_display = state['context'].copy()
        if "collected_data" in ctx_display:
            # 只显示 key，不显示具体长文本
            ctx_display["collected_data"] = list(ctx_display["collected_data"].keys())
        if "report" in ctx_display:
            ctx_display["report"] = "已生成(略)"

        return f"""
你是一个智能金融助手。当前任务："{state['query']}"
已执行步数：{state['step_count']}
当前已知信息状态：{json.dumps(ctx_display, ensure_ascii=False)}

历史操作：
{history_str}

请从以下动作中选择下一步：
{[a.value for a in available_actions]}

逻辑规则：
1. 初始必须先 REWRITE。
2. 得到改写结果后，必须执行 CHASE 进行完整性检查。
3. 如果 CHASE 返回 integrity_ok: false，必须立刻执行 FINISH，并在理由中注明追问问题。
4. 只有当 CHASE 返回 integrity_ok: true 时，才能执行 PLAN。
5. 有计划后，执行搜索和总结。

输出 JSON：
{{
    "action": "动作名",
    "parameters": {{ "force_web": true/false }},
    "reason": "决策理由"
}}
"""

    def _parse_response(self, response: str, available_actions: List[ActionType]) -> Action:
        try:
            data = json.loads(response)
            action_type = ActionType(data.get("action"))
            if action_type not in available_actions:
                return Action(ActionType.FINISH, reason="模型选择了非法动作")
            return Action(
                type=action_type,
                parameters=data.get("parameters", {}),
                reason=data.get("reason", "无理由")
            )
        except:
            return Action(ActionType.FINISH, reason="解析失败")


# ==================== 5. Agent 主体 ====================

class LearnableLeaderAgent:
    def __init__(self, policy: LLMPolicy, max_steps: int = 10):
        self.policy = policy
        self.max_steps = max_steps

        # 注册真实 Skill
        self.skills = {
            ActionType.REWRITE: RewriteSkill(),
            ActionType.PLAN: PlanningSkill(),
            ActionType.CHASE: ChaseSkill(),
            ActionType.SEARCH_DB: SearchSkill(),  # 映射到同一个 Collector
            ActionType.SEARCH_WEB: SearchSkill(),  # 映射到同一个 Collector
            ActionType.SUMMARIZE: SummarizeSkill(),
        }

    async def process(self, query: str) -> Dict:
        state: AgentState = {
            "query": query, "context": {}, "history": [],
            "step_count": 0, "accumulated_reward": 0.0
        }

        trajectory = []

        print(f"🎬 [Leader] 开始任务: {query}")

        while state["step_count"] < self.max_steps:
            # ... (这中间的循环逻辑完全不用动) ...
            # 1. 获取动作
            available = self._get_available_actions(state)
            # 2. 决策
            action = await self.policy.select_action(state, available)
            print(f"   🤖 [Step {state['step_count']}] 决策: {action.type.value} | 理由: {action.reason}")

            if action.type == ActionType.FINISH: break  # 👈 这里跳出循环

            # 3. 执行
            skill = self.skills.get(action.type)
            if skill:
                obs = await skill.execute(action.parameters, state)
            else:
                obs = Observation(False, None, error_msg="工具未定义")

            # 4. 更新状态
            self._update_state(state, action, obs)

            # 5. 记录
            trajectory.append({"state": state.copy(), "action": action, "obs": obs})

            if obs.success and obs.data and obs.data.get("is_complete"):
                break

        # ==================== 🔥 修改这里 🔥 ====================

        # 1. 先准备一个默认的返回结构
        result = {
            "status": "success",  # 默认状态是成功
            "final_report": state["context"].get("report", "未生成"),
            "trajectory": trajectory,
            "total_reward": state["accumulated_reward"]
        }

        # 2. 检查上下文中是否有“追问问题”
        # 如果 context 里有 suggested_question，说明任务是“被迫中断”等待用户输入的
        if "suggested_question" in state["context"]:
            result["status"] = "need_input"  # 👈 改变状态标记
            result["clarification_question"] = state["context"]["suggested_question"]
            result["clarification_options"] = state["context"].get("suggested_options", [])

        return result

    def _get_available_actions(self, state) -> List[ActionType]:
        ctx = state["context"]
        # 强逻辑约束，引导模型走正确流程
        if "rewritten_query" not in ctx: return [ActionType.REWRITE]
        if "integrity_ok" not in ctx and "suggested_question" not in ctx:
            return [ActionType.CHASE]
        if "suggested_question" in ctx:
            return [ActionType.FINISH]
            # 3. 只有检查通过了，且没有计划，才允许 Plan
        if "plan" not in ctx:
            return [ActionType.PLAN]


        options = [ActionType.SEARCH_DB, ActionType.SEARCH_WEB]
        if "collected_data" in ctx: options.append(ActionType.SUMMARIZE)

        return options

    def _update_state(self, state, action, obs):
        state["step_count"] += 1
        if obs.success and obs.data:
            if "collected_data" not in state["context"]: state["context"]["collected_data"] = {}

            # 特殊处理：合并搜集到的数据
            if "collected_data" in obs.data:
                state["context"]["collected_data"].update(obs.data["collected_data"])
            else:
                state["context"].update(obs.data)

        state["history"].append({
            "step": state["step_count"],
            "action": action.type.value,
            "success": obs.success
        })