import asyncio
import os
import sys
import json
import config

# 1. 确保能导入 agents 目录
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 2. 导入组件
from agents.leader import (
    OpenAILLM,
    LLMPolicy,
    LearnableLeaderAgent,
    ActionType
)

# ==================== 配置区域 ====================
API_KEY = config.LLM_API_KEY
BASE_URL = config.LLM_BASE_URL
MODEL_NAME = config.LLM_MODEL_NAME


# ================================================

async def main():
    print(f"🚀 初始化 Model-Driven Agent (基于 {MODEL_NAME})...\n")

    # 1. 实例化核心组件
    llm = OpenAILLM(api_key=API_KEY, base_url=BASE_URL, model_name=MODEL_NAME)
    policy = LLMPolicy(llm)
    agent = LearnableLeaderAgent(policy, max_steps=15)

    # 2. 获取初始问题
    print("请输入您的问题（直接回车使用默认测试问题）：")
    user_input = input("> ").strip()
    if user_input == "":
        user_input = "公司23年营收？"  # 默认测试问题
        print(f"检测到直接回车，已使用默认问题：{user_input}")

    current_query = user_input

    # ==================== 🔥 核心修改区域开始 🔥 ====================
    # 使用 while 循环来支持多轮对话（追问机制）

    while True:
        print(f"\n🎬 [System] 正在处理任务: {current_query}")
        print("-" * 30)

        try:
            # 执行 Agent 流程
            result = await agent.process(current_query)

            # --- 分支 A: Agent 请求追问 (Need Input) ---
            if result.get("status") == "need_input":
                question = result.get("clarification_question")
                options = result.get("clarification_options")

                print(f"\n🤖 [Agent 追问]: {question}")
                if options:
                    print(f"   (参考选项: {options})")

                # 获取用户补充信息
                print("\n" + "-" * 30)
                supplement = input("👤 [请输入您的回答] (输入 'q' 退出): ").strip()

                if supplement.lower() == 'q':
                    print("用户取消任务。")
                    break

                # 简单策略：将补充信息拼接到原问题后面
                # 例如： "公司23年营收？" + " " + "贵州茅台"
                current_query = f"{current_query} {supplement}"
                print(f"🔄 [System] 信息已更新，重新规划任务...")
                continue  # 跳过本次循环剩下的代码，带入新 query 重新 process

            # --- 分支 B: 任务完成 (Success) ---
            else:
                # 输出结果分析
                print("\n" + "=" * 60)
                print(f"✅ 任务完成！总步数: {len(result.get('trajectory', []))}")
                print(f"💰 获得奖励 (Reward): {result.get('total_reward', 0):.2f}")
                print("-" * 60)
                print(f"📄 最终报告:\n{result.get('final_report', '无报告')}")
                print("=" * 60)

                # 打印思维链
                print("\n🧠 模型决策轨迹 (思维链):")
                for i, step in enumerate(result.get('trajectory', [])):
                    action = step['action']

                    print(f"\n[Step {i + 1}] 🤖 动作: {action.type.value}")
                    print(f"         🤔 思考: {action.reason}")
                    print(f"         🛠️ 参数: {json.dumps(action.parameters, ensure_ascii=False)}")

                break  # 任务真正完成，退出 while 循环

        except Exception as e:
            print(f"\n❌ 运行出错: {e}")
            import traceback
            traceback.print_exc()
            break

    # ==================== 🔥 核心修改区域结束 🔥 ====================


if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main())