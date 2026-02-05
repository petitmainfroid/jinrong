import asyncio
import os
import sys
import json
import config
# 1. 确保能导入 agents 目录
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 2. 导入可学习 Leader 的核心组件
# 假设你把上一段代码保存为了 agents/learnable_leader.py
from agents.leader import (
    OpenAILLM, 
    LLMPolicy, 
    LearnableLeaderAgent,
    ActionType
)

# ==================== 配置区域 ====================
# 这里填入你的大模型 API 配置
# 建议先用 DeepSeek-V3 或 GPT-4o 这种强逻辑模型来测试效果
API_KEY = config.LLM_API_KEY      # 引用 config 里的 DeepSeek Key
BASE_URL = config.LLM_BASE_URL    # 引用 config 里的 Base URL
MODEL_NAME = config.LLM_MODEL_NAME # 引用 config 里的模型名称

# 如果你想测试 Qwen (通过兼容接口):
# BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# MODEL_NAME = "qwen-plus" 
# ================================================

async def main():
    print(f"🚀 初始化 Model-Driven Agent (基于 {MODEL_NAME})...\n")

    # 1. 实例化 LLM 客户端
    # 这个类会自动处理 JSON Mode，保证模型输出能被程序解析
    llm = OpenAILLM(api_key=API_KEY, base_url=BASE_URL, model_name=MODEL_NAME)

    # 2. 实例化策略 (Policy)
    # temperature=0.1 很重要！让模型决策更稳定，不做随机尝试
    policy = LLMPolicy(llm)

    # 3. 实例化 Agent
    # max_steps=15: 防止模型陷入死循环
    agent = LearnableLeaderAgent(policy, max_steps=15)

    # 4. 准备测试问题
    # 找一个稍微复杂、需要多步操作的问题，才能看出模型“执行代码”的能力
    test_query = "分析一下贵州茅台2023年的营收情况，并帮我计算一下如果2024年增长15%是多少。"
    
    print(f"👤 用户问题: {test_query}")
    print("-" * 60)

    # 5. 开始执行
    # 这里会进入 While 循环，模型每一步都会自己决定调用哪个函数
    try:
        result = await agent.process(test_query)

        # 6. 输出结果分析
        print("\n" + "=" * 60)
        print(f"✅ 任务完成！总步数: {len(result['trajectory'])}")
        print(f"💰 获得奖励 (Reward): {result['total_reward']:.2f}")
        print("-" * 60)
        print(f"📄 最终报告:\n{result['final_report']}")
        print("=" * 60)

        # 7. 打印思维链 (SFT 数据的核心)
        print("\n🧠 模型决策轨迹 (思维链):")
        for i, step in enumerate(result['trajectory']):
            action = step['action']
            obs = step.get('observation', {}) # 如果你修改了代码结构，注意这里
            
            # 打印格式：步骤 - [动作类型] - 理由
            print(f"\n[Step {i+1}] 🤖 动作: {action.type.value}")
            print(f"         🤔 思考: {action.reason}")
            print(f"         🛠️ 参数: {json.dumps(action.parameters, ensure_ascii=False)}")
            
            # 如果有 Reward 信息
            reward = step.get('reward', 0)
            print(f"         🏆 奖励: {reward}")

    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Windows 环境下的事件循环策略
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main())