import os
import re
import json
from dataclasses import dataclass
from typing import Dict, Optional, Any
from openai import AsyncOpenAI

# 引入你的统一配置
try:
    import config
except ImportError:
    # 简单的兜底，防止 IDE 报错
    class ConfigMock:
        SKILLS_DIR = "skills"
        LLM_API_KEY = ""
        LLM_BASE_URL = ""
    config = ConfigMock()

@dataclass
class SkillConfig:
    """Skill 配置数据结构"""
    name: str
    description: str
    model: str
    temperature: float
    max_tokens: int
    response_format: Optional[dict] # 新增：支持定义返回格式(json_object)
    system_prompt: str
    user_prompt_template: str

    def render_prompt(self, **kwargs) -> str:
        """渲染 User Prompt，填充变量"""
        try:
            result = self.user_prompt_template
            # 按长度降序排序，避免短变量名干扰 (如 {a} 和 {abc})
            sorted_items = sorted(kwargs.items(), key=lambda x: len(x[0]), reverse=True)
            for key, value in sorted_items:
                # 简单转义处理，防止注入
                val_str = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
                # 如果是简单字符串，去掉首尾引号，看起来更自然
                if isinstance(value, str):
                    val_str = value
                
                placeholder = "{" + key + "}"
                if placeholder in result:
                    result = result.replace(placeholder, val_str)
            return result
        except Exception as e:
            raise ValueError(f"Prompt 渲染错误: {e}")

class SkillLoader:
    """
    全能 Skill 加载与执行器
    负责：读取 Markdown -> 解析配置 -> 调用 LLM -> 返回结果
    """
    
    def __init__(self, api_key: str = None, base_url: str = None):
        # 优先使用传入参数，否则使用 config
        self.api_key = api_key or getattr(config, 'LLM_API_KEY', None)
        self.base_url = base_url or getattr(config, 'LLM_BASE_URL', None)
        
        if not self.api_key:
            raise ValueError("未配置 API Key，请检查 config.py 或传入参数")

        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    async def execute_skill(self, skill_file: str, inputs: Dict[str, Any]) -> Dict:
        """
        核心方法：执行一个 Skill
        :param skill_file: skill 文件名 (如 'rewrite.md')
        :param inputs: 填充 Prompt 的变量字典
        :return: LLM 返回的字典 (如果是 JSON) 或包含 content 的字典
        """
        # 1. 确定文件路径
        # 如果传入的是绝对路径就用绝对路径，否则去 config.SKILLS_DIR 找
        if os.path.isabs(skill_file):
            file_path = skill_file
        else:
            file_path = os.path.join(getattr(config, 'SKILLS_DIR', 'skills'), skill_file)

        # 2. 加载并解析 Markdown
        skill_config = self._load_markdown(file_path)

        # 3. 渲染 User Prompt
        user_msg = skill_config.render_prompt(**inputs)

        # 4. 调用 LLM
        # print(f"🚀 [Skill] Executing: {skill_config.name} ({skill_config.model})") 
        try:
            response = await self.client.chat.completions.create(
                model=skill_config.model,
                messages=[
                    {"role": "system", "content": skill_config.system_prompt},
                    {"role": "user", "content": user_msg}
                ],
                temperature=skill_config.temperature,
                max_tokens=skill_config.max_tokens,
                response_format=skill_config.response_format
            )

            content = response.choices[0].message.content
            
            # 5. 结果处理 (尝试解析 JSON)
            # 如果配置里要求了 json_object，或者内容看起来像 JSON
            if skill_config.response_format and skill_config.response_format.get('type') == 'json_object':
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    print(f"⚠️ [Warning] LLM 返回的不是有效 JSON: {content[:50]}...")
                    return {"raw_content": content, "error": "json_parse_fail"}
            else:
                return {"content": content}

        except Exception as e:
            print(f"❌ Skill Execution Failed [{skill_file}]: {e}")
            # 返回空字典或错误信息，防止主程序崩溃
            return {"error": str(e)}

    def _load_markdown(self, md_path: str) -> SkillConfig:
        """内部方法：读取并解析 Markdown 文件"""
        if not os.path.exists(md_path):
            raise FileNotFoundError(f"Skill file not found: {md_path}")

        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 解析头部元数据
        meta = self._parse_front_matter(content)
        
        # 处理 Response Format (支持在 md 头部写 "ResponseFormat: json_object")
        resp_format = None
        if meta.get('responseformat') == 'json_object':
            resp_format = {"type": "json_object"}

        # 提取 Prompts
        system_prompt = self._extract_section(content, "System Prompt")
        user_template = self._extract_section(content, "User Prompt Template")

        if not system_prompt or not user_template:
            raise ValueError(f"Invalid skill file: {md_path}. 缺少 System Prompt 或 User Prompt Template 章节。")

        return SkillConfig(
            name=meta.get('name', 'unknown_skill'),
            description=meta.get('description', ''),
            model=meta.get('model', getattr(config, 'LLM_MODEL_NAME', 'deepseek-chat')),
            temperature=float(meta.get('temperature', getattr(config, 'AGENT_TEMPERATURE', 0.1))),
            max_tokens=int(meta.get('maxtokens', 2000)),
            response_format=resp_format,
            system_prompt=system_prompt,
            user_prompt_template=user_template
        )

    @staticmethod
    def _parse_front_matter(content: str) -> Dict[str, str]:
        """解析 # Key: Value"""
        meta = {}
        for line in content.split('\n'):
            line = line.strip()
            if not line: continue
            if not line.startswith('#'): break # 碰到非注释行停止
            
            # 去掉开头的 # 
            content_line = line[1:].strip()
            if ':' in content_line:
                key, value = content_line.split(':', 1)
                meta[key.strip().lower().replace('_', '')] = value.strip()
        return meta

    @staticmethod
    def _extract_section(content: str, section_name: str) -> Optional[str]:
        """提取 ## Section 下的内容"""
        # 匹配 ## SectionName 到下一个 ## 或文件结束
        pattern = rf'## {section_name}\s*\n(.*?)(?=\n## |\Z)'
        match = re.search(pattern, content, re.DOTALL)
        return match.group(1).strip() if match else None