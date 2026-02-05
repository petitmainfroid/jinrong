import os
from dotenv import load_dotenv

# =======================================================
# 0. 加载环境变量 (必须放在最前面)
# =======================================================
# 这行代码会搜索项目根目录下的 .env 文件并读取其中的变量
load_dotenv()

# =======================================================
# 1. 基础路径配置 (自动适应 Windows/Linux)
# =======================================================

# 获取项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据存储目录
DATA_PATH = os.path.join(BASE_DIR, 'data')
SKILLS_DIR = os.path.join(BASE_DIR, 'skills')
PDF_TEXT_DIR = os.path.join(DATA_PATH, 'pdf_docs')
ERROR_PDF_DIR = os.path.join(DATA_PATH, 'error_pdfs')
FAISS_INDEX_PATH = os.path.join(DATA_PATH, 'faiss_index')
DOCS_INFO_PATH = os.path.join(DATA_PATH, 'docs.pkl')

print(f"[Config] Project Root: {BASE_DIR}")
print(f"[Config] Data Path:    {DATA_PATH}")


# =======================================================
# 2. LLM (大模型) 配置
# =======================================================

# 从环境变量中读取 DeepSeek Key，如果没读到则默认为 None
LLM_API_KEY = os.getenv("DEEPSEEK_API_KEY")
LLM_BASE_URL = "https://api.deepseek.com"
LLM_MODEL_NAME = "deepseek-chat"

# Agent 参数设置
AGENT_TEMPERATURE = 0.1
CREATIVE_TEMPERATURE = 0.3
MAX_RETRIES = 2


# =======================================================
# 3. 工具 (Tools) 配置
# =======================================================

# 从环境变量中读取 Tavily Key
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# RAG 配置
ONLINE = False
RAG_TOP_K = 5


# =======================================================
# 4. 兼容性配置 & 目录创建 (保持不变)
# =======================================================
CLASSIFY_CHECKPOINT_PATH = ""
KEYWORDS_CHECKPOINT_PATH = ""
NL2SQL_CHECKPOINT_PATH = ""
XPDF_PATH = ""

def ensure_directories():
    """在程序启动时自动创建不存在的文件夹"""
    dirs_to_create = [DATA_PATH, SKILLS_DIR, PDF_TEXT_DIR, ERROR_PDF_DIR]
    for d in dirs_to_create:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"[Config] Created directory: {d}")

ensure_directories()