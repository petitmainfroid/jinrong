import os

# 定义一些常量
PDF_TEXT_DIR = 'pdf_docs'
ERROR_PDF_DIR = 'error_pdfs'

# 强制设置为本地模式 (False)
ONLINE = False

# =======================================================
# 核心修改：让路径自动适应你的 Windows 本地目录
# =======================================================

# 获取 config.py 所在的文件夹路径 (即 D:\projet\RAG)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 设置数据根目录为当前目录下的 data 文件夹
# 最终 DATA_PATH 会变成 D:\projet\RAG\data\
DATA_PATH = os.path.join(BASE_DIR, 'data')

# 下面这些模型路径在“检索阶段”用不到，先留空或者保持默认即可，防止报错
CLASSIFY_CHECKPOINT_PATH = ""
KEYWORDS_CHECKPOINT_PATH = ""
NL2SQL_CHECKPOINT_PATH = ""
XPDF_PATH = ""  # pdfplumber 不需要 xpdf，这里留空即可

# 打印一下路径，运行的时候方便确认
print(f"[Config] Data Path set to: {DATA_PATH}")