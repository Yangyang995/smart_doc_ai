import os#用于操作系统路径、环境变量
from dotenv import load_dotenv#用于加载 .env 文件中的环境变量

# 加载环境变量 — 指定 config 目录下的 .env 文件
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# os.path.abspath(__file__)：获取当前settings.py文件的绝对路径；
# 两次os.path.dirname()：向上两级目录，得到项目的根目录（比如settings.py在config/下，根目录就是config/的上一级）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 拼接出项目的data目录路径（用于存放原始文档 / 数据）；
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
# 拼接出 Chroma 向量数据库的存储目录路径
CHROMA_DB_DIR = os.path.join(PROJECT_ROOT, "chroma_db")

# 文档处理配置
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 3

# LLM 配置
MODEL_NAME = os.getenv("MODEL_NAME", "qwen3-max")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2000"))

# API 密钥
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# 向量嵌入配置
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v2") # 默认使用 OpenAI 的嵌入模型
