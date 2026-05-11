import logging
import os
from config.settings import PROJECT_ROOT

# 日志目录
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "app.log")),
        logging.StreamHandler()
    ]
)

# 创建日志记录器
def get_logger(name):
    """
    获取日志记录器
    :param name: 记录器名称
    :return: 日志记录器
    """
    return logging.getLogger(name)
