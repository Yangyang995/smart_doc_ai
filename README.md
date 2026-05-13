# 智能文档AI助手（RAG + 工具Agent）

## 项目简介

智能文档AI助手是一个基于 Python 的私有化部署智能文档处理系统，核心功能包括：

- **私有 RAG 知识库**：支持上传 PDF、Word、纯文本和 Markdown 格式的文档，构建专属知识库
- **Agent 自主决策**：内置多个文档处理工具，通过 LangGraph 实现自主判断和工具调用
- **Web 界面**：使用 FastAPI + 静态前端页面提供简单易用的用户界面，支持会话记忆和历史对话

## 环境搭建

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API 密钥

在 `config/.env` 文件中配置 API 密钥：

```
# API Keys
OPENAI_API_KEY=
DASHSCOPE_API_KEY=
DEEPSEEK_API_KEY=

# Model Configuration
MODEL_NAME=deepseek-chat
TEMPERATURE=0.3
MAX_TOKENS=2000

# Embedding Model
EMBEDDING_MODEL=text-embedding-3-small
```

**注意**：至少需要配置以下 API 密钥之一：
- `DASHSCOPE_API_KEY`（用于通义千问模型）
- `OPENAI_API_KEY`（用于 OpenAI 模型）
- `DEEPSEEK_API_KEY`（用于 DeepSeek 模型）

## 运行命令

```bash
python api.py
```

运行后，在浏览器中访问 `http://localhost:8000` 即可使用。

## 核心模块说明

### 1. RAG 知识库模块

- **文档加载器**：支持 PDF、Word、纯文本和 Markdown 格式
- **文本分块器**：使用 LangChain 的 RecursiveCharacterTextSplitter
- **向量库管理**：使用 Chroma 向量数据库，支持持久化存储
- **检索器**：基于相似度的向量检索，返回相关文档片段

### 2. Agent 工具调用模块

- **RAG 问答工具**：从知识库中检索信息，回答用户问题
- **文档总结工具**：支持摘要、要点、思维导图式三种总结模式
- **文本翻译工具**：支持中英文互译
- **专业术语解释工具**：识别并解释专业术语
- **文本润色改写工具**：支持润色、扩写、缩写、调整格式

### 3. 会话与交互模块

- **Web 界面**：Streamlit 实现，包含侧边栏和主对话区
- **会话记忆**：使用 LangChain 的 ConversationBufferMemory
- **历史对话管理**：支持查看和清空历史对话

## 示例使用场景

### 1. 知识库问答

**用户**：请问文档中关于人工智能的定义是什么？

**助手**：根据知识库中的内容，人工智能是指计算机系统模拟人类智能的能力，包括学习、推理、感知等方面。

### 2. 文档总结

**用户**：请对《人工智能导论》文档进行要点总结。

**助手**：
1. 人工智能的定义和发展历程
2. 人工智能的主要技术领域
3. 人工智能的应用场景
4. 人工智能的伦理和挑战

### 3. 文本翻译

**用户**：请将以下内容翻译成英文："人工智能正在改变我们的生活。"

**助手**：Artificial intelligence is changing our lives.

### 4. 术语解释

**用户**：请解释什么是机器学习。

**助手**：机器学习是人工智能的一个分支，通过算法让计算机从数据中学习，无需明确编程即可执行任务。

### 5. 文本润色

**用户**：请润色以下文本："这个产品很好，大家都喜欢。"

**助手**：这款产品品质卓越，深受广大用户的喜爱和好评。

## 项目结构

```
smart_doc_ai/
├── app.py                 # Streamlit主入口
├── config/
│   ├── settings.py        # 全局配置、模型参数、路径配置
│   └── .env               # 环境变量（API密钥等）
├── core/
│   ├── rag/
│   │   ├── loader.py      # 文档加载器
│   │   ├── splitter.py    # 文本分块器
│   │   ├── vector_store.py # 向量库管理
│   │   └── retriever.py   # 检索器封装
│   ├── agent/
│   │   ├── tools.py       # 所有工具定义
│   │   └── agent.py       # LangGraph Agent实现
│   └── memory/
│       └── conversation.py # 会话记忆管理
├── utils/
│   ├── logger.py          # 日志工具
│   └── file_utils.py      # 文件处理工具
├── data/                  # 上传的文档存储目录
├── chroma_db/             # Chroma向量库持久化目录
├── requirements.txt       # 依赖清单
└── README.md              # 项目说明文档
```

## 注意事项

1. 首次运行时，系统会自动创建必要的目录结构
2. 上传的文档会存储在 `data/` 目录中
3. 向量库会持久化存储在 `chroma_db/` 目录中
4. 日志文件会生成在 `logs/` 目录中
5. 请确保配置了至少一个 API 密钥，否则系统无法正常运行
