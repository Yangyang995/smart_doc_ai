import streamlit as st
import os
from core.rag.loader import DocumentLoader
from core.rag.splitter import TextSplitter
from core.rag.retriever import Retriever
from core.agent.agent import Agent
from core.memory.conversation import ConversationMemory
from utils.file_utils import FileUtils
from utils.logger import get_logger

# 初始化日志记录器
logger = get_logger(__name__)

# 初始化组件
retriever = Retriever()
agent = Agent()
memory = ConversationMemory()

# 初始化 session_state
if "messages" not in st.session_state:
    st.session_state.messages = memory.get_history()

if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0

# 获取当前向量库中的文档列表
try:
    existing_docs = retriever.get_document_list()
except Exception as e:
    existing_docs = []

# Streamlit 配置
st.set_page_config(
    page_title="智能文档AI助手",
    page_icon="📚",
    layout="wide"
)

# 侧边栏
with st.sidebar:
    st.title("智能文档AI助手")
    
    # 文档上传
    st.header("文档上传")
    uploaded_files = st.file_uploader("选择文件", accept_multiple_files=True, type=["pdf", "docx", "txt", "md"], key=f"file_uploader_{st.session_state.uploader_key}")
    
    if uploaded_files:
        has_new_file = False
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in existing_docs:
                try:
                    logger.info(f"开始处理文件: {uploaded_file.name}")

                    file_path = FileUtils.save_uploaded_file(uploaded_file)
                    logger.info(f"文件保存路径: {file_path}")

                    documents = DocumentLoader.load_document(file_path)
                    logger.info(f"加载到 {len(documents)} 个文档")

                    chunks = TextSplitter.split_text(documents)
                    logger.info(f"分块后得到 {len(chunks)} 个块")

                    retriever.add_document(chunks)
                    logger.info("文档已添加到向量库")

                    existing_docs.append(uploaded_file.name)
                    has_new_file = True

                    new_docs = retriever.get_document_list()
                    logger.info(f"上传后向量库中的文档: {new_docs}")

                    st.success(f"文件 '{uploaded_file.name}' 上传成功！")
                    logger.info(f"文件 '{uploaded_file.name}' 上传成功")

                except Exception as e:
                    st.error(f"文件 '{uploaded_file.name}' 上传失败: {str(e)}")
                    logger.error(f"文件 '{uploaded_file.name}' 上传失败: {str(e)}")
            else:
                logger.info(f"文件 {uploaded_file.name} 已存在于向量库中，跳过")
                st.info(f"文件 '{uploaded_file.name}' 已上传过，无需重复上传！")

        if has_new_file:
            st.session_state.uploader_key += 1
            import time
            time.sleep(1)
            st.rerun()
    
    # 知识库管理
    st.header("知识库管理")
    
    # 查看已上传文档
    st.subheader("已上传文档")
    try:
        # 每次都从向量库读取最新的文档列表
        document_list = retriever.get_document_list()
        logger.info(f"当前文档列表: {document_list}")
        
        if document_list:
            for doc in document_list:
                col1, col2 = st.columns([3, 1]) # 分栏：显示文档名 + 删除按钮
                col1.write(doc)
                if col2.button("删除", key=f"delete_{doc}"):
                    try:
                        retriever.delete_document(doc)
                        FileUtils.delete_file(doc)
                        # 使用success显示提示（会一直显示直到刷新）
                        st.success(f"文档 '{doc}' 删除成功！")
                        logger.info(f"文档 '{doc}' 删除成功")
                        # 更新上传器key，强制重置上传器
                        st.session_state.uploader_key += 1
                        # 添加短暂延迟确保提示显示
                        import time
                        time.sleep(0.5)
                        # 强制刷新页面
                        st.rerun()
                    except Exception as e:
                        st.error(f"文档 '{doc}' 删除失败: {str(e)}")
                        logger.error(f"文档 '{doc}' 删除失败: {str(e)}")
        else:
            st.write("暂无上传的文档")
    except Exception as e:
        st.error(f"获取文档列表失败: {str(e)}")
        logger.error(f"获取文档列表失败: {str(e)}")
    
    # 会话设置
    st.header("会话设置")
    if st.button("清空会话历史"):
        memory.clear_memory()
        st.session_state.messages = []
        st.rerun()

# 主对话区
st.title("智能文档AI助手")

# 显示历史对话
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 输入区
if user_input := st.chat_input("请输入您的问题..."):
    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(user_input)

    # 保存用户消息
    st.session_state.messages.append({"role": "user", "content": user_input})
    memory.add_message("user", user_input)

    # 流式输出助手回复
    with st.chat_message("assistant"):
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown("🤔 AI 正在思考中...")

        response_placeholder = st.empty()
        full_response = ""
        first_chunk = True

        for chunk in agent.stream(user_input):
            if first_chunk:
                thinking_placeholder.empty()
                first_chunk = False
            full_response += chunk
            response_placeholder.markdown(full_response + "▌")

        if first_chunk:
            thinking_placeholder.empty()
        response_placeholder.markdown(full_response)

    # 保存助手回复
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    memory.add_message("assistant", full_response)

    logger.info(f"用户输入: {user_input}")
    logger.info(f"Agent 响应: {full_response}")
