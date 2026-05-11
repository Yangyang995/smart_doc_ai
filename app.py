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
retriever = Retriever()  # 检索器（文档向量化、存储、检索）
agent = Agent()          # 智能代理（处理用户问题、调用检索、生成回答）
memory = ConversationMemory()  # 会话记忆（存储/管理聊天记录）

# 获取当前向量库中的文档列表（用于检查是否重复上传）
try:
    existing_docs = retriever.get_document_list()
except Exception as e:
    existing_docs = []

# 初始化上传器重置计数器
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0

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
        for uploaded_file in uploaded_files:
            # 只处理新上传的文件，避免重复上传（检查向量库中是否已存在）
            if uploaded_file.name not in existing_docs:
                try:
                    logger.info(f"开始处理文件: {uploaded_file.name}")
                    
                    # 保存文件
                    file_path = FileUtils.save_uploaded_file(uploaded_file)
                    logger.info(f"文件保存路径: {file_path}")
                    
                    # 加载文档
                    documents = DocumentLoader.load_document(file_path)
                    logger.info(f"加载到 {len(documents)} 个文档")
                    
                    # 分块
                    chunks = TextSplitter.split_text(documents)
                    logger.info(f"分块后得到 {len(chunks)} 个块")
                    
                    # 向量化并存储
                    retriever.add_document(chunks)
                    logger.info("文档已添加到向量库")
                    
                    # 更新已存在文档列表
                    existing_docs.append(uploaded_file.name)
                    
                    # 验证是否成功添加
                    new_docs = retriever.get_document_list()
                    logger.info(f"上传后向量库中的文档: {new_docs}")
                    
                    # 使用success显示提示（会一直显示直到刷新）
                    st.success(f"文件 '{uploaded_file.name}' 上传成功！")
                    logger.info(f"文件 '{uploaded_file.name}' 上传成功")

                    # 更新上传器key，强制重置上传器
                    st.session_state.uploader_key += 1
                    # 添加短暂延迟确保提示显示
                    import time
                    time.sleep(0.5)
                    # 强制刷新页面
                    st.rerun()
                except Exception as e:
                    st.error(f"文件 '{uploaded_file.name}' 上传失败: {str(e)}")
                    logger.error(f"文件 '{uploaded_file.name}' 上传失败: {str(e)}")
            else:
                logger.info(f"文件 {uploaded_file.name} 已存在于向量库中，跳过")
                # 使用success显示提示（会一直显示直到刷新）
                st.success(f"文件 '{uploaded_file.name}' 已上传过，无需重复上传！")
                # 重置上传器，清除已选择的文件
                st.session_state.uploader_key += 1
                # 添加短暂延迟确保提示显示
                import time
                time.sleep(0.5)
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
    memory_type = st.selectbox("会话记忆类型", ["buffer", "summary"])
    if st.button("清空会话历史"):
        memory.clear_memory()
        st.success("会话历史已清空")

# 主对话区
st.title("智能文档AI助手")

# 聊天历史
st.subheader("聊天历史")
chat_history = memory.get_history()
for message in chat_history:
    if message["role"] == "user":
        st.chat_message("user").write(message["content"])
    else:
        st.chat_message("assistant").write(message["content"])

# 输入区
user_input = st.chat_input("请输入您的问题...")

if user_input:
    # 显示用户输入
    st.chat_message("user").write(user_input)
    
    # 添加到会话记忆
    memory.add_message("user", user_input)
    
    # 显示思考中提示
    with st.chat_message("assistant"):
        # 显示思考状态
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown("🤔 智能客服思考中...")
        
        # 使用流式输出显示 Agent 响应
        response_container = st.empty()
        full_response = ""
        
        # 首次获取响应时清除思考提示
        first_response = True
        for chunk in agent.stream(user_input):
            if first_response:
                # 清除思考提示
                thinking_placeholder.empty()
                first_response = False
            
            # 逐字追加输出
            full_response += chunk
            response_container.markdown(full_response)
        
        # 添加到会话记忆
        memory.add_message("assistant", full_response)
        
        logger.info(f"用户输入: {user_input}")
        logger.info(f"Agent 响应: {full_response}")
