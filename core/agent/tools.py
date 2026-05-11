from langchain.tools import tool
from langchain_community.chat_models.tongyi import ChatTongyi
from core.rag.retriever import Retriever
from config.settings import MODEL_NAME, TEMPERATURE, MAX_TOKENS, DASHSCOPE_API_KEY

# ====================== 工具 1：RAG 问答 ======================
@tool
def rag_qa(query: str) -> str:
    """从知识库中检索信息，回答用户关于文档内容的问题"""
    try:
        retriever = Retriever()
        llm = ChatTongyi(
            model=MODEL_NAME,
            dashscope_api_key=DASHSCOPE_API_KEY,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        docs = retriever.retrieve(query)
        context = "\n".join([f"文档: {doc.metadata.get('filename', '未知')}\n内容: {doc.page_content}" for doc in docs])
        prompt = f"根据以下上下文回答问题：\n\n上下文：\n{context}\n\n问题：{query}\n\n回答："
        response = llm.invoke(prompt)
        sources = "\n\n来源：\n" + "".join([f"{i+1}. {doc.metadata.get('filename','未知')}\n" for i, doc in enumerate(docs)])
        return response.content + sources
    except Exception as e:
        return f"RAG 问答失败: {str(e)}"

# ====================== 工具 2：文档总结 ======================
@tool
def document_summary(document_name: str, summary_type: str = "摘要") -> str:
    """对文档进行总结，支持摘要、要点、思维导图式三种模式"""
    try:
        retriever = Retriever()
        llm = ChatTongyi(
            model=MODEL_NAME,
            dashscope_api_key=DASHSCOPE_API_KEY,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        query = f"请获取 {document_name} 的全部内容"
        docs = retriever.retrieve(query)
        relevant_docs = [doc for doc in docs if doc.metadata.get('filename') == document_name]
        if not relevant_docs:
            return f"未找到文档: {document_name}"
        content = "\n".join([doc.page_content for doc in relevant_docs])
        prompt = f"请对以下内容{summary_type}：\n{content}\n\n{summary_type}："
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"文档总结失败: {str(e)}"

# ====================== 工具 3：文本翻译 ======================
@tool
def text_translation(text: str = "", target_language: str = "英文", document_name: str = "") -> str:
    """翻译文本或文档内容。翻译文档时传入document_name（文档名），翻译指定文本时传入text。target_language为目标语言，默认英文"""
    try:
        llm = ChatTongyi(
            model=MODEL_NAME,
            dashscope_api_key=DASHSCOPE_API_KEY,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )

        content_to_translate = text

        # 如果指定了文档名，从向量库检索文档内容
        if document_name:
            retriever = Retriever()
            query = f"请获取 {document_name} 的全部内容"
            docs = retriever.retrieve(query)
            relevant_docs = [doc for doc in docs if doc.metadata.get('filename') == document_name]
            if not relevant_docs:
                return f"未找到文档: {document_name}"
            content_to_translate = "\n".join([doc.page_content for doc in relevant_docs])

        if not content_to_translate:
            return "请提供需要翻译的文本或文档名称"

        prompt = f"请将以下文本翻译成{target_language}：\n{content_to_translate}\n\n{target_language}："
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"翻译失败: {str(e)}"

# ====================== 工具 4：术语解释 ======================
@tool
def term_explanation(term: str) -> str:
    """识别并解释专业术语"""
    try:
        retriever = Retriever()
        llm = ChatTongyi(
            model=MODEL_NAME,
            dashscope_api_key=DASHSCOPE_API_KEY,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        docs = retriever.retrieve(term)
        context = "\n".join([doc.page_content for doc in docs])
        prompt = f"根据上下文解释术语 '{term}'：\n{context}\n\n解释："
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"术语解释失败: {str(e)}"

# ====================== 工具 5：文本润色 ======================
@tool
def text_polish(text: str, polish_type: str = "润色") -> str:
    """支持对文本进行润色、扩写、缩写、调整格式"""
    try:
        llm = ChatTongyi(
            model=MODEL_NAME,
            dashscope_api_key=DASHSCOPE_API_KEY,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        prompt = f"请对文本{polish_type}：\n{text}\n\n{polish_type}后："
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"文本处理失败: {str(e)}"