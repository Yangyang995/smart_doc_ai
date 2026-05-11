from langchain.agents import create_agent
from langchain_community.chat_models.tongyi import ChatTongyi
from core.agent.tools import rag_qa, document_summary, text_translation, term_explanation, text_polish
from config.settings import MODEL_NAME, TEMPERATURE, MAX_TOKENS, DASHSCOPE_API_KEY


class Agent:
    """使用 langgraph create_agent 创建的 Agent（支持流式输出）"""

    def __init__(self):
        """初始化 Agent"""
        self.model = self._get_model()
        self.tools = [rag_qa, document_summary, text_translation, term_explanation, text_polish]
        self.system_prompt = self._load_system_prompts()
        self.agent = self._create_agent()

    def _get_model(self):
        """获取聊天模型"""
        if DASHSCOPE_API_KEY:
            return ChatTongyi(
                model=MODEL_NAME,
                dashscope_api_key=DASHSCOPE_API_KEY,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS
            )
        else:
            raise ValueError("请配置 DASHSCOPE_API_KEY")

    def _load_system_prompts(self):
        """加载系统提示"""
        return """你是一个专业的文档分析助手，你的任务是根据用户上传的文档内容详细回答问题。

## 可用工具：
1. rag_qa(query): 从知识库中检索信息回答问题
2. document_summary(document_name, summary_type): 对指定文档进行总结
3. text_translation(text, target_language): 文本翻译
4. term_explanation(term): 术语解释
5. text_polish(text, polish_type): 文本润色

## 工作流程：
1. 如果用户要求"总结"某个文档，调用 document_summary 工具
2. 如果用户问问题，调用 rag_qa 工具检索相关信息
3. 如果用户要求翻译，调用 text_translation 工具
4. 如果用户要求解释术语，调用 term_explanation 工具
5. 如果用户要求润色文本，调用 text_polish 工具

请根据用户的问题选择最合适的工具！"""

    def _create_agent(self):
        """创建 Agent"""
        return create_agent(
            model=self.model,
            system_prompt=self.system_prompt,
            tools=self.tools
        )

    def run(self, query):
        """
        运行 Agent（同步调用）
        :param query: 用户输入
        :return: Agent 执行结果
        """
        try:
            result = self.agent.invoke({"messages": [{"role": "user", "content": query}]})
            final_message = result["messages"][-1]
            if hasattr(final_message, 'content'):
                return final_message.content
            else:
                return "Agent 执行失败"
        except Exception as e:
            return f"Agent 运行失败: {str(e)}"

    def stream(self, query):
        """
        流式输出 Agent 响应（逐字输出）
        :param query: 用户输入
        :return: 生成器，逐字返回响应内容
        """
        try:
            input_dict = {
                "messages": [
                    {"role": "user", "content": query}
                ]
            }
            
            previous_content = ""
            first_chunk = True
            
            for chunk in self.agent.stream(input_dict, stream_mode="values"):
                latest_message = chunk["messages"][-1]
                
                # 跳过第一个 chunk（通常是用户消息的回显）
                if first_chunk:
                    first_chunk = False
                    continue
                
                # 只返回 assistant 的消息
                if hasattr(latest_message, 'content') and latest_message.content:
                    content = latest_message.content.strip()
                    
                    # 检查消息内容是否是用户的问题（避免重复）
                    if content == query.strip():
                        continue
                    
                    # 计算新增的内容（增量输出）
                    if len(content) > len(previous_content):
                        new_content = content[len(previous_content):]
                        # 逐字输出
                        for char in new_content:
                            yield char
                        previous_content = content
        except Exception as e:
            yield f"Agent 运行失败: {str(e)}"