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
3. text_translation(text, target_language, document_name): 翻译文本或文档。翻译文档时传document_name（文档名），翻译指定文本时传text
4. term_explanation(term): 术语解释
5. text_polish(text, polish_type): 文本润色

## 工作流程：
1. 如果用户要求"总结"某个文档，调用 document_summary 工具，传入文档名
2. 如果用户问问题，调用 rag_qa 工具检索相关信息
3. 如果用户要求翻译某篇文档，调用 text_translation 工具，传入 document_name（文档名）和 target_language（目标语言）
4. 如果用户要求翻译一段具体文本，调用 text_translation 工具，传入 text（文本内容）
5. 如果用户要求解释术语，调用 term_explanation 工具
6. 如果用户要求润色文本，调用 text_polish 工具

请根据用户的问题选择最合适的工具和参数！"""

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

    TOOL_LABELS = {
        "rag_qa": "检索知识库",
        "document_summary": "文档总结",
        "text_translation": "文本翻译",
        "term_explanation": "术语解释",
        "text_polish": "文本润色",
    }

    def stream(self, query):
        """
        流式输出 Agent 响应，支持思考过程展示。
        返回结构化事件：tool_call / tool_result / content / end
        """
        try:
            input_dict = {"messages": [{"role": "user", "content": query}]}
            previous_content = ""

            for chunk in self.agent.stream(input_dict, stream_mode="updates"):
                for node_name, node_output in chunk.items():
                    if "messages" not in node_output:
                        continue

                    for msg in node_output["messages"]:
                        # 工具调用事件
                        if hasattr(msg, 'tool_calls') and msg.tool_calls:
                            for tc in msg.tool_calls:
                                tool_name = tc.get("name", "unknown")
                                label = self.TOOL_LABELS.get(tool_name, tool_name)
                                yield {
                                    "type": "tool_call",
                                    "tool": tool_name,
                                    "label": label,
                                    "args": tc.get("args", {})
                                }

                        # 工具返回结果事件
                        if hasattr(msg, 'name') and msg.name:
                            label = self.TOOL_LABELS.get(msg.name, msg.name)
                            result_preview = str(msg.content)[:300] if msg.content else ""
                            yield {
                                "type": "tool_result",
                                "tool": msg.name,
                                "label": label,
                                "preview": result_preview
                            }

                        # 最终文本内容（非工具调用消息）
                        if hasattr(msg, 'content') and msg.content and isinstance(msg.content, str):
                            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                continue
                            if hasattr(msg, 'name') and msg.name:
                                continue

                            content = msg.content
                            if len(content) > len(previous_content):
                                new_content = content[len(previous_content):]
                                previous_content = content
                                # 将新内容拆分为小段，模拟逐字流式输出
                                chunk_size = 3
                                for i in range(0, len(new_content), chunk_size):
                                    yield {"type": "content", "content": new_content[i:i + chunk_size]}

            if not previous_content:
                yield {"type": "content", "content": "抱歉，未能生成回复，请重试。"}

        except Exception as e:
            yield {"type": "content", "content": f"Agent 运行失败: {str(e)}"}

        yield {"type": "end"}