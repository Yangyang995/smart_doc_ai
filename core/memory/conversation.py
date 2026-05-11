from langchain_classic.memory import ConversationBufferMemory # 完整缓存所有对话消息
from langchain_classic.memory import ConversationSummaryMemory # 用大模型自动总结长对话
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_models.tongyi import ChatTongyi
from config.settings import MODEL_NAME, TEMPERATURE, DASHSCOPE_API_KEY


class ConversationMemory:
    """会话记忆管理"""

    def __init__(self, memory_type="buffer"):
        """
        初始化会话记忆
        :param memory_type: 记忆类型，buffer 或 summary
        """
        self.memory_type = memory_type # 保存记忆类型（默认buffer）
        self.llm = self._get_llm()     # 初始化大模型实例（用于summary模式的总结）
        self.memory = self._create_memory() # 创建对应类型的记忆实例

    def _get_llm(self):
        """获取 LLM 实例"""
        if DASHSCOPE_API_KEY:
            # 初始化阿里云百炼的通义千问模型
            return ChatTongyi(
                model=MODEL_NAME,  # 注意参数名从model_name改为model
                dashscope_api_key=DASHSCOPE_API_KEY,  # 参数名改为dashscope_api_key
                temperature=TEMPERATURE
            )
        else:
            raise ValueError("请配置 DASHSCOPE_API_KEY")

    def _create_memory(self):
        """
        创建会话记忆
        """
        if self.memory_type == "summary":
            # summary模式：需要传入LLM，用于自动总结对话
            return ConversationSummaryMemory(llm=self.llm, return_messages=True)
        else:
            # buffer模式：无需LLM，直接保存所有消息
            return ConversationBufferMemory(return_messages=True)

    def add_message(self, role, content):
        """
        添加消息到记忆
        :param role: 角色，user 或 assistant
        :param content: 消息内容
        """
        if role == "user":
            self.memory.chat_memory.add_user_message(content)
        else:
            self.memory.chat_memory.add_ai_message(content)

    def get_memory(self):
        """
        获取会话记忆
        :return: 会话记忆
        """
        return self.memory

    def clear_memory(self):
        """
        清空会话记忆
        """
        self.memory.clear()

    def switch_type(self, new_type):
        """
        切换记忆类型（保留现有消息）
        :param new_type: 新的记忆类型，buffer 或 summary
        """
        if new_type == self.memory_type:
            return
        old_messages = self.get_history()
        self.memory_type = new_type
        self.memory = self._create_memory()
        for msg in old_messages:
            self.add_message(msg["role"], msg["content"])

    def get_history(self):
        """
        获取历史对话
        :return: 历史对话列表
        """
        messages = []
        for msg in self.memory.chat_memory.messages:
            if isinstance(msg, HumanMessage):
                # 转换为自定义格式：{"role": "user", "content": 内容}
                messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                messages.append({"role": "assistant", "content": msg.content})
        return messages
