from core.rag.vector_store import VectorStore
from config.settings import TOP_K

class Retriever:
    """检索器封装"""
    
    def __init__(self):
        """初始化检索器"""
        self.vector_store = VectorStore()
        # 初始化检索器，指定返回TOP_K个相似文档
        self.retriever = self.vector_store.get_retriever(k=TOP_K)
    
    def retrieve(self, query):
        """
        根据查询检索相关文档
        :param query: 查询文本
        :return: 相关文档列表
        """
        try:
            docs = self.retriever.invoke(query)
            return docs
        except Exception as e:
            raise Exception(f"检索失败: {str(e)}")
    
    def add_document(self, documents):
        """
        添加文档到向量库
        :param documents: 分块后的文档列表
        :return: 文档ID列表
        """
        return self.vector_store.add_documents(documents)
    
    def delete_document(self, filename):
        """
        删除指定文档
        :param filename: 文件名
        :return: 是否删除成功
        """
        return self.vector_store.delete_document(filename)
    
    def get_document_list(self):
        """
        获取已上传的文档列表
        :return: 文档名列表
        """
        return self.vector_store.get_document_list()
