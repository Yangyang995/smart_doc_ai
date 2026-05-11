from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import DashScopeEmbeddings
from config.settings import CHROMA_DB_DIR, EMBEDDING_MODEL, DASHSCOPE_API_KEY, OPENAI_API_KEY
import os

class VectorStore:
    """向量库管理"""
    
    def __init__(self):
        """初始化向量库"""
        self.embeddings = self._get_embeddings() # 获取嵌入模型实例
        self.vector_store = Chroma(
            persist_directory=CHROMA_DB_DIR,     # Chroma数据持久化目录（本地路径）
            embedding_function=self.embeddings   # 绑定嵌入模型（将文本转为向量）
        )
    
    def _get_embeddings(self):
        """
        获取嵌入模型
        :return: 嵌入模型实例
        """
        if DASHSCOPE_API_KEY:
            return DashScopeEmbeddings(model="text-embedding-v2", dashscope_api_key=DASHSCOPE_API_KEY)
        elif OPENAI_API_KEY:
            return OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
        else:
            raise ValueError("请配置 DASHSCOPE_API_KEY 或 OPENAI_API_KEY")
    
    def add_documents(self, documents):
        """
        添加文档到向量库
        :param documents: 分块后的文档列表
        :return: 文档ID列表
        """
        try:
            # 入库：将文档文本转为向量并存储，返回ID列表
            ids = self.vector_store.add_documents(documents)
            self.vector_store.persist() # 持久化到本地（Chroma需手动触发）
            return ids
        except Exception as e:
            raise Exception(f"添加文档失败: {str(e)}")
    
    def delete_document(self, filename):
        """
        删除指定文档
        :param filename: 文件名
        :return: 是否删除成功
        """
        try:
            # 获取所有文档
            all_docs = self.vector_store.get()
            # 找到指定文件名的文档ID
            ids_to_delete = []
            # 遍历元数据，找到所有文件名匹配的文档ID
            for i, metadata in enumerate(all_docs['metadatas']):
                if metadata.get('filename') == filename:
                    ids_to_delete.append(all_docs['ids'][i])

            # 有匹配ID则删除并持久化
            if ids_to_delete:
                self.vector_store.delete(ids=ids_to_delete)
                self.vector_store.persist()
                return True
            return False
        except Exception as e:
            raise Exception(f"删除文档失败: {str(e)}")
    
    def get_retriever(self, k=3):
        """
        获取检索器
        :param k: 返回的相关文档数量
        :return: 检索器实例
        """
        return self.vector_store.as_retriever(search_kwargs={"k": k})
    
    def get_document_list(self):
        """
        获取已上传的文档列表
        :return: 文档名列表
        """
        try:
            all_docs = self.vector_store.get()
            filenames = set()  # 用集合去重（一个文件可能被分多个块）
            for metadata in all_docs['metadatas']:
                if 'filename' in metadata:
                    filenames.add(metadata['filename'])
            return list(filenames)
        except Exception as e:
            raise Exception(f"获取文档列表失败: {str(e)}")
