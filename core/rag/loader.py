from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
import os

class DocumentLoader:
    """文档加载器，支持多种格式"""
    
    @staticmethod
    def load_document(file_path):
        """
        加载文档
        :param file_path: 文件路径
        :return: 文档对象列表
        """
        # 获取文件后缀（小写），判断格式
        ext = os.path.splitext(file_path)[1].lower()
        
        try:
            # 根据后缀选择对应的加载器
            if ext == '.pdf':
                loader = PyPDFLoader(file_path)
            elif ext == '.docx':
                loader = Docx2txtLoader(file_path)
            elif ext == '.txt':
                loader = TextLoader(file_path, encoding='utf-8')
            elif ext == '.md':
                loader = UnstructuredMarkdownLoader(file_path)
            else:
                raise ValueError(f"不支持的文件格式: {ext}")
            
            documents = loader.load()
            # 为每个文档添加元数据
            for doc in documents:
                doc.metadata['filename'] = os.path.basename(file_path)
            
            return documents
        except Exception as e:
            raise Exception(f"文档加载失败: {str(e)}")
