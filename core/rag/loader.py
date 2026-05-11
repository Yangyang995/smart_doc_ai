from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
import os


class DocumentLoader:
    """文档加载器，支持 PDF / DOCX / TXT / MD 格式"""

    @staticmethod
    def load_document(file_path):
        """
        加载文档
        :param file_path: 文件路径
        :return: 文档对象列表
        """
        ext = os.path.splitext(file_path)[1].lower()

        try:
            if ext == '.pdf':
                loader = PyPDFLoader(file_path)
            elif ext == '.docx':
                loader = Docx2txtLoader(file_path)
            elif ext in ('.txt', '.md'):
                loader = TextLoader(file_path, encoding='utf-8')
            else:
                raise ValueError(f"不支持的文件格式: {ext}")

            documents = loader.load()

            filename = os.path.basename(file_path)
            for doc in documents:
                doc.metadata['filename'] = filename

            return documents
        except Exception as e:
            raise Exception(f"文档加载失败 ({os.path.basename(file_path)}): {str(e)}")
