from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP

class TextSplitter:
    """文本分块器"""
    
    @staticmethod # 静态方法
    def split_text(documents):
        """
        对文档进行分块
        :param documents: 文档对象列表
        :return: 分块后的文档列表
        """
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                length_function=len,
                add_start_index=True
            )
            
            chunks = splitter.split_documents(documents)
            return chunks
        except Exception as e:
            raise Exception(f"文本分块失败: {str(e)}")
