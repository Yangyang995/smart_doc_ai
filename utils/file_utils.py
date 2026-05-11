import os
from config.settings import DATA_DIR

class FileUtils:
    """文件处理工具"""
    
    @staticmethod
    def save_uploaded_file(uploaded_file):
        """
        保存上传的文件
        :param uploaded_file: 上传的文件对象
        :return: 保存后的文件路径
        """
        try:
            # 确保数据目录存在
            if not os.path.exists(DATA_DIR):
                os.makedirs(DATA_DIR)
            
            # 构建保存路径
            file_path = os.path.join(DATA_DIR, uploaded_file.name)
            
            # 保存文件
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            return file_path
        except Exception as e:
            raise Exception(f"保存文件失败: {str(e)}")
    
    @staticmethod
    def get_file_list():
        """
        获取数据目录中的文件列表
        :return: 文件列表
        """
        try:
            if not os.path.exists(DATA_DIR):
                return []
            return os.listdir(DATA_DIR)
        except Exception as e:
            raise Exception(f"获取文件列表失败: {str(e)}")
    
    @staticmethod
    def delete_file(file_name):
        """
        删除文件
        :param file_name: 文件名
        :return: 是否删除成功
        """
        try:
            file_path = os.path.join(DATA_DIR, file_name)
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
            return False
        except Exception as e:
            raise Exception(f"删除文件失败: {str(e)}")
