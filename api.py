from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from core.rag.loader import DocumentLoader
from core.rag.splitter import TextSplitter
from core.rag.retriever import Retriever
from core.agent.agent import Agent
from core.memory.conversation import ConversationMemory
from utils.file_utils import FileUtils
from utils.logger import get_logger
import asyncio
import os
import json

logger = get_logger(__name__)

retriever = Retriever()
agent = Agent()
memory = ConversationMemory()

app = FastAPI(title="智能文档AI助手 API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
async def health_check():
    """健康检查接口"""
    return {"status": "ok", "message": "智能文档AI助手 API 运行正常"}

@app.post("/api/documents/upload")
async def upload_documents(files: list[UploadFile] = File(...)):
    """上传文档到知识库"""
    results = []
    existing_docs = retriever.get_document_list()
    
    for uploaded_file in files:
        try:
            if uploaded_file.filename in existing_docs:
                results.append({
                    "filename": uploaded_file.filename,
                    "status": "skipped",
                    "message": "文件已存在，跳过"
                })
                continue
            
            file_path = FileUtils.save_uploaded_file(uploaded_file)
            documents = DocumentLoader.load_document(file_path)
            chunks = TextSplitter.split_text(documents)
            retriever.add_document(chunks)
            
            results.append({
                "filename": uploaded_file.filename,
                "status": "success",
                "message": "上传成功"
            })
            logger.info(f"文件 '{uploaded_file.filename}' 上传成功")
            
        except Exception as e:
            results.append({
                "filename": uploaded_file.filename,
                "status": "error",
                "message": str(e)
            })
            logger.error(f"文件 '{uploaded_file.filename}' 上传失败: {str(e)}")
    
    return {"results": results}

@app.get("/api/documents/list")
async def get_document_list():
    """获取已上传文档列表"""
    try:
        documents = retriever.get_document_list()
        return {"documents": documents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取文档列表失败: {str(e)}")

@app.delete("/api/documents/{filename}")
async def delete_document(filename: str):
    """删除指定文档"""
    try:
        retriever.delete_document(filename)
        FileUtils.delete_file(filename)
        logger.info(f"文档 '{filename}' 删除成功")
        return {"status": "success", "message": f"文档 '{filename}' 删除成功"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除文档失败: {str(e)}")

@app.post("/api/chat/stream")
async def chat_stream(message: dict):
    """对话接口（流式响应）"""
    try:
        user_input = message.get("message", "")
        if not user_input:
            raise HTTPException(status_code=400, detail="消息内容不能为空")
        
        memory.add_message("user", user_input)
        
        async def generate():
            full_response = ""
            for event in agent.stream(user_input):
                event_type = event.get("type", "content")

                if event_type == "content":
                    content = event.get("content", "")
                    full_response += content
                    yield f"data: {json.dumps({'type': 'content', 'content': content})}\n\n"
                elif event_type == "tool_call":
                    yield f"data: {json.dumps({'type': 'tool_call', 'tool': event.get('tool'), 'label': event.get('label'), 'args': event.get('args')})}\n\n"
                elif event_type == "tool_result":
                    yield f"data: {json.dumps({'type': 'tool_result', 'tool': event.get('tool'), 'label': event.get('label'), 'preview': event.get('preview')})}\n\n"
                elif event_type == "end":
                    break

                await asyncio.sleep(0.03)

            memory.add_message("assistant", full_response)
            yield 'data: {"type": "end"}\n\n'

            logger.info(f"用户输入: {user_input}")
            logger.info(f"Agent 响应: {full_response}")
        
        headers = {
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        }
        return StreamingResponse(generate(), headers=headers, media_type="text/event-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"对话失败: {str(e)}")

@app.post("/api/chat/clear")
async def clear_conversation():
    """清空会话历史"""
    try:
        memory.clear_memory()
        logger.info("会话历史已清空")
        return {"status": "success", "message": "会话历史已清空"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清空会话失败: {str(e)}")

@app.get("/api/chat/history")
async def get_conversation_history():
    """获取会话历史"""
    try:
        history = memory.get_history()
        return {"history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取会话历史失败: {str(e)}")

@app.get("/")
async def serve_index():
    """提供前端页面"""
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)