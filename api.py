"""
Gemini Web API 服务
基于 FastAPI 的 RESTful API，提供 Google Gemini 的完整功能
"""
import asyncio
import os
import tempfile
import shutil
import time
import uuid
import json
import hashlib
from typing import Optional, List, Dict, Any
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
import uvicorn

from gemini_webapi import GeminiClient, set_log_level
from gemini_webapi.constants import Model

# ==================== 日志配置 ====================
# 设置 gemini_webapi 日志级别为 INFO，减少 DEBUG 噪音
# 可选值: DEBUG, INFO, WARNING, ERROR, CRITICAL
set_log_level("INFO")

# ==================== 配置 ====================
def load_config():
    """加载配置文件"""
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    
    # 优先使用环境变量
    secure_1psid = os.getenv("GEMINI_1PSID")
    secure_1psidts = os.getenv("GEMINI_1PSIDTS")
    
    # 如果环境变量不存在，尝试从配置文件读取
    if not secure_1psid or not secure_1psidts:
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    secure_1psid = secure_1psid or config.get("gemini", {}).get("secure_1psid", "")
                    secure_1psidts = secure_1psidts or config.get("gemini", {}).get("secure_1psidts", "")
                    proxy = config.get("gemini", {}).get("proxy")
            except Exception as e:
                print(f"⚠️ 读取配置文件失败: {e}")
                secure_1psid = secure_1psid or ""
                secure_1psidts = secure_1psidts or ""
                proxy = None
        else:
            print(f"⚠️ 配置文件不存在: {config_path}")
            print(f"💡 请创建 config.json 或设置环境变量 GEMINI_1PSID 和 GEMINI_1PSIDTS")
            secure_1psid = secure_1psid or ""
            secure_1psidts = secure_1psidts or ""
            proxy = None
    else:
        proxy = None
    
    # 读取服务器配置
    server_config = {
        "host": "0.0.0.0",
        "port": 8000,
        "log_level": "info"
    }
    client_config = {
        "timeout": 30,
        "auto_close": False,
        "close_delay": 300,
        "auto_refresh": True
    }
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                server_config.update(config.get("server", {}))
                client_config.update(config.get("client", {}))
                if proxy is None:
                    proxy = config.get("gemini", {}).get("proxy")
        except:
            pass
    
    return {
        "secure_1psid": secure_1psid,
        "secure_1psidts": secure_1psidts,
        "proxy": proxy,
        "server": server_config,
        "client": client_config
    }

# 加载配置
config = load_config()
Secure_1PSID = config["secure_1psid"]
Secure_1PSIDTS = config["secure_1psidts"]
Proxy = config["proxy"]
ServerConfig = config["server"]
ClientConfig = config["client"]

# ==================== 全局变量 ====================
client: Optional[GeminiClient] = None
chat_sessions: Dict[str, Any] = {}  # 存储会话对象（用于 /chat/session 端点）
openai_sessions: Dict[str, Any] = {}  # 存储 OpenAI 格式的会话（用于 /v1/chat/completions）


# ==================== 生命周期管理 ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化客户端
    global client
    try:
        client = GeminiClient(Secure_1PSID, Secure_1PSIDTS, proxy=Proxy)
        await client.init(
            timeout=ClientConfig["timeout"],
            auto_close=ClientConfig["auto_close"],
            close_delay=ClientConfig["close_delay"],
            auto_refresh=ClientConfig["auto_refresh"]
        )
        print("✅ Gemini 客户端初始化成功")
    except Exception as e:
        print(f"❌ Gemini 客户端初始化失败: {e}")
        raise
    
    yield
    
    # 关闭时清理资源
    if client:
        try:
            await client.close()
            print("✅ Gemini 客户端已关闭")
        except Exception as e:
            print(f"⚠️ 关闭客户端时出错: {e}")


# ==================== FastAPI 应用 ====================
app = FastAPI(
    title="Gemini Web API",
    description="基于 gemini-webapi 的 RESTful API 服务",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== 错误处理 ====================
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """处理请求验证错误"""
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "message": "请求格式错误",
                "type": "invalid_request_error",
                "param": None,
                "code": None,
                "details": exc.errors()
            }
        }
    )



# ==================== 请求/响应模型 ====================
class ChatRequest(BaseModel):
    """单次对话请求"""
    message: str = Field(..., description="要发送的消息")
    model: Optional[str] = Field(None, description="模型名称，如 gemini-2.5-pro")
    gem: Optional[str] = Field(None, description="Gemini Gem ID")


# OpenAI 兼容格式的请求/响应模型
class OpenAIMessage(BaseModel):
    """OpenAI 消息格式"""
    role: str = Field(..., description="角色: user, assistant, system")
    content: Any = Field(..., description="消息内容（可以是字符串或列表）")
    
    def get_text_content(self) -> str:
        """获取文本内容"""
        if isinstance(self.content, str):
            return self.content
        elif isinstance(self.content, list):
            # 处理内容块列表
            text_parts = []
            for item in self.content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif "text" in item:
                        text_parts.append(item["text"])
                elif isinstance(item, str):
                    text_parts.append(item)
            return " ".join(text_parts)
        else:
            return str(self.content)


class OpenAICompletionRequest(BaseModel):
    """OpenAI Chat Completions 请求格式"""
    model: Optional[str] = Field(default="gemini-2.5-flash", description="模型名称")
    messages: List[OpenAIMessage] = Field(..., description="消息列表")
    temperature: Optional[float] = Field(default=None, description="温度参数")
    max_tokens: Optional[int] = Field(default=None, description="最大token数")
    stream: Optional[bool] = Field(default=False, description="是否流式输出")
    
    class Config:
        # 允许额外字段，提高兼容性
        extra = "allow"


class OpenAIChoice(BaseModel):
    """OpenAI Choice 格式"""
    index: int
    message: OpenAIMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    """Token 使用统计"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class OpenAICompletionResponse(BaseModel):
    """OpenAI Chat Completions 响应格式"""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[OpenAIChoice]
    usage: Optional[Usage] = None


class ChatResponse(BaseModel):
    """对话响应"""
    text: str = Field(..., description="生成的文本")
    images: Optional[List[Dict[str, Any]]] = Field(None, description="图片列表")
    thoughts: Optional[str] = Field(None, description="模型的思考过程")
    candidates_count: Optional[int] = Field(None, description="候选回复数量")
    metadata: Optional[Dict[str, Any]] = Field(None, description="会话元数据")


class SessionCreateRequest(BaseModel):
    """创建会话请求"""
    model: Optional[str] = Field(None, description="模型名称")
    gem: Optional[str] = Field(None, description="Gemini Gem ID")
    metadata: Optional[Dict[str, Any]] = Field(None, description="恢复会话的元数据")


class SessionMessageRequest(BaseModel):
    """会话消息请求"""
    message: str = Field(..., description="要发送的消息")


class SessionResponse(BaseModel):
    """会话响应"""
    session_id: str = Field(..., description="会话ID")
    message: str = Field(..., description="响应消息")


class GemCreateRequest(BaseModel):
    """创建 Gem 请求"""
    name: str = Field(..., description="Gem 名称")
    prompt: str = Field(..., description="系统提示词")
    description: Optional[str] = Field(None, description="Gem 描述")


class GemUpdateRequest(BaseModel):
    """更新 Gem 请求"""
    name: str = Field(..., description="Gem 名称")
    prompt: str = Field(..., description="系统提示词")
    description: Optional[str] = Field(None, description="Gem 描述")


# 生命周期事件已移至 lifespan 上下文管理器


# ==================== 工具函数 ====================
def format_response(response) -> ChatResponse:
    """格式化响应对象"""
    images = None
    if response.images:
        images = []
        for img in response.images:
            img_dict = {
                "title": getattr(img, 'title', None),
                "url": getattr(img, 'url', None),
                "alt": getattr(img, 'alt', None),
            }
            # 添加图片类型
            img_type = type(img).__name__
            img_dict["type"] = img_type
            images.append(img_dict)
    
    metadata = None
    if hasattr(response, 'metadata'):
        metadata = response.metadata
    
    return ChatResponse(
        text=response.text,
        images=images,
        thoughts=getattr(response, 'thoughts', None),
        candidates_count=len(response.candidates) if hasattr(response, 'candidates') else None,
        metadata=metadata
    )


# ==================== API 端点 ====================
@app.get("/")
async def root():
    """根路径"""
    return {
        "service": "Gemini Web API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "ok",
        "client_initialized": client is not None,
        "active_sessions": len(chat_sessions)
    }


@app.post("/v1/chat/completions/debug")
async def chat_completions_debug(request: OpenAICompletionRequest):
    """
    调试端点 - 返回详细的响应信息，用于诊断 Chatbox 问题
    """
    if not client:
        raise HTTPException(status_code=503, detail="客户端未初始化，请稍后重试")
    
    try:
        import time
        import uuid
        
        # 将消息列表转换为单个消息（取最后一条用户消息）
        user_messages = [msg for msg in request.messages if msg.role == "user"]
        if not user_messages:
            return {
                "error": "至少需要一条用户消息"
            }
        
        message_content = user_messages[-1].get_text_content()
        
        # 调用 Gemini API
        kwargs = {}
        if request.model:
            kwargs["model"] = request.model
        
        response = await client.generate_content(message_content, **kwargs)
        response_text = response.text if response and hasattr(response, 'text') and response.text else ""
        
        # 估算 token
        prompt_tokens = len(message_content.encode('utf-8')) // 2
        completion_tokens = len(response_text.encode('utf-8')) // 2 if response_text else 0
        total_tokens = prompt_tokens + completion_tokens
        model_name = request.model or "gemini-2.5-flash"
        
        # 构建响应
        response_data = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:29]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text if response_text else "无内容"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }
        }
        
        # 返回调试信息
        return {
            "debug": {
                "response_text_length": len(response_text),
                "response_text_type": type(response_text).__name__,
                "response_text_preview": response_text[:100] if response_text else "空",
                "has_content": bool(response_text),
                "usage_calculated": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens
                }
            },
            "response": response_data,
            "raw_response_attributes": {
                "has_text": hasattr(response, 'text'),
                "text_value": str(response.text) if hasattr(response, 'text') else "N/A",
                "has_candidates": hasattr(response, 'candidates'),
                "candidates_count": len(response.candidates) if hasattr(response, 'candidates') else 0
            }
        }
    except Exception as e:
        return {
            "error": str(e),
            "error_type": type(e).__name__
        }


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(request: OpenAICompletionRequest, http_request: Request):
    """
    OpenAI 兼容格式的对话接口
    支持标准的 OpenAI API 格式，可用于 Chatbox 等客户端
    支持流式和非流式响应
    支持上下文管理（多轮对话）
    """
    if not client:
        raise HTTPException(status_code=503, detail="客户端未初始化，请稍后重试")
    
    try:
        import time
        import uuid
        import hashlib
        
        # 检查消息列表
        if not request.messages:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": "消息列表不能为空",
                        "type": "invalid_request_error"
                    }
                }
            )
        
        # 获取最后一条用户消息
        user_messages = [msg for msg in request.messages if msg.role == "user"]
        if not user_messages:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": "至少需要一条用户消息",
                        "type": "invalid_request_error"
                    }
                }
            )
        
        # 使用最后一条用户消息作为当前输入
        current_user_message = user_messages[-1].get_text_content()
        
        # 判断是否需要使用会话（如果消息历史中有 assistant 消息，说明是继续对话）
        has_assistant_message = any(msg.role == "assistant" for msg in request.messages)
        use_session = has_assistant_message and len(request.messages) > 1
        
        # 获取或创建会话
        session_key = None
        chat = None
        
        if use_session:
            # 尝试从请求头获取会话ID（如果客户端提供了）
            session_id_header = http_request.headers.get("X-Session-ID")
            
            if session_id_header:
                session_key = session_id_header
            else:
                # 如果没有提供会话ID，基于模型生成一个固定的会话键
                # 这样同一个模型的对话会使用同一个会话
                model_name = request.model or "gemini-2.5-flash"
                session_key = f"openai_session_{model_name}"
            
            # 检查是否已有会话
            if session_key in openai_sessions:
                chat = openai_sessions[session_key]["chat"]
                # 验证会话是否仍然有效
                try:
                    # 尝试访问会话属性来验证
                    _ = chat.metadata if hasattr(chat, 'metadata') else None
                except:
                    # 会话已失效，创建新会话
                    chat = None
                    del openai_sessions[session_key]
            
            # 检查会话中已有的消息数量
            existing_message_count = openai_sessions.get(session_key, {}).get("message_count", 0) if session_key in openai_sessions else 0
            history_user_messages = [m for m in request.messages[:-1] if m.role == "user"]
            history_count = len(history_user_messages)
            
            if not chat:
                # 创建新会话
                kwargs = {}
                if request.model:
                    kwargs["model"] = request.model
                
                chat = client.start_chat(**kwargs)
                
                # 如果是新会话且有历史消息，需要将历史消息发送给 Gemini
                # 这样 Gemini 才能记住之前的对话
                if history_count > 0:
                    system_messages = [m for m in request.messages if m.role == "system"]
                    system_prompt = "\n".join([m.get_text_content() for m in system_messages]) if system_messages else None
                    
                    # 遍历历史消息，成对发送 user-assistant
                    i = 0
                    user_msg_index = 0
                    while i < len(request.messages) - 1:  # 排除最后一条用户消息
                        msg = request.messages[i]
                        if msg.role == "user":
                            # 发送用户消息
                            user_content = msg.get_text_content()
                            # 如果是第一条用户消息且有系统消息，添加系统提示
                            if user_msg_index == 0 and system_prompt:
                                user_content = f"{system_prompt}\n\n{user_content}"
                            
                            # 发送消息，Gemini 会生成回复
                            await chat.send_message(user_content)
                            user_msg_index += 1
                            
                            # 跳过下一条 assistant 消息（因为 Gemini 已经生成了）
                            if i + 1 < len(request.messages) - 1 and request.messages[i + 1].role == "assistant":
                                i += 2
                            else:
                                i += 1
                        else:
                            i += 1
                
                openai_sessions[session_key] = {
                    "chat": chat,
                    "model": request.model or "gemini-2.5-flash",
                    "created": int(time.time()),
                    "message_count": history_count
                }
            elif existing_message_count < history_count:
                # 会话存在但消息数量不足，需要补充历史消息
                # 这种情况不应该发生，但如果发生了，补充缺失的消息
                missing_count = history_count - existing_message_count
                # 这里简化处理：重新发送所有历史消息（实际应该只发送缺失的部分）
                # 为了简化，我们暂时跳过这个复杂逻辑
                pass
            
            # 更新消息计数（当前消息会在下面发送）
            if session_key in openai_sessions:
                openai_sessions[session_key]["message_count"] = history_count + 1
        
        # 准备消息内容
        message_content = current_user_message
        
        # 调用 Gemini API
        if use_session and chat:
            # 使用会话发送消息（保持上下文）
            # 注意：如果会话是新创建的，历史消息已经在上面发送过了
            # 这里只需要发送当前用户消息
            response = await chat.send_message(message_content)
        else:
            # 单次对话（无上下文）
            # 如果有系统消息，添加到提示中
            system_messages = [msg for msg in request.messages if msg.role == "system"]
            if system_messages:
                system_prompt = "\n".join([msg.get_text_content() for msg in system_messages])
                message_content = f"{system_prompt}\n\n{message_content}"
            
            kwargs = {}
            if request.model:
                kwargs["model"] = request.model
            response = await client.generate_content(message_content, **kwargs)
        
        # 确保响应文本不为空
        response_text = response.text if response and hasattr(response, 'text') and response.text else ""
        
        # 如果响应为空，尝试从其他属性获取
        if not response_text:
            # 尝试从 candidates 获取
            if hasattr(response, 'candidates') and response.candidates:
                first_candidate = response.candidates[0]
                if hasattr(first_candidate, 'text'):
                    response_text = first_candidate.text
                elif hasattr(first_candidate, 'content'):
                    response_text = str(first_candidate.content)
        
        # 如果仍然为空，返回默认消息
        if not response_text:
            response_text = "抱歉，未能生成回复内容。"
        
        # 估算 token 数量（简单估算：中文字符按2个token，英文按1个token）
        prompt_tokens = len(message_content.encode('utf-8')) // 2  # 简单估算
        completion_tokens = len(response_text.encode('utf-8')) // 2
        total_tokens = prompt_tokens + completion_tokens
        
        # 确保 model 字段不为空
        model_name = request.model or "gemini-2.5-flash"
        response_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
        created_time = int(time.time())
        
        # 检查是否需要流式响应
        if request.stream:
            # 流式响应（SSE 格式）
            import json as json_lib
            async def generate_stream():
                # 发送初始数据
                initial_data = {
                    'id': response_id,
                    'object': 'chat.completion.chunk',
                    'created': created_time,
                    'model': model_name,
                    'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''}, 'finish_reason': None}]
                }
                yield f"data: {json_lib.dumps(initial_data, ensure_ascii=False)}\n\n"
                
                # 逐字符发送内容（模拟流式）
                for char in response_text:
                    chunk_data = {
                        'id': response_id,
                        'object': 'chat.completion.chunk',
                        'created': created_time,
                        'model': model_name,
                        'choices': [{'index': 0, 'delta': {'content': char}, 'finish_reason': None}]
                    }
                    yield f"data: {json_lib.dumps(chunk_data, ensure_ascii=False)}\n\n"
                
                # 发送结束标记
                final_data = {
                    'id': response_id,
                    'object': 'chat.completion.chunk',
                    'created': created_time,
                    'model': model_name,
                    'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]
                }
                yield f"data: {json_lib.dumps(final_data, ensure_ascii=False)}\n\n"
                
                # 发送 usage
                usage_data = {
                    'id': response_id,
                    'object': 'chat.completion.chunk',
                    'created': created_time,
                    'model': model_name,
                    'choices': [],
                    'usage': {
                        'prompt_tokens': prompt_tokens,
                        'completion_tokens': completion_tokens,
                        'total_tokens': total_tokens
                    }
                }
                yield f"data: {json_lib.dumps(usage_data, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                }
            )
        else:
            # 非流式响应（标准 JSON）
            response_data = {
                "id": response_id,
                "object": "chat.completion",
                "created": created_time,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response_text
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens
                }
            }
            
            # 返回 JSONResponse 确保正确的 Content-Type
            return JSONResponse(
                content=response_data,
                headers={
                    "Content-Type": "application/json; charset=utf-8"
                }
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成内容时出错: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    单次对话接口
    不保存历史记录，每次都是独立对话
    """
    if not client:
        raise HTTPException(status_code=503, detail="客户端未初始化，请稍后重试")
    
    try:
        kwargs = {}
        if request.model:
            kwargs["model"] = request.model
        if request.gem:
            kwargs["gem"] = request.gem
        
        response = await client.generate_content(request.message, **kwargs)
        return format_response(response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成内容时出错: {str(e)}")


@app.post("/chat/with-files", response_model=ChatResponse)
async def chat_with_files(
    message: str = Form(...),
    files: List[UploadFile] = File(None),
    model: Optional[str] = Form(None),
    gem: Optional[str] = Form(None)
):
    """
    带文件上传的单次对话
    支持图片和文档文件
    """
    if not client:
        raise HTTPException(status_code=503, detail="客户端未初始化，请稍后重试")
    
    temp_dir = None
    try:
        file_paths = []
        if files:
            temp_dir = tempfile.mkdtemp()
            for file in files:
                file_path = os.path.join(temp_dir, file.filename)
                with open(file_path, "wb") as f:
                    content = await file.read()
                    f.write(content)
                file_paths.append(file_path)
        
        kwargs = {}
        if model:
            kwargs["model"] = model
        if gem:
            kwargs["gem"] = gem
        if file_paths:
            kwargs["files"] = file_paths
        
        response = await client.generate_content(message, **kwargs)
        return format_response(response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成内容时出错: {str(e)}")
    finally:
        # 清理临时文件
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


@app.post("/chat/session", response_model=SessionResponse)
async def create_session(request: SessionCreateRequest):
    """
    创建新的对话会话
    会话会保存历史记录，支持多轮对话
    """
    if not client:
        raise HTTPException(status_code=503, detail="客户端未初始化，请稍后重试")
    
    try:
        kwargs = {}
        if request.model:
            kwargs["model"] = request.model
        if request.gem:
            kwargs["gem"] = request.gem
        if request.metadata:
            kwargs["metadata"] = request.metadata
        
        chat = client.start_chat(**kwargs)
        session_id = str(id(chat))
        chat_sessions[session_id] = {
            "chat": chat,
            "metadata": chat.metadata if hasattr(chat, 'metadata') else None
        }
        
        return SessionResponse(
            session_id=session_id,
            message="会话创建成功"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建会话时出错: {str(e)}")


@app.post("/chat/session/{session_id}", response_model=ChatResponse)
async def session_message(session_id: str, request: SessionMessageRequest):
    """
    向指定会话发送消息
    会话会保持历史记录
    """
    if not client:
        raise HTTPException(status_code=503, detail="客户端未初始化，请稍后重试")
    
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    try:
        chat = chat_sessions[session_id]["chat"]
        response = await chat.send_message(request.message)
        
        # 更新会话元数据
        if hasattr(response, 'metadata'):
            chat_sessions[session_id]["metadata"] = response.metadata
        
        return format_response(response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"发送消息时出错: {str(e)}")


@app.get("/chat/session/{session_id}/metadata")
async def get_session_metadata(session_id: str):
    """获取会话元数据"""
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    return {
        "session_id": session_id,
        "metadata": chat_sessions[session_id].get("metadata")
    }


@app.delete("/chat/session/{session_id}")
async def delete_session(session_id: str):
    """删除指定的会话"""
    if session_id in chat_sessions:
        del chat_sessions[session_id]
        return {"message": "会话已删除"}
    else:
        raise HTTPException(status_code=404, detail="会话不存在")


@app.get("/chat/sessions")
async def list_sessions():
    """列出所有活跃的会话"""
    sessions = []
    for session_id, session_data in chat_sessions.items():
        sessions.append({
            "session_id": session_id,
            "metadata": session_data.get("metadata")
        })
    
    return {
        "sessions": sessions,
        "count": len(sessions)
    }


@app.post("/chat/session/{session_id}/choose-candidate")
async def choose_candidate(session_id: str, index: int = 0):
    """
    选择会话中的候选回复
    """
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    try:
        chat = chat_sessions[session_id]["chat"]
        candidate = chat.choose_candidate(index=index)
        return {
            "message": "候选回复已选择",
            "candidate_text": candidate.text if hasattr(candidate, 'text') else str(candidate)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"选择候选回复时出错: {str(e)}")


# ==================== Gems 管理 ====================
@app.get("/gems")
async def list_gems(include_hidden: bool = False):
    """
    获取所有 Gems（系统提示词）
    """
    if not client:
        raise HTTPException(status_code=503, detail="客户端未初始化，请稍后重试")
    
    try:
        await client.fetch_gems(include_hidden=include_hidden)
        gems = client.gems
        
        gems_list = []
        for gem in gems:
            gems_list.append({
                "id": gem.id,
                "name": gem.name,
                "description": getattr(gem, 'description', None),
                "predefined": getattr(gem, 'predefined', False)
            })
        
        return {
            "gems": gems_list,
            "count": len(gems_list)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取 Gems 时出错: {str(e)}")


@app.post("/gems")
async def create_gem(request: GemCreateRequest):
    """
    创建自定义 Gem
    """
    if not client:
        raise HTTPException(status_code=503, detail="客户端未初始化，请稍后重试")
    
    try:
        new_gem = await client.create_gem(
            name=request.name,
            prompt=request.prompt,
            description=request.description
        )
        
        return {
            "id": new_gem.id,
            "name": new_gem.name,
            "description": getattr(new_gem, 'description', None),
            "message": "Gem 创建成功"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建 Gem 时出错: {str(e)}")


@app.put("/gems/{gem_id}")
async def update_gem(gem_id: str, request: GemUpdateRequest):
    """
    更新自定义 Gem
    """
    if not client:
        raise HTTPException(status_code=503, detail="客户端未初始化，请稍后重试")
    
    try:
        updated_gem = await client.update_gem(
            gem=gem_id,
            name=request.name,
            prompt=request.prompt,
            description=request.description
        )
        
        return {
            "id": updated_gem.id,
            "name": updated_gem.name,
            "description": getattr(updated_gem, 'description', None),
            "message": "Gem 更新成功"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新 Gem 时出错: {str(e)}")


@app.delete("/gems/{gem_id}")
async def delete_gem(gem_id: str):
    """
    删除自定义 Gem
    """
    if not client:
        raise HTTPException(status_code=503, detail="客户端未初始化，请稍后重试")
    
    try:
        await client.delete_gem(gem_id)
        return {"message": "Gem 删除成功"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除 Gem 时出错: {str(e)}")


# ==================== 模型信息 ====================
@app.get("/models")
async def list_models():
    """获取可用的模型列表"""
    return {
        "models": [
            {
                "id": "unspecified",
                "name": "默认模型",
                "description": "Gemini 默认模型"
            },
            {
                "id": "gemini-3.0-pro",
                "name": "Gemini 3.0 Pro",
                "description": "Gemini 3.0 Pro 模型"
            },
            {
                "id": "gemini-2.5-pro",
                "name": "Gemini 2.5 Pro",
                "description": "Gemini 2.5 Pro 模型"
            },
            {
                "id": "gemini-2.5-flash",
                "name": "Gemini 2.5 Flash",
                "description": "Gemini 2.5 Flash 模型（快速）"
            }
        ]
    }


if __name__ == "__main__":
    # 配置 uvicorn 日志级别
    # 可选值: critical, error, warning, info, debug, trace
    log_level = os.getenv("LOG_LEVEL", ServerConfig.get("log_level", "info")).lower()
    
    uvicorn.run(
        app,
        host=ServerConfig.get("host", "0.0.0.0"),
        port=ServerConfig.get("port", 8000),
        log_level=log_level
    )

