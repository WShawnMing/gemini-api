# Gemini Web API

基于 FastAPI 的 RESTful API 服务，提供 Google Gemini 的完整功能。

## 功能特性

- ✅ 单次对话（无历史记录）
- ✅ 多轮会话（保存历史记录）
- ✅ 文件上传支持（图片、文档）
- ✅ Gemini Gems 管理（系统提示词）
- ✅ 多模型支持（gemini-3.0-pro, gemini-2.5-pro, gemini-2.5-flash）
- ✅ 候选回复选择
- ✅ OpenAI 兼容格式（/v1/chat/completions）
- ✅ 自动 Cookie 刷新
- ✅ 完整的错误处理
- ✅ 上下文管理（多轮对话记忆）

## 安装

```bash
pip install -r requirements.txt
```

## 配置

### 方式一：使用配置文件（推荐）

1. 复制示例配置文件：
```bash
cp config.yaml.example config.yaml
```

2. 编辑 `config.yaml`，填入你的 Cookie 值：
```yaml
gemini:
  secure_1psid: "你的__Secure-1PSID值"
  secure_1psidts: "你的__Secure-1PSIDTS值"
  proxy: null

server:
  host: "0.0.0.0"
  port: 8000
  log_level: "info"

client:
  timeout: 30
  auto_close: false
  close_delay: 300
  auto_refresh: true
```

**注意**：也支持 `config.json` 格式，但推荐使用 YAML。

### 方式二：环境变量

```bash
export GEMINI_1PSID="你的cookie值"
export GEMINI_1PSIDTS="你的cookie值"
```

## 运行

```bash
./start.sh
```

或者：

```bash
python api.py
```

服务将在 `http://0.0.0.0:8000` 启动。

## API 文档

启动服务后，访问以下地址查看交互式 API 文档：

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 主要 API 端点

### OpenAI 兼容格式

```bash
POST /v1/chat/completions
Content-Type: application/json

{
  "model": "gemini-2.5-flash",
  "messages": [
    {"role": "user", "content": "你好"}
  ]
}
```

### 单次对话

```bash
POST /chat
Content-Type: application/json

{
  "message": "你好",
  "model": "gemini-2.5-flash"
}
```

### 会话管理

```bash
# 创建会话
POST /chat/session

# 发送消息
POST /chat/session/{session_id}

# 删除会话
DELETE /chat/session/{session_id}
```

## Chatbox 配置

在 Chatbox 中配置：

- **API 端点**: `http://localhost:8000/v1/chat/completions`
- **API Key**: `gemini-webapi`（任意值）
- **模型**: `gemini-2.5-flash`

## 支持的模型

- `gemini-3.0-pro` - Gemini 3.0 Pro
- `gemini-2.5-pro` - Gemini 2.5 Pro
- `gemini-2.5-flash` - Gemini 2.5 Flash（快速）
- `unspecified` - 默认模型

## 注意事项

1. **Cookie 管理**：Cookie 会自动刷新，但建议定期检查服务状态
2. **会话存储**：会话存储在内存中，重启服务会丢失所有会话
3. **配置文件**：`config.json` 包含敏感信息，已添加到 `.gitignore`
4. **生产环境**：建议添加认证、限流和日志记录机制

## 许可证

本项目基于 AGPL-3.0 许可证。
