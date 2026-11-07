FROM python:3.11-slim

WORKDIR /app

# 安装 CPU 版 PyTorch，避免拉取巨大的 CUDA 依赖
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

# 复制并安装其他依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码和模型文件
COPY . .

# Render 会注入 PORT 环境变量，需在启动命令中使用
# 使用 $PORT 确保服务监听在 Render 期望的端口上
CMD uvicorn app:app --host 0.0.0.0 --port $PORT