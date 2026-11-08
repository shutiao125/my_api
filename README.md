# my_api · 自研轻量中文大模型（记账 NER）

一个**从零开始**的中文大模型全流程项目：自训练分词器 → MLM 预训练 → NER 微调 → FastAPI 服务化 → Docker/Render 部署。全程不依赖任何预训练模型（无 BERT/ERNIE）与第三方推理 API。

模型能力：把「昨天打车花了35块」这类自然语言解析成结构化账单（金额 / 动作 / 分类 / 日期），为「快记账！」App 提供智能记账能力（客户端见 [ios 仓库](https://github.com/shutiao125/ios)）。

## 仓库内容

| 文件 | 说明 |
|---|---|
| `app.py` | 模型结构定义（手写 Transformer）+ 推理与后处理 + FastAPI 服务 |
| `chinese_bpe_tokenizer.json` | 自训练 **2 万词中文 BPE 分词器** |
| `my_ner_model_final.pt` | 微调后的 NER 模型权重（约 2600 万参数 / 99.6 MiB） |
| `Dockerfile` | CPU 版镜像，适配 Render 部署 |
| `requirements.txt` | 依赖清单 |

## 技术栈

| 层次 | 技术 |
|---|---|
| 语言 | Python 3.11 |
| 深度学习 | PyTorch（6 层 Transformer Encoder / 384 维 / 6 头 / 上下文 256） |
| 分词 | HuggingFace tokenizers（加载自训练 BPE 词表） |
| 服务框架 | FastAPI + Uvicorn（ASGI）+ Pydantic（参数校验） |
| 部署 | Docker / Render |

## 模型解释：MiniEncoder

结构在 [app.py](app.py) 中手写实现，未借助 transformers 等库：

| 组件 | 说明 |
|---|---|
| 骨干 | 6 层 Transformer Encoder，`n_embd=384`、`n_head=6`、block_size=256 |
| 注意力 | 双向自注意力，`F.scaled_dot_product_attention` + padding mask |
| 输出头 | MLM 阶段用 `lm_head`（词表投影），NER 阶段换 `ner_head`（5 分类） |
| 标注体系 | `O / B-AMOUNT / I-AMOUNT / B-ACTION / I-ACTION` |

推理链路的工程处理：

- 先用正则剥离日期短语再送模型，避免「25号」里的数字被误标为金额
- 模型漏标金额时，按「50块 / ¥35」等模式正则兜底
- 分类由 ACTION 关键词规则映射（餐饮 / 购物 / 交通 / 居住 / 医疗 / 娱乐），再从原文二次兜底
- CPU 单线程推理，规避 Render 免费容器多线程死锁

## 训练过程

每一步的产物都独立可复用：

1. **自训练分词器**：中文语料上训练 2 万词 BPE，得到 `chinese_bpe_tokenizer.json`
2. **MLM 预训练**：随机遮盖 token，模型还原原词（`MiniEncoderForMLM`），让模型学会中文语义表示
3. **NER 微调**：冻结语义骨干、接 5 分类标注头（`MiniEncoderForNER`），在标注好的记账语料上微调，得到 `my_ner_model_final.pt`

> 训练脚本不在本仓库，本仓库承载模型定义与线上服务。

## 部署

### 本地运行

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

模型与分词器在启动时自动加载，日志出现 `✅ 模型加载成功` 即就绪。

### Docker

```bash
docker build -t my-api .
docker run -p 8000:8000 my-api
```

### Render（免服务器）

仓库自带 Dockerfile（CPU 版 PyTorch，避免拉取 CUDA 依赖），Render 新建 Web Service 连接本仓库即可自动部署；免费实例 15 分钟无流量会休眠，可用定时任务周期请求 `/health` 保活。

## 调用

核心接口只有一个：`POST /api/extract`（模型推理入口）。

```bash
curl -X POST http://localhost:8000/api/extract \
  -H "Content-Type: application/json" \
  -d '{"text": "昨天午饭吃了35块"}'
```

响应：

```json
{
  "code": 200,
  "msg": "success",
  "data": {"amount": "35", "action": "午饭吃", "category": "餐饮", "date": "2026-11-08"}
}
```

更多调用示例：

| 输入文本 | 返回 `data` |
|---|---|
| `今天打车上班花了23块` | `{"amount":"23","action":"打车上班","category":"交通","date":"2026-11-09"}` |
| `10月25号买了一件外套299元` | `{"amount":"299","action":"买了一件外套","category":"购物","date":"2026-10-25"}` |
| `昨天看电影花了80块` | `{"amount":"80","action":"看电影","category":"娱乐","date":"2026-11-08"}` |

> 示例中的 `date` 以 2026-11-09 为「今天」。日期支持「10月25号 / 2026-10-25 / 昨天 / 明天」，识别不出返回 `null`（前端兜底今天）。
