import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import re

# ==========================================
# 0. 容器环境优化
# ==========================================
# 强制限制 PyTorch 使用单线程，防止在 Render 免费容器中因为多线程死锁卡住
torch.set_num_threads(1)

# ==========================================
# 1. 手写模型结构定义
# ==========================================
class BidirectionalAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)

    def forward(self, x, mask=None):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(C, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        attn_mask = mask.view(B, 1, 1, T).expand(B, self.n_head, T, T) if mask is not None else None
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = BidirectionalAttention(n_embd, n_head)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd)
        )

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln_1(x), mask)
        x = x + self.mlp(self.ln_2(x))
        return x

class MiniEncoderForMLM(nn.Module):
    def __init__(self, vocab_size, block_size=256, n_embd=384, n_head=6, n_layer=6):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx, mask=None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        for block in self.blocks:
            x = block(x, mask)
        return self.lm_head(self.ln_f(x))

class MiniEncoderForNER(nn.Module):
    def __init__(self, base_encoder, num_labels=5):
        super().__init__()
        self.encoder = base_encoder
        self.ner_head = nn.Linear(384, num_labels)

    def forward(self, idx, mask=None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.encoder.wte(idx) + self.encoder.wpe(pos)
        for block in self.encoder.blocks:
            x = block(x, mask)
        x = self.encoder.ln_f(x)
        return self.ner_head(x)

# ==========================================
# 2. 初始化 FastAPI 与 跨域配置
# ==========================================
app = FastAPI(title="记账模型 API", description="用于提取账单金额与分类")

# 允许所有来源的跨域请求（App 端请求必备）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 3. 加载模型与 Tokenizer (必须在启动时加载)
# ==========================================
device = torch.device("cpu")
REAL_VOCAB_SIZE = 20000
label_list = ["O", "B-AMOUNT", "I-AMOUNT", "B-ACTION", "I-ACTION"]
id2label = {i: l for i, l in enumerate(label_list)}

print("正在加载模型和分词器...")
# 注意：这里使用的是相对路径，确保这两个文件和 app.py 在同一个目录
try:
    tokenizer = Tokenizer.from_file("chinese_bpe_tokenizer.json")
    base_encoder = MiniEncoderForMLM(vocab_size=REAL_VOCAB_SIZE, block_size=256).to(device)
    ner_model = MiniEncoderForNER(base_encoder).to(device)
    ner_model.load_state_dict(torch.load("my_ner_model_final.pt", map_location=device))
    ner_model.eval()
    print("✅ 模型加载成功！")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    # 在生产环境中，如果模型加载失败，可以在这里抛出异常或记录日志

# ==========================================
# 4. 推理与后处理逻辑
# ==========================================
def extract_bill_logic(text: str):
    input_ids = [tokenizer.token_to_id(t) or tokenizer.token_to_id("[UNK]") for t in text]
    x = torch.tensor([input_ids]).to(device)
    mask = torch.ones_like(x, dtype=torch.bool).to(device)
    
    with torch.no_grad():
        logits = ner_model(x, mask)
        preds = torch.argmax(logits, dim=-1)[0].cpu().numpy()
        
    amount, action = "", ""
    for char, pred_id in zip(text, preds):
        label = id2label.get(pred_id, "O")
        if "AMOUNT" in label: 
            amount += char
        elif "ACTION" in label: 
            action += char
            
    # 金额清洗（只保留数字和小数点）
    amount_clean = re.sub(r'[^\d.]', '', amount)
    
    # 商业后处理兜底（规则映射）
    category = "其他"
    if any(k in action for k in ["影", "游", "玩", "戏", "KTV", "旅"]): category = "娱乐"
    elif any(k in action for k in ["病", "药", "医", "检"]): category = "医疗"
    elif any(k in action for k in ["房", "水", "物", "家", "空", "租"]): category = "居住"
    elif any(k in action for k in ["吃", "喝", "餐", "菜", "堡", "茶", "啡", "麦当劳", "星巴克"]): category = "餐饮"
    elif any(k in action for k in ["衣", "鞋", "买", "键", "耳", "妆", "淘宝", "礼物"]): category = "购物"
    elif any(k in action for k in ["车", "铁", "交", "油", "地铁"]): category = "交通"
    
    # 如果模型没抽出有效词，强行在原文里捞
    if category == "其他":
        if "衣服" in text or "鞋" in text or "礼物" in text: category = "购物"
        elif "打车" in text or "地铁" in text: category = "交通"
        elif "奶茶" in text or "星巴克" in text: category = "餐饮"
        elif "房租" in text: category = "居住"
        
    return {
        "amount": amount_clean, 
        "action": action, 
        "category": category
    }

# ==========================================
# 5. API 路由定义
# ==========================================
class UserInput(BaseModel):
    text: str

@app.get("/health")
def health_check():
    """用于 Render 等平台的健康检查和保活"""
    return {"status": "ok"}

@app.post("/api/extract")
def api_extract(data: UserInput):
    """核心记账识别接口"""
    try:
        if not data.text.strip():
            return {"code": 400, "msg": "输入文本不能为空", "data": None}
            
        result = extract_bill_logic(data.text)
        return {"code": 200, "msg": "success", "data": result}
    except Exception as e:
        return {"code": 500, "msg": str(e), "data": None}

# 这一行仅用于本地直接 python app.py 测试，Docker 中会通过 CMD uvicorn 启动
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)