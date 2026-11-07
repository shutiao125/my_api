import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import re
import datetime

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
# 日期短语正则：10月25号 / 2026-10-25 / 10月25日 / 25号
DATE_RE = re.compile(
    r'\d{4}\s*[-/年]\s*\d{1,2}\s*[-/月]\s*\d{1,2}\s*[号日]?'
    r'|\d{1,4}\s*月\s*\d{1,2}\s*[号日]?'
    r'|\d{1,2}\s*[号日]'
)

def extract_date(text: str):
    """从原文解析记账日期，返回 YYYY-MM-DD；无法识别时返回 None（前端兜底今天）"""
    today = datetime.date.today()
    if "昨天" in text:
        return str(today - datetime.timedelta(days=1))
    if "明天" in text:
        return str(today + datetime.timedelta(days=1))
    m = re.search(r'(\d{4})\s*[-/年]\s*(\d{1,2})\s*[-/月]\s*(\d{1,2})', text)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    else:
        m = re.search(r'(\d{1,2})\s*月\s*(\d{1,2})', text)
        if m:
            y, mo, d = today.year, int(m.group(1)), int(m.group(2))
        else:
            m = re.search(r'(\d{1,2})\s*[号日]', text)
            if m:
                y, mo, d = today.year, today.month, int(m.group(1))
            else:
                return None
    try:
        return str(datetime.date(y, mo, d))
    except ValueError:
        return None

def extract_bill_logic(text: str):
    # 先剥离日期短语再推理，避免 NER 把日期里的数字误标为金额
    date_str = extract_date(text)
    text_for_ner = DATE_RE.sub('', text)

    input_ids = [tokenizer.token_to_id(t) or tokenizer.token_to_id("[UNK]") for t in text_for_ner]
    x = torch.tensor([input_ids]).to(device)
    mask = torch.ones_like(x, dtype=torch.bool).to(device)

    with torch.no_grad():
        logits = ner_model(x, mask)
        preds = torch.argmax(logits, dim=-1)[0].cpu().numpy()

    amount, action = "", ""
    for char, pred_id in zip(text_for_ner, preds):
        label = id2label.get(pred_id, "O")
        if "AMOUNT" in label:
            amount += char
        elif "ACTION" in label:
            action += char

    # 金额清洗（只保留数字和小数点）
    amount_clean = re.sub(r'[^\d.]', '', amount)

    # 兜底：模型没标出金额时，用正则找 "50块" / "35元" / "¥35"
    if not amount_clean:
        m = re.search(r'(\d+(?:\.\d+)?)\s*[块元]', text_for_ner) \
            or re.search(r'[¥￥](\d+(?:\.\d+)?)', text_for_ner)
        if m:
            amount_clean = m.group(1)
    
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
        "category": category,
        "date": date_str
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


# ==========================================
# 6. 邮箱验证码服务
# 优先使用 Brevo HTTP API（Render 免费实例封禁 SMTP 出站端口，
# HTTP API 走 443 端口不受限）；未配置 BREVO_API_KEY 时回退 SMTP
# ==========================================
import json
import os
import random
import smtplib
import time
import urllib.request
from email.header import Header
from email.mime.text import MIMEText
from email.utils import formataddr

SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.qq.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "465"))
SMTP_USER = os.environ.get("SMTP_USER", "2498383161@qq.com")  # 发件邮箱
SMTP_PASS = os.environ.get("SMTP_PASS", "")  # QQ邮箱SMTP授权码（仅 SMTP 回退方式使用）
BREVO_API_KEY = os.environ.get("BREVO_API_KEY", "")  # Brevo API Key（不支持免费邮箱发件人，已弃用）
SENDER_NAME = os.environ.get("SENDER_NAME", "记账本")

# EmailJS（推荐）：在 emailjs.com 绑定 QQ 邮箱 SMTP，通过其 443 端口 HTTP 接口发信
EMAILJS_SERVICE_ID = os.environ.get("EMAILJS_SERVICE_ID", "")
EMAILJS_TEMPLATE_ID = os.environ.get("EMAILJS_TEMPLATE_ID", "")
EMAILJS_PUBLIC_KEY = os.environ.get("EMAILJS_PUBLIC_KEY", "")
EMAILJS_PRIVATE_KEY = os.environ.get("EMAILJS_PRIVATE_KEY", "")  # REST 发信必需的私钥

CODE_TTL = 5 * 60           # 验证码有效期 5 分钟
CODE_RESEND_INTERVAL = 60   # 同一邮箱 60 秒内只能发一次
CODE_MAX_ATTEMPTS = 5       # 最多可错 5 次

code_store = {}  # 内存存储 {email: {code, expiry, last_send, attempts}}


def send_code_email(to_email: str, code: str):
    html = f"""
    <div style="max-width:480px;margin:0 auto;font-family:Arial,'Microsoft YaHei',sans-serif;
                padding:28px;border:1px solid #e5e7eb;border-radius:12px;background:#ffffff;">
      <h2 style="color:#1f2937;margin:0 0 8px;">记账本 · 邮箱验证码</h2>
      <p style="color:#6b7280;font-size:14px;">你正在注册账号，本次验证码 <b>5 分钟内有效</b>：</p>
      <p style="font-size:34px;font-weight:bold;letter-spacing:10px;color:#2b6bf3;margin:18px 0;">{code}</p>
      <p style="color:#9ca3af;font-size:12px;">如果这不是你的操作，请忽略本邮件。</p>
    </div>
    """

    # 方式一：EmailJS HTTP API（443 端口，云平台不受限，推荐；发件人 = 绑定的 QQ 邮箱）
    if EMAILJS_SERVICE_ID and EMAILJS_TEMPLATE_ID and EMAILJS_PUBLIC_KEY:
        payload = {
            "service_id": EMAILJS_SERVICE_ID,
            "template_id": EMAILJS_TEMPLATE_ID,
            "user_id": EMAILJS_PUBLIC_KEY,
            "accessToken": EMAILJS_PRIVATE_KEY,  # REST 调用需要私钥
            "template_params": {
                "to_email": to_email,
                "code": code,
                "subject": "记账本 · 邮箱验证码",
            },
        }
        req = urllib.request.Request(
            "https://api.emailjs.com/api/v1.0/email/send",
            data=json.dumps(payload).encode("utf-8"),
            headers={"accept": "application/json", "content-type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8", "ignore")
            if resp.status != 200:
                raise RuntimeError(f"EmailJS 返回 {resp.status}: {body}")
        return

    # 方式二：Brevo HTTP API（要求发件人域名通过验证，免费邮箱不可用）
    if BREVO_API_KEY:
        payload = {
            "sender": {"name": SENDER_NAME, "email": SMTP_USER},
            "to": [{"email": to_email}],
            "subject": "记账本 · 邮箱验证码",
            "htmlContent": html,
        }
        req = urllib.request.Request(
            "https://api.brevo.com/v3/smtp/email",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "accept": "application/json",
                "content-type": "application/json",
                "api-key": BREVO_API_KEY,
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            if resp.status not in (200, 201):
                raise RuntimeError(f"Brevo API 返回 {resp.status}")
        return

    # 方式二：SMTP 直发（需要所在平台放行 587/465 出站端口）
    msg = MIMEText(html, "html", "utf-8")
    msg["Subject"] = Header("记账本 · 邮箱验证码", "utf-8")
    msg["From"] = formataddr((SENDER_NAME, SMTP_USER))
    msg["To"] = to_email

    attempts = [(587, "starttls"), (SMTP_PORT, "ssl")]
    errors = []
    for port, mode in attempts:
        try:
            if mode == "ssl":
                with smtplib.SMTP_SSL(SMTP_HOST, port, timeout=20) as server:
                    server.login(SMTP_USER, SMTP_PASS)
                    server.sendmail(SMTP_USER, [to_email], msg.as_string())
            else:
                with smtplib.SMTP(SMTP_HOST, port, timeout=20) as server:
                    server.ehlo()
                    server.starttls()
                    server.ehlo()
                    server.login(SMTP_USER, SMTP_PASS)
                    server.sendmail(SMTP_USER, [to_email], msg.as_string())
            return  # 发送成功
        except smtplib.SMTPAuthenticationError:
            raise  # 授权码错误换端口也没用，直接抛出让上层返回明确提示
        except Exception as e:
            errors.append(f"{port}端口({mode}): {e}")
    raise RuntimeError("; ".join(errors))


class EmailCodeInput(BaseModel):
    email: str
    code: str = ""


@app.post("/api/send-code")
def api_send_code(data: EmailCodeInput):
    """向指定邮箱发送 6 位验证码"""
    email = data.email.strip().lower()
    if not re.match(r"^[^\s@]+@[^\s@]+\.[^\s@]+$", email):
        return {"code": 400, "msg": "邮箱格式不正确"}
    if not (BREVO_API_KEY or SMTP_PASS):
        return {"code": 500, "msg": "邮件服务未配置（缺少 BREVO_API_KEY 环境变量）"}

    now = time.time()
    rec = code_store.get(email)
    if rec and now - rec["last_send"] < CODE_RESEND_INTERVAL:
        return {"code": 429, "msg": "发送太频繁，请稍后再试"}

    code = f"{random.randint(0, 999999):06d}"

    # 先写入限流占位，防止发送耗时期间并发重复发信（发送失败再回滚允许重试）
    code_store[email] = {
        "code": code,
        "expiry": now + CODE_TTL,
        "last_send": now,
        "attempts": 0,
    }
    try:
        send_code_email(email, code)
    except smtplib.SMTPAuthenticationError:
        code_store.pop(email, None)
        return {"code": 500, "msg": "发件邮箱认证失败，请检查 SMTP_PASS 是否为QQ邮箱SMTP授权码（非QQ登录密码）"}
    except Exception as e:
        code_store.pop(email, None)
        return {"code": 500, "msg": f"邮件发送失败：{e}"}

    return {"code": 200, "msg": "验证码已发送"}


@app.post("/api/verify-code")
def api_verify_code(data: EmailCodeInput):
    """校验邮箱验证码，验证通过后该码立即作废"""
    email = data.email.strip().lower()
    code = data.code.strip()
    rec = code_store.get(email)
    if not rec:
        return {"code": 400, "msg": "请先获取验证码"}
    if time.time() > rec["expiry"]:
        code_store.pop(email, None)
        return {"code": 400, "msg": "验证码已过期，请重新获取"}

    rec["attempts"] += 1
    if rec["attempts"] > CODE_MAX_ATTEMPTS:
        code_store.pop(email, None)
        return {"code": 429, "msg": "错误次数过多，请重新获取验证码"}
    if code != rec["code"]:
        return {"code": 400, "msg": "验证码错误"}

    code_store.pop(email, None)
    return {"code": 200, "msg": "验证通过"}