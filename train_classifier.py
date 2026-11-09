# -*- coding: utf-8 -*-
"""
句级分类头训练：在自研 MiniEncoder 骨干上加 mean-pooling + Linear 分类头
- 骨干权重来自 my_ner_model_final.pt（NER 阶段产出，含完整预训练+微调权重）
- 数据：cls_dataset.json（199 条标注 = 49 调参集 + 150 留出集，分层切分）
用法: python3 train_classifier.py
"""
import json, math, random, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

torch.set_num_threads(4)
random.seed(42)
torch.manual_seed(42)

# ---------- 模型结构（与 app.py 完全一致） ----------
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
            nn.Linear(n_embd, 4 * n_embd), nn.GELU(), nn.Linear(4 * n_embd, n_embd)
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

class MiniEncoderForCLS(nn.Module):
    """句级分类：masked mean-pooling + Linear"""
    def __init__(self, base_encoder, num_classes):
        super().__init__()
        self.encoder = base_encoder
        self.cls_head = nn.Linear(384, num_classes)

    def forward(self, idx, mask):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        h = self.encoder.wte(idx) + self.encoder.wpe(pos)
        for block in self.encoder.blocks:
            h = block(h, mask)
        h = self.encoder.ln_f(h)
        w = mask.unsqueeze(-1).float()
        pooled = (h * w).sum(1) / w.sum(1).clamp(min=1e-6)
        return self.cls_head(pooled)

# ---------- 数据 ----------
import sys
DATA_FILE = "cls_dataset.json"
for i, a in enumerate(sys.argv):
    if a == "--data":
        DATA_FILE = sys.argv[i + 1]
VOCAB = 20000
MAXLEN = 32
data = json.load(open(DATA_FILE, encoding="utf-8"))
labels = data["labels"]
label2id = {l: i for i, l in enumerate(labels)}
train_s, val_s = data["train"], data["val"]
print(f"标签: {labels}\n训练 {len(train_s)} / 验证 {len(val_s)}")

tokenizer = Tokenizer.from_file("chinese_bpe_tokenizer.json")

def encode(samples):
    ids_all, y_all = [], []
    for s in samples:
        enc = tokenizer.encode(s["text"]).ids[:MAXLEN]
        ids_all.append(enc)
        y_all.append(label2id[s["label"]])
    L = max(len(x) for x in ids_all)
    ids = torch.zeros(len(ids_all), L, dtype=torch.long)
    mask = torch.zeros(len(ids_all), L, dtype=torch.bool)
    for i, x in enumerate(ids_all):
        ids[i, : len(x)] = torch.tensor(x)
        mask[i, : len(x)] = True
    return ids, mask, torch.tensor(y_all)

tr_ids, tr_mask, tr_y = encode(train_s)
va_ids, va_mask, va_y = encode(val_s)

# ---------- 骨干加载（来自 NER checkpoint，剥离前缀） ----------
backbone = MiniEncoderForMLM(vocab_size=VOCAB, block_size=256)
ner_sd = torch.load("my_ner_model_final.pt", map_location="cpu")
backbone_sd = {k[len("encoder."):]: v for k, v in ner_sd.items() if k.startswith("encoder.")}
missing, unexpected = backbone.load_state_dict(backbone_sd, strict=False)
print(f"骨干加载: 缺失 {missing or '无'} | 多余 {unexpected or '无'}")

model = MiniEncoderForCLS(backbone, num_classes=len(labels))

# ---------- 训练 ----------
import sys
FREEZE = "--freeze" in sys.argv
if FREEZE:
    for p in model.encoder.parameters():
        p.requires_grad = False
    print("模式: 冻结骨干，仅训练分类头")
EPOCHS, BS = (30, 8) if FREEZE else (12, 8)
opt = torch.optim.AdamW([
    {"params": model.encoder.parameters(), "lr": 2e-5},
    {"params": model.cls_head.parameters(), "lr": 1e-3},
], weight_decay=0.01)
steps = math.ceil(len(tr_ids) / BS)

def eval_acc():
    model.eval()
    with torch.no_grad():
        logits = model(va_ids, va_mask)
        pred = logits.argmax(-1)
    model.train()
    return (pred == va_y).float().mean().item()

best = 0.0
for ep in range(1, EPOCHS + 1):
    perm = torch.randperm(len(tr_ids))
    tot, hit, loss_sum = 0, 0, 0.0
    for i in range(steps):
        idx = perm[i * BS: (i + 1) * BS]
        logits = model(tr_ids[idx], tr_mask[idx])
        loss = F.cross_entropy(logits, tr_y[idx], label_smoothing=0.1)
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_sum += loss.item() * len(idx)
        hit += (logits.argmax(-1) == tr_y[idx]).sum().item()
        tot += len(idx)
    va = eval_acc()
    print(f"epoch {ep:>2} | train_loss {loss_sum/tot:.4f} | train_acc {hit/tot:.3f} | val_acc {va:.3f}")
    if va >= best:
        best = va
        torch.save({"model": model.state_dict(), "labels": labels}, "cls_classifier.pt")

print(f"\n✅ 训练完成，最佳验证集准确率 {best:.1%}，已保存 cls_classifier.pt")

# 验证集错误明细
model.load_state_dict(torch.load("cls_classifier.pt", map_location="cpu")["model"])
model.eval()
with torch.no_grad():
    pred = model(va_ids, va_mask).argmax(-1)
print("--- 验证集错误 ---")
for s, p in zip(val_s, pred.tolist()):
    if p != label2id[s["label"]]:
        print(f"{s['text']} | 期望 {s['label']} -> 预测 {labels[p]}")
