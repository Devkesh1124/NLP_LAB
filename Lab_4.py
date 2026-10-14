"""
GPT-2 (124M style) from-scratch pipeline
Data -> Tokenize -> Dataset/DataLoader -> Model -> Train -> Evaluate -> Generate -> Save

Usage:
    python gpt_pipeline.py --data my_text.txt --epochs 5
"""
import argparse
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer

# ----------------------------------------------------------------------
# 1. Config
# ----------------------------------------------------------------------
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,   # 1024 for full GPT-2; 256 is lighter for training
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}


# ----------------------------------------------------------------------
# 2. Data: tokenizer + sliding-window dataset
# ----------------------------------------------------------------------
class GPTDataset(Dataset):
    """Chops a long token stream into (input, target) windows; target = input shifted by 1."""

    def __init__(self, text, tokenizer, max_length, stride):
        ids = tokenizer.encode(text)
        assert len(ids) > max_length, "Text too short for the chosen context_length"
        self.inputs, self.targets = [], []
        for i in range(0, len(ids) - max_length, stride):
            self.inputs.append(torch.tensor(ids[i : i + max_length]))
            self.targets.append(torch.tensor(ids[i + 1 : i + max_length + 1]))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def make_loaders(text, tokenizer, cfg, batch_size=2, train_ratio=0.9):
    split = int(train_ratio * len(text))
    ctx = cfg["context_length"]
    train_ds = GPTDataset(text[:split], tokenizer, ctx, stride=ctx)
    val_ds = GPTDataset(text[split:], tokenizer, ctx, stride=ctx)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_dl, val_dl


# ----------------------------------------------------------------------
# 3. Model
# ----------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, n_heads, context_length, drop_rate, qkv_bias=False):
        super().__init__()
        assert emb_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = emb_dim // n_heads
        self.q = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)
        self.k = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)
        self.v = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)
        self.out = nn.Linear(emb_dim, emb_dim)
        self.drop = nn.Dropout(drop_rate)
        # Causal mask: token i can only look at tokens <= i
        mask = torch.tril(torch.ones(context_length, context_length)).bool()
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        q = self.q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        att = att.masked_fill(~self.mask[:T, :T], float("-inf"))
        att = self.drop(torch.softmax(att, dim=-1))

        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)


class FeedForward(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.GELU(),
            nn.Linear(4 * emb_dim, emb_dim),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-LayerNorm block (what GPT-2 actually uses)."""

    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg["emb_dim"])
        self.att = MultiHeadAttention(
            cfg["emb_dim"], cfg["n_heads"], cfg["context_length"],
            cfg["drop_rate"], cfg["qkv_bias"],
        )
        self.ln2 = nn.LayerNorm(cfg["emb_dim"])
        self.ff = FeedForward(cfg["emb_dim"])
        self.drop = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        x = x + self.drop(self.att(self.ln1(x)))
        x = x + self.drop(self.ff(self.ln2(x)))
        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])  # learned, like GPT-2
        self.drop = nn.Dropout(cfg["drop_rate"])
        self.blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_ln = nn.LayerNorm(cfg["emb_dim"])
        self.head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying (saves ~38M params)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        x = self.blocks(x)
        return self.head(self.final_ln(x))  # (B, T, vocab_size)


# ----------------------------------------------------------------------
# 4. Training + evaluation
# ----------------------------------------------------------------------
def batch_loss(model, x, y, device):
    x, y = x.to(device), y.to(device)
    logits = model(x)
    return nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten())


@torch.no_grad()
def evaluate(model, loader, device, max_batches=None):
    model.eval()
    total, n = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        total += batch_loss(model, x, y, device).item()
        n += 1
    model.train()
    return total / max(n, 1)


def train(model, train_dl, val_dl, tokenizer, cfg, device,
          epochs=5, lr=4e-4, eval_every=50, start_text="Every day is your"):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    step = 0
    model.train()
    for epoch in range(epochs):
        for x, y in train_dl:
            opt.zero_grad()
            loss = batch_loss(model, x, y, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1

            if step % eval_every == 0:
                val = evaluate(model, val_dl, device, max_batches=5)
                print(f"epoch {epoch+1} | step {step} | train {loss.item():.3f} | val {val:.3f}")

        # sample after each epoch to see progress
        print(f"--- sample after epoch {epoch+1}: "
              f"{generate(model, tokenizer, start_text, 30, cfg, device, temperature=0.8, top_k=40)}")


# ----------------------------------------------------------------------
# 5. Generation (greedy, or temperature + top-k sampling)
# ----------------------------------------------------------------------
@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens, cfg, device,
             temperature=0.0, top_k=None):
    model.eval()
    idx = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -cfg["context_length"]:]
        logits = model(idx_cond)[:, -1, :]  # (1, vocab)

        if top_k is not None:
            kth = torch.topk(logits, top_k).values[:, -1, None]
            logits = logits.masked_fill(logits < kth, float("-inf"))

        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
        else:
            nxt = torch.argmax(logits, dim=-1, keepdim=True)

        idx = torch.cat((idx, nxt), dim=1)

    model.train()
    return tokenizer.decode(idx.squeeze(0).tolist())


# ----------------------------------------------------------------------
# 6. Main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="path to a .txt training file")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--prompt", type=str, default="Every day is your")
    ap.add_argument("--save", type=str, default="gpt_checkpoint.pt")
    args = ap.parse_args()

    torch.manual_seed(123)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    with open(args.data, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"Characters: {len(text):,} | Tokens: {len(tokenizer.encode(text)):,}")

    train_dl, val_dl = make_loaders(text, tokenizer, GPT_CONFIG_124M, args.batch_size)
    print(f"Train batches: {len(train_dl)} | Val batches: {len(val_dl)}")

    model = GPTModel(GPT_CONFIG_124M).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("Before training:", generate(model, tokenizer, args.prompt, 15, GPT_CONFIG_124M, device))

    train(model, train_dl, val_dl, tokenizer, GPT_CONFIG_124M, device,
          epochs=args.epochs, lr=args.lr, start_text=args.prompt)

    torch.save({"model": model.state_dict(), "config": GPT_CONFIG_124M}, args.save)
    print(f"Saved checkpoint -> {args.save}")

    print("Final:", generate(model, tokenizer, args.prompt, 50, GPT_CONFIG_124M, device,
                             temperature=0.8, top_k=40))


if __name__ == "__main__":
    main()
