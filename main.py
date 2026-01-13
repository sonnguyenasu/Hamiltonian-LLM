# ==========================================
# HAMILTONIAN NANO v8.1 - GOLDEN CONFIG
# 124M Params | Effective Batch 128 | 1.3B Tokens
# ==========================================
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, IterableDataset
from dataclasses import dataclass
from transformers import AutoTokenizer
from tqdm.auto import tqdm
from typing import List

# Hardware Optimization
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

try:
    from datasets import load_dataset
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "-q", "datasets"])
    from datasets import load_dataset

@dataclass
class NanoConfig:
    vocab_size: int = 50304  # Safety Pad
    
    # --- 124M ARCHITECTURE ---
    d_model: int = 640       # Adjusted for Momentum overhead
    n_layers: int = 12
    n_heads: int = 10        # 640/64 = 10
    head_dim: int = 64
    seq_len: int = 1024
    conv_kernel: int = 3
    dropout: float = 0.1
    ema_beta: float = 0.9

    # --- THE GOLDEN CONFIG ---
    batch_size: int = 8          # Physical (Fits in VRAM)
    grad_accum_steps: int = 16   # Logical (8 * 16 = 128 Effective)
    
    # Learning Rate: 6e-4 is standard for Batch 128
    learning_rate: float = 6e-4  
    
    # 10,000 Steps * 128 Batch * 1024 Seq = ~1.3 Billion Tokens
    max_steps: int = 10000       
    warmup_steps: int = 1000     # Increased warmup for stability
    save_dir: str = "./Hamiltonian_v8_124M_Golden"

cfg = NanoConfig()
os.makedirs(cfg.save_dir, exist_ok=True)

# ==========================================
# 1. PHYSICS KERNEL
# ==========================================
@torch.jit.script
def ema_scan_with_reset(velocity: torch.Tensor, beta: float, reset_mask: torch.Tensor):
    v_steps = velocity.unbind(1)
    r_steps = reset_mask.unbind(1)
    m_t = torch.zeros_like(v_steps[0])
    momentum_list: List[torch.Tensor] = []

    for i in range(len(v_steps)):
        v_t = v_steps[i]
        r_t = r_steps[i]
        m_next = beta * m_t + (1.0 - beta) * v_t
        # Viscous Damping
        m_t = m_next * (1.0 - r_t)
        momentum_list.append(m_t)

    return torch.stack(momentum_list, dim=1)

class EMAMomentumEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.beta = config.ema_beta
        self.q_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.p_emb = nn.Embedding(config.vocab_size, config.d_model)

    def forward(self, idx, reset_mask=None):
        idx = idx.clamp(0, 50256)
        q = self.q_emb(idx)
        p_raw = self.p_emb(idx)

        p_prev = torch.roll(p_raw, shifts=1, dims=1)
        mask = torch.ones_like(p_raw); mask[:, 0, :] = 0
        velocity = (p_raw - p_prev) * mask

        if reset_mask is None:
            reset_mask = torch.zeros((idx.size(0), idx.size(1), 1), device=idx.device)
        else:
            if reset_mask.dim() == 2: reset_mask = reset_mask.unsqueeze(-1)

        momentum = ema_scan_with_reset(velocity, self.beta, reset_mask.float())
        return q + momentum

# ==========================================
# 2. MODEL ARCHITECTURE
# ==========================================
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
    def forward(self, x, seq_len=None):
        if seq_len is None: seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb[None, None, :, :]

def apply_rotary_pos_emb(x, pos):
    x1, x2 = x.chunk(2, dim=-1)
    return (x * pos.cos()) + (torch.cat((-x2, x1), dim=-1) * pos.sin())

class AdaptiveGravityAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.scale = config.head_dim ** -0.5
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.gravity_net = nn.Linear(config.d_model, config.n_heads)
        nn.init.normal_(self.gravity_net.weight, std=0.01)
        nn.init.constant_(self.gravity_net.bias, 0.5)

    def forward(self, x, rope):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q = apply_rotary_pos_emb(q, rope)
        k = apply_rotary_pos_emb(k, rope)
        
        att = (q @ k.transpose(-2, -1)) * self.scale
        g_val = F.softplus(self.gravity_net(x)).transpose(1, 2).unsqueeze(-1)
        indices = torch.arange(T, device=x.device)
        dist = (indices.view(1, 1, T, 1) - indices.view(1, 1, 1, T)).abs()
        dist = torch.clamp(dist, max=64) 
        att = att - (g_val * dist * 0.1)
        mask = torch.tril(torch.ones(T, T, device=x.device)).bool()
        att = att.masked_fill(~mask, float('-inf'))
        y = F.softmax(att, dim=-1) @ v
        return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))

class HamiltonianBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = AdaptiveGravityAttention(config)
        self.reflex = nn.Sequential(
            nn.Conv1d(config.d_model, config.d_model, config.conv_kernel, padding=config.conv_kernel-1, groups=config.d_model),
            nn.SiLU()
        )
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model, bias=False),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model, bias=False)
        )
    def forward(self, x, rope):
        x_reflex = self.reflex(x.transpose(1, 2))[:, :, :x.size(1)].transpose(1, 2)
        x = x + self.attn(self.ln1(x), rope) + x_reflex
        x = x + self.mlp(self.ln2(x))
        return x

class HamiltonianNano_v8(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = EMAMomentumEmbedding(config),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([HamiltonianBlock(config) for _ in range(config.n_layers)]),
            ln_f = nn.LayerNorm(config.d_model),
        ))
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.rope = RotaryEmbedding(config.head_dim, config.seq_len)
        self.transformer.wte.q_emb.weight = self.lm_head.weight

    def forward(self, idx, targets=None, reset_mask=None):
        B, T = idx.size()
        x = self.transformer.wte(idx, reset_mask=reset_mask)
        x = self.transformer.drop(x)
        rope = self.rope(x, seq_len=T)
        for block in self.transformer.h:
            x = block(x, rope)
        x = self.transformer.ln_f(x)
        
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
            return logits, loss
        return self.lm_head(x[:, [-1], :]), None

# ==========================================
# 3. TRAINING LOOP
# ==========================================
class FineWebStream(IterableDataset):
    def __init__(self, tokenizer, seq_len=1024):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        print("🌊 Connecting to FineWeb-Edu Stream...")
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
        self.dataset = ds.shuffle(seed=42, buffer_size=10000)
    def __iter__(self):
        buffer = []
        for sample in self.dataset:
            tokens = self.tokenizer.encode(sample['text']) + [self.tokenizer.eos_token_id]
            buffer.extend(tokens)
            while len(buffer) >= self.seq_len + 1:
                yield torch.tensor(buffer[:self.seq_len + 1])
                buffer = buffer[self.seq_len + 1:]

def main():
    device = "cuda"
    print(f"🚀 Initializing v8.1 (124M Golden) on {device}...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    model = HamiltonianNano_v8(cfg).to(device)
    #model = torch.compile(model)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Total Params: {params/1e6:.2f}M")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    ds = FineWebStream(tokenizer, cfg.seq_len)
    dl = DataLoader(ds, batch_size=cfg.batch_size, num_workers=4, pin_memory=True)
    scaler = torch.amp.GradScaler('cuda')
    
    model.train()
    micro_step = 0
    global_step = 0
    pbar = tqdm(total=cfg.max_steps, desc="Training", unit="step")
    DATA_CLAMP = 50256
    EOS_ID = tokenizer.eos_token_id

    for batch in dl:
        if global_step >= cfg.max_steps: break
        
        batch = batch.clamp(0, DATA_CLAMP).to(device, non_blocking=True)
        x, y = batch[:, :-1], batch[:, 1:]
        reset_mask = (x == EOS_ID).float() * 0.5
        
        with torch.amp.autocast('cuda', dtype=dtype):
            logits, loss = model(x, targets=y, reset_mask=reset_mask)
            loss = loss / cfg.grad_accum_steps
            
        scaler.scale(loss).backward()
        
        if (micro_step + 1) % cfg.grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item()*cfg.grad_accum_steps:.4f}")
            
            if global_step % 500 == 0:
                torch.save(model.state_dict(), f"{cfg.save_dir}/v8_golden_step{global_step}.pt")
                # Sanity Generation
                with torch.no_grad():
                     idx = torch.tensor([[tokenizer.bos_token_id or 50256]], device=device)
                     for _ in range(15):
                         logits, _ = model(idx)
                         idx = torch.cat((idx, torch.argmax(logits[:,-1], dim=-1, keepdim=True)), 1)
                     pbar.write(f"Sample: {tokenizer.decode(idx[0])}")

        micro_step += 1

    print("✅ Training Complete.")

if __name__ == "__main__":
    main()
