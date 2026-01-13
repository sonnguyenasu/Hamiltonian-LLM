import os
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoTokenizer
from dataclasses import dataclass
from typing import List, Optional

# ==========================================
# 1. CONFIGURATION (Must match Training!)
# ==========================================
@dataclass
class NanoConfig:
    vocab_size: int = 50304   # Safety Pad
    d_model: int = 640        # 124M Strict Scale
    n_layers: int = 12
    n_heads: int = 10         # 640/64 = 10
    head_dim: int = 64
    seq_len: int = 1024
    conv_kernel: int = 3
    dropout: float = 0.0      # No dropout during inference
    ema_beta: float = 0.9

# ==========================================
# 2. PHYSICS KERNEL & EMBEDDINGS
# ==========================================
@torch.jit.script
def ema_scan_with_reset(velocity: torch.Tensor, beta: float, reset_mask: torch.Tensor):
    """
    Scans through the sequence applying momentum conservation with viscous damping.
    reset_mask: 1.0 = Full Stop (Friction), 0.0 = Free Flow.
    """
    v_steps = velocity.unbind(1)
    r_steps = reset_mask.unbind(1)
    m_t = torch.zeros_like(v_steps[0])
    momentum_list: List[torch.Tensor] = []

    for i in range(len(v_steps)):
        v_t = v_steps[i]
        r_t = r_steps[i]
        # Update Momentum: Beta * Old + (1-Beta) * New
        m_next = beta * m_t + (1.0 - beta) * v_t
        # Apply Viscous Damping (Reset)
        m_t = m_next * (1.0 - r_t)
        momentum_list.append(m_t)

    return torch.stack(momentum_list, dim=1)

class EMAMomentumEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.beta = config.ema_beta
        self.q_emb = nn.Embedding(config.vocab_size, config.d_model) # Position (Potential)
        self.p_emb = nn.Embedding(config.vocab_size, config.d_model) # Velocity (Kinetic)

    def forward(self, idx, reset_mask=None):
        idx = idx.clamp(0, 50256)
        q = self.q_emb(idx)
        p_raw = self.p_emb(idx)

        # Calculate Velocity (Change in p)
        p_prev = torch.roll(p_raw, shifts=1, dims=1)
        # Zero out the very first step's velocity to prevent wrap-around artifacts
        mask = torch.ones_like(p_raw); mask[:, 0, :] = 0
        velocity = (p_raw - p_prev) * mask

        if reset_mask is None:
            reset_mask = torch.zeros((idx.size(0), idx.size(1), 1), device=idx.device)
        else:
            if reset_mask.dim() == 2: reset_mask = reset_mask.unsqueeze(-1)

        # Apply Hamiltonian Integration
        momentum = ema_scan_with_reset(velocity, self.beta, reset_mask.float())
        
        # H = T + V -> Return Combined State
        return q + momentum

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

# ==========================================
# 3. ATTENTION & BLOCKS
# ==========================================
class AdaptiveGravityAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.scale = config.head_dim ** -0.5
        
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.gravity_net = nn.Linear(config.d_model, config.n_heads)

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
        
        # Gravity Penalty
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

    def forward(self, idx, reset_mask=None, steer_vec=None):
        B, T = idx.size()
        x = self.transformer.wte(idx, reset_mask=reset_mask)
        if steer_vec is not None: x = x + steer_vec
        x = self.transformer.drop(x)
        rope = self.rope(x, seq_len=T)
        for block in self.transformer.h:
            x = block(x, rope)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        
        # --- RETURN LOGITS AND HIDDEN STATE ---
        return logits, x

# ==========================================
# 4. GENERATION ENGINES
# ==========================================
def get_concept_vector(model, tokenizer, concept_word, device):
    """Extracts the semantic vector (q) for a specific word."""
    try:
        token_id = tokenizer.encode(concept_word)[0]
    except IndexError:
        print(f"⚠️ Warning: Word '{concept_word}' not found.")
        return None
    with torch.no_grad():
        vec = model.transformer.wte.q_emb.weight[token_id]
    return vec.view(1, 1, -1)

def generate_standard(model, tokenizer, prompt, steer_word=None, steer_strength=0.0, max_new_tokens=100, temperature=0.7, top_k=50, repetition_penalty=1.2, device='cuda'):
    """Standard Greedy/Sampling Generation with optional Steering."""
    model.eval()
    idx = tokenizer.encode(prompt, return_tensors="pt").to(device)
    steer_vec = None
    if steer_word and steer_strength != 0:
        raw_vec = get_concept_vector(model, tokenizer, steer_word, device)
        if raw_vec is not None:
            steer_vec = raw_vec * steer_strength
            print(f"🧲 Steering towards '{steer_word}' with force {steer_strength}...")

    print(f"\nPrompt: {prompt}")
    print("-" * 40)
    print(prompt, end="", flush=True)

    generated_tokens = []
    for _ in range(max_new_tokens):
        reset_mask = (idx == tokenizer.eos_token_id).float() * 0.5
        with torch.no_grad():
            logits, _ = model(idx, reset_mask=reset_mask, steer_vec=steer_vec) # Ignore hidden state
            
        logits = logits[:, -1, :] / temperature
        for t in set(generated_tokens):
            if logits[0, t] < 0: logits[0, t] *= repetition_penalty
            else: logits[0, t] /= repetition_penalty

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        token = tokenizer.decode(idx_next[0])
        print(token, end="", flush=True)
        generated_tokens.append(idx_next.item())
        idx = torch.cat((idx, idx_next), dim=1)
        if idx.size(1) > 1024: idx = idx[:, -1024:]
    print("\n" + "-" * 40)

def generate_least_action(model, tokenizer, prompt, lookahead_steps=3, candidates=5, max_new_tokens=50, repetition_window=15, device='cuda'):
    """Mode 3: Minimizes Probability Action + Pauli Exclusion."""
    model.eval()
    idx = tokenizer.encode(prompt, return_tensors="pt").to(device)
    print(f"\nPrompt: {prompt}")
    print(f"🔮 Planning {lookahead_steps} steps ahead with {candidates} branches (Pauli)...")
    print("-" * 40)
    print(prompt, end="", flush=True)

    for _ in range(max_new_tokens):
        reset_mask = (idx == tokenizer.eos_token_id).float() * 0.5
        with torch.no_grad():
            logits, _ = model(idx, reset_mask=reset_mask)
        
        next_logits = logits[:, -1, :]
        probs = F.softmax(next_logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, candidates)
        
        best_token = None
        lowest_action = float('inf')
        history = idx[0, -repetition_window:].tolist()

        for i in range(candidates):
            candidate_token = top_indices[0, i].item()
            prob = max(top_probs[0, i].item(), 1e-9)
            current_action = -math.log(prob)
            if candidate_token in history: current_action += 10.0
            
            sim_idx = torch.cat((idx, torch.tensor([[candidate_token]], device=device)), dim=1)
            sim_history = history + [candidate_token]
            
            for _ in range(lookahead_steps):
                with torch.no_grad():
                    sim_logits, _ = model(sim_idx)
                step_probs = F.softmax(sim_logits[:, -1, :], dim=-1)
                best_next = torch.argmax(step_probs).item()
                step_energy = -math.log(max(step_probs[0, best_next].item(), 1e-9))
                if best_next in sim_history[-repetition_window:]: step_energy += 10.0
                current_action += step_energy
                sim_idx = torch.cat((sim_idx, torch.tensor([[best_next]], device=device)), dim=1)
                sim_history.append(best_next)
                if sim_idx.size(1) > 1024: break

            avg_action = current_action / (1 + lookahead_steps)
            if avg_action < lowest_action:
                lowest_action = avg_action
                best_token = candidate_token

        if best_token is None: best_token = top_indices[0,0].item()
        token = tokenizer.decode(best_token)
        print(token, end="", flush=True)
        idx = torch.cat((idx, torch.tensor([[best_token]], device=device)), dim=1)
        if idx.size(1) > 1024: idx = idx[:, -1024:]
    print("\n" + "-" * 40)

def generate_hamiltonian_flow(model, tokenizer, prompt, candidates=5, max_new_tokens=50, momentum_weight=2.0, device='cuda'):
    """Mode 4: Minimizes Probability Action + Vector Flow Turbulence."""
    model.eval()
    idx = tokenizer.encode(prompt, return_tensors="pt").to(device)
    print(f"\nPrompt: {prompt}")
    print(f"🌊 Planning with Hamiltonian Flow (Weight: {momentum_weight})...")
    print("-" * 40)
    print(prompt, end="", flush=True)

    for _ in range(max_new_tokens):
        reset_mask = (idx == tokenizer.eos_token_id).float() * 0.5
        with torch.no_grad():
            logits, last_hidden = model(idx, reset_mask=reset_mask)
        
        current_vector = last_hidden[:, -1, :] 
        next_logits = logits[:, -1, :]
        probs = F.softmax(next_logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, candidates)
        
        best_token = None
        lowest_action = float('inf')

        # Lightweight Pauli Check for Flow Mode
        history = idx[0, -32:].tolist()

        for i in range(candidates):
            candidate_token = top_indices[0, i].item()
            prob = max(top_probs[0, i].item(), 1e-9)
            potential_E = -math.log(prob)
            
            # Repetition is High Energy
            if candidate_token in history: potential_E += 5.0

            # Simulate one step to check Vector Alignment
            sim_idx = torch.cat((idx, torch.tensor([[candidate_token]], device=device)), dim=1)
            with torch.no_grad():
                _, sim_hidden = model(sim_idx)
            
            candidate_vector = sim_hidden[:, -1, :]
            
            # Kinetic Cost: How much does the vector turn?
            cosine_dist = 1.0 - F.cosine_similarity(current_vector, candidate_vector, dim=-1).item()
            kinetic_E = cosine_dist * momentum_weight

            total_action = potential_E + kinetic_E
            if total_action < lowest_action:
                lowest_action = total_action
                best_token = candidate_token

        token = tokenizer.decode(best_token)
        print(token, end="", flush=True)
        idx = torch.cat((idx, torch.tensor([[best_token]], device=device)), dim=1)
        if idx.size(1) > 1024: idx = idx[:, -1024:]
    print("\n" + "-" * 40)

# ==========================================
# 5. MAIN INTERFACE
# ==========================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # UPDATE YOUR PATH
    CHECKPOINT_PATH = "./v8_golden_step10000.pt"

    print(f"🚀 Loading Hamiltonian v8.1 Engine on {device}...")
    cfg = NanoConfig()
    model = HamiltonianNano_v8(cfg).to(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    if os.path.exists(CHECKPOINT_PATH):
        try:
            state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module."): k = k[7:]
                new_state_dict[k] = v
            model.load_state_dict(new_state_dict)
            print("✅ Weights Loaded.")
        except Exception as e:
            print(f"❌ Error loading weights: {e}")
    else:
        print(f"⚠️ Checkpoint not found. Using Random Weights.")

    print("\n--- CONTROLS ---")
    print("1. Standard Generate")
    print("2. Steered Generate")
    print("3. Plan Ahead (Probability + Pauli)")
    print("4. Hamiltonian Flow (Probability + Vector Smoothness)")
    print("q. Quit")

    while True:
        mode = input("\nSelect Mode (1-4/q): ").strip().lower()
        if mode == 'q': break
        
        prompt = input("Enter Prompt: ")
        if not prompt: continue

        if mode == '1':
            generate_standard(model, tokenizer, prompt, device=device)
        elif mode == '2':
            target = input("Target Word: ")
            strength = float(input("Strength: "))
            generate_standard(model, tokenizer, prompt, steer_word=target, steer_strength=strength, device=device)
        elif mode == '3':
            lookahead = int(input("Lookahead Steps (e.g. 3): "))
            generate_least_action(model, tokenizer, prompt, lookahead_steps=lookahead, device=device)
        elif mode == '4':
            weight = float(input("Momentum Weight (e.g. 2.0): "))
            generate_hamiltonian_flow(model, tokenizer, prompt, momentum_weight=weight, device=device)
