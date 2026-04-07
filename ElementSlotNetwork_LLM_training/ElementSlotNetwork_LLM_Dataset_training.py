# -*- coding: utf-8 -*-
"""
Element-Slot Network — Training / Fine-tuning Version
Supports full training and fine-tuning from saved weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import os

# =========================
# PATCH EMBEDDING
# =========================
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=28, patch_size=4, in_ch=1, embed_dim=64):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Linear(patch_size * patch_size * in_ch, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.unfold(2, p, p).unfold(3, p, p)
        x = x.contiguous().view(B, -1, p * p * C)
        x = self.proj(x) + self.pos_embed.to(x.device)
        return x

# =========================
# ELEMENT SLOT LAYER
# =========================
class ElementSlotLayer(nn.Module):
    def __init__(self, num_elements, num_slots, dim=64, ticks=6, hidden=128):
        super().__init__()
        self.N = num_elements
        self.K = num_slots
        self.D = dim
        self.T = ticks
        self.tau = 1.0

        self.router = nn.Sequential(
            nn.Linear(dim + 1 + 2 + dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_slots)
        )

        self.memory = nn.Parameter(torch.randn(num_slots, dim) * 0.05)

        self.mem_update = nn.Sequential(
            nn.Linear(dim * 2, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim)
        )
        self.mem_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

        self.G = nn.Sequential(
            nn.Linear(dim * 2 + 1, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim)
        )

        self.stall_net = nn.Sequential(
            nn.Linear(dim, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1)
        )

    def set_tau(self, tau):
        self.tau = max(float(tau), 0.1)

    def forward(self, x):
        B, N, D = x.shape
        mem = self.memory.unsqueeze(0).expand(B, -1, -1).clone()
        routing_hist = []

        for t in range(self.T):
            t_val = t / max(self.T - 1, 1)
            t_emb = torch.full((B, N, 1), t_val, device=x.device, dtype=x.dtype)

            if routing_hist:
                prev_routing = routing_hist[-1]
                expected_slot_load = prev_routing.sum(dim=1)
                load_expected = torch.einsum('bnk,bk->bn', prev_routing, expected_slot_load + 1e-6).unsqueeze(-1)
                load_var = expected_slot_load.var(dim=1, keepdim=True).unsqueeze(1).expand(-1, N, 1)
                load_feat = torch.cat([load_expected, load_var], dim=-1)
            else:
                load_feat = torch.zeros(B, N, 2, device=x.device, dtype=x.dtype)

            mem_summary = mem.mean(dim=1, keepdim=True).expand(-1, N, -1)
            r_in = torch.cat([x, t_emb, load_feat, mem_summary], dim=-1)
            logits = self.router(r_in)

            routing = F.gumbel_softmax(logits, tau=self.tau, hard=False, dim=-1) if self.training else F.softmax(logits, dim=-1)
            routing_hist.append(routing)

            write = torch.einsum('bns,bnd->bsd', routing, x)
            mem_in = torch.cat([mem, write], dim=-1)
            delta = self.mem_update(mem_in)
            alpha = self.mem_gate(mem_in)
            mem = alpha * delta + (1.0 - alpha) * mem
            mem = F.layer_norm(mem, [self.D])
            mem = torch.tanh(mem) * 0.95

            ctx = torch.einsum('bns,bsd->bnd', routing, mem)
            g_in = torch.cat([x, ctx, t_emb], dim=-1)
            new_x = self.G(g_in)

            stall_prob = torch.sigmoid(self.stall_net(new_x)).squeeze(-1)
            gate = 1.0 - stall_prob if self.training else (stall_prob < 0.5).float()
            x = gate.unsqueeze(-1) * (new_x + 0.15 * x) + (1.0 - gate.unsqueeze(-1)) * x
            x = F.layer_norm(x, [self.D])

        return x, torch.stack(routing_hist)

# =========================
# MAIN MODEL
# =========================
class ElementSlotNet(nn.Module):
    def __init__(self, img_size=28, in_ch=1):
        super().__init__()
        self.patch = PatchEmbedding(img_size, patch_size=4, in_ch=in_ch, embed_dim=64)
        num_patches = (img_size // 4) ** 2
        self.layers = nn.ModuleList([
            ElementSlotLayer(num_patches, num_slots=32, dim=64, ticks=6, hidden=128)
            for _ in range(3)
        ])
        self.norm = nn.LayerNorm(64)
        self.head = nn.Linear(64, 10)

    def set_tau(self, tau):
        for layer in self.layers:
            layer.set_tau(tau)

    def forward(self, x):
        x = self.patch(x)
        routings = []
        for layer in self.layers:
            out, r = layer(x)
            x = x + out
            routings.append(r)
        x = self.norm(x.mean(dim=1))
        logits = self.head(x)
        return logits, routings

# =========================
# ROUTING LOSS
# =========================
def routing_loss(routings, ent_weight=0.015, bal_weight=0.12, var_weight=0.04):
    loss_ent = loss_bal = loss_var = 0.0
    for r in routings:
        p = r.clamp(min=1e-8, max=1.0)
        entropy = -(p * torch.log(p)).sum(dim=-1).mean()
        slot_load = r.sum(dim=1)
        target = torch.full_like(slot_load, r.shape[1] / r.shape[2])
        balance = F.kl_div(slot_load.log_softmax(dim=-1), target.log_softmax(dim=-1), reduction='batchmean')
        variance = slot_load.var(dim=1).mean()
        loss_ent += entropy
        loss_bal += balance
        loss_var += variance
    return -ent_weight * loss_ent + bal_weight * loss_bal + var_weight * loss_var

# =========================
# VALIDATION
# =========================
def validate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits, _ = model(images)
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total

# =========================
# TRAIN / FINE-TUNE
# =========================
def train(finetune=False, weight_filename='best_element_slot.pt'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Device: {device}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weight_path = os.path.join(script_dir, weight_filename)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, 'data')
    
    dataset = torchvision.datasets.MNIST(data_path, train=True, download=True, transform=transform)
    train_set, val_set = random_split(dataset, [55000, 5000], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    model = ElementSlotNet().to(device)

    # 🔹 Load weights if finetune
    start_epoch = 0
    if finetune and os.path.exists(weight_path):
        print(f"♻ Loading pre-trained weights from {weight_path}")
        model.load_state_dict(torch.load(weight_path, map_location=device))
        print("✅ Weights loaded successfully.")

    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4 if finetune else 3e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=6e-4 if finetune else 6e-3, epochs=12, steps_per_epoch=len(train_loader),
        pct_start=0.1, anneal_strategy='cos'
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device == 'cuda'))

    best_acc = 0.0
    patience = 3
    no_improve = 0

    for epoch in range(12):
        model.train()
        model.set_tau(max(1.0 - epoch * 0.085, 0.15))

        correct = total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/12")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device=='cuda')):
                logits, routings = model(images)
                loss_cls = F.cross_entropy(logits, labels)
                loss_route = routing_loss(routings)
                loss = loss_cls + 0.55 * loss_route
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix({'acc': f'{100*correct/total:.2f}%'})

        val_acc = validate(model, val_loader, device)
        print(f"Epoch {epoch+1:2d} | Train Acc: {100*correct/total:.2f}% | Val Acc: {val_acc:.2f}%")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), weight_path)
            no_improve = 0
            print(f"💾 New best saved: {best_acc:.2f}%")
        else:
            no_improve += 1
            if no_improve >= patience:
                print("⏹ Early stopping triggered")
                break

    model.load_state_dict(torch.load(weight_path, map_location=device))
    final_acc = validate(model, val_loader, device)
    print(f"\n✅ Training/Fine-tuning completed! Best Val Acc: {best_acc:.2f}% | Final Val Acc: {final_acc:.2f}%")

if __name__ == "__main__":
    # 🔹 For the first launch: train(finetune=False)
    # 🔹 For additional training: train(finetune=True)
    train(finetune=True)