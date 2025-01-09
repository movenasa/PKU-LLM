#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
huggingface 教程
单机单卡（调试模式）：
  $ python train_flash_shakespeare.py --batch_size=8 --max_iters=200

单机多卡（例如 4 卡）：
  $ torchrun --standalone --nproc_per_node=4 train_flash_shakespeare.py --batch_size=8 --max_iters=200

多机(2 节点，每个节点 8 卡)：
  - 假设 master 节点 IP 123.456.123.456，端口 1234
    master 节点:
      $ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train_flash_shakespeare.py
    worker 节点:
      $ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train_flash_shakespeare.py
"""

import os
import time
import math
import argparse
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from contextlib import nullcontext
from dataclasses import dataclass

# --------------------------- 1. 数据预处理部分 ---------------------------
def prepare_shakespeare_data(data_dir: str):
    """
    将 data_dir 目录下的 train.txt 和 val.txt 进行简单的 GPT-2 分词，
    并生成 train.bin 和 val.bin 文件，便于后续训练加载。
    """
    train_path = os.path.join(data_dir, "train.txt")
    val_path = os.path.join(data_dir, "val.txt")
    train_bin_path = os.path.join(data_dir, "train.bin")
    val_bin_path = os.path.join(data_dir, "val.bin")
    meta_path = os.path.join(data_dir, "meta.pkl")

    if not os.path.isfile(train_path) or not os.path.isfile(val_path):
        raise FileNotFoundError(
            f"请确保 {train_path} 和 {val_path} 存在！")

    # 这里使用 transformers 的 GPT2TokenizerFast
    try:
        from transformers import GPT2TokenizerFast
    except ImportError:
        raise ImportError("请先安装 transformers 库: pip install transformers")

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    def read_txt_and_tokenize(txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            data = f.read()
        # 分词为 token id
        enc = tokenizer.encode(data)
        return np.array(enc, dtype=np.uint16)  # GPT-2 范围 <= 50257, uint16 足够

    # 处理 train
    print(f"[DataPrep] 正在处理 {train_path} ...")
    train_ids = read_txt_and_tokenize(train_path)
    print(f"[DataPrep] 训练集 tokens: {len(train_ids):,}")

    # 处理 val
    print(f"[DataPrep] 正在处理 {val_path} ...")
    val_ids = read_txt_and_tokenize(val_path)
    print(f"[DataPrep] 验证集 tokens: {len(val_ids):,}")

    # 保存
    print(f"[DataPrep] 保存到 {train_bin_path}, {val_bin_path}")
    train_ids.tofile(train_bin_path)
    val_ids.tofile(val_bin_path)

    # 也可在 meta.pkl 里存储 vocab_size, eot_token 等信息
    meta = {
        'vocab_size': tokenizer.vocab_size,
        'eot_token': tokenizer.eos_token_id,
        'model_type': 'gpt2',
    }
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)

    print("[DataPrep] 数据预处理完成！")


# --------------------------- 2. 模型与 Flash Attention 定义 ---------------------------
@dataclass
class GPTConfig:
    block_size: int = 1024   # max sequence length
    vocab_size: int = 50257  # GPT-2 默认词表大小
    n_layer: int = 12        # transformer blocks
    n_head: int = 12         # multi-head
    n_embd: int = 768        # embedding size
    dropout: float = 0.0
    bias: bool = False       # 是否在 Linear, LayerNorm 使用 bias


class FlashCausalSelfAttention(nn.Module):
    """
    使用 PyTorch 2.x 的 scaled_dot_product_attention + is_causal=True，
    自动启用 Memory-Efficient / Flash Attention。
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd 必须能被 n_head 整除"
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = config.dropout

    def forward(self, x):
        B, T, C = x.size()

        # 拆分 Q, K, V
        qkv = self.c_attn(x)  # (B, T, 3C)
        q, k, v = qkv.split(C, dim=2)  # 分成 3 份，每份 (B, T, C)

        # 变形多头 (B, T, n_head*head_dim) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # 使用 PyTorch 2.x 的 Flash / Memory-Efficient Attention
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True
        )
        # 输出形状 (B, n_head, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 输出投影
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, elementwise_affine=config.bias)
        self.attn = FlashCausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, elementwise_affine=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config.vocab_size, config.n_embd),
            'wpe': nn.Embedding(config.block_size, config.n_embd),
            'drop': nn.Dropout(config.dropout),
            'h': nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            'ln_f': nn.LayerNorm(config.n_embd, elementwise_affine=config.bias),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Embedding 权重共享
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.block_size, "输入长度超过 block_size!"

        # token + position embedding
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx)    # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)    # (T, n_embd)
        x = tok_emb + pos_emb.unsqueeze(0)     # (B, T, n_embd)

        x = self.transformer.drop(x)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # (B,T,vocab_size) -> (B*T,vocab_size)；targets -> (B*T)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """简单配置 AdamW。"""
        decay = set()
        no_decay = set()
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                if not p.requires_grad:
                    continue
                full_name = f"{mn}.{pn}" if mn else pn
                if full_name.endswith("bias"):
                    no_decay.add(full_name)
                elif 'ln_' in full_name or 'LayerNorm' in full_name:
                    no_decay.add(full_name)
                else:
                    decay.add(full_name)
        
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        valid_decay = [pn for pn in decay if pn in param_dict]
        valid_no_decay = [pn for pn in no_decay if pn in param_dict]
        assert not (decay & no_decay)
        assert not (param_dict.keys() - (decay | no_decay))

        optim_groups = [
            {
                'params': [param_dict[pn] for pn in sorted(valid_decay)],
                'weight_decay': weight_decay
            },
            {
                'params': [param_dict[pn] for pn in sorted(valid_no_decay)],
                'weight_decay': 0.0
            },
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    def estimate_mfu(self, batch_size, dt):
        """
        与 Karpathy 代码类似，简单估算 "Model FLOPS Utilization" (MFU)。
        """
        # 这里做一个大致估算，具体数字仅供参考
        n_layer = self.config.n_layer
        n_embd = self.config.n_embd
        seq_len = self.config.block_size
        # 每 token FLOP 大约 2 * n_layer * (n_embd^2)
        flop_per_token = 2.0 * n_layer * (n_embd * n_embd)
        # 每步处理 batch_size * seq_len 个 token
        flops_per_iter = flop_per_token * (batch_size * seq_len)
        tflops_per_iter = flops_per_iter / 1e12
        # 假设 GPU 125 TFLOPs，dt 秒一 iter
        tflops = tflops_per_iter / dt
        return tflops / 125


# --------------------------- 3. 辅助函数 & 训练循环 ---------------------------

def get_batch(split, data_dir, block_size, batch_size, device, device_type):
    """
    从 .bin 文件里随机采样 batch。
    """
    bin_path = os.path.join(data_dir, f"{split}.bin")
    data = np.memmap(bin_path, dtype=np.uint16, mode='r')
    # 随机起点
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([
        torch.from_numpy((data[i : i+block_size]).astype(np.int64)) for i in ix
    ])
    y = torch.stack([
        torch.from_numpy((data[i+1 : i+1+block_size]).astype(np.int64)) for i in ix
    ])

    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, eval_iters, block_size, batch_size, data_dir, device, device_type):
    """
    评估 train/val loss 的小函数，用多 batch 求平均。
    """
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            X, Y = get_batch(split, data_dir, block_size, batch_size, device, device_type)
            _, loss = model(X, Y)
            losses[k] = loss
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/',
                        help="莎士比亚数据所在目录，包含 train.txt, val.txt 等")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--block_size', type=int, default=256)
    parser.add_argument('--max_iters', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.99)
    parser.add_argument('--eval_interval', type=int, default=200)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--eval_iters', type=int, default=10)
    parser.add_argument('--out_dir', type=str, default='out')
    parser.add_argument('--resume', action='store_true',
                        help="uesd, make back training from out_dir/ckpt.pt")
    parser.add_argument('--compile', action='store_true',
                        help="use PyTorch 2.0 的 torch.compile() accelerate")
    parser.add_argument('--backend', type=str, default='nccl',
                        help="DDP backend 类型，通常 nccl/gloo")
    args = parser.parse_args()

    # 先检查并生成数据
    train_bin = os.path.join(args.data_dir, "train.bin")
    val_bin = os.path.join(args.data_dir, "val.bin")
    if not (os.path.exists(train_bin) and os.path.exists(val_bin)):
        print("未找到 train.bin / val.bin，先执行数据预处理 ...")
        prepare_shakespeare_data(args.data_dir)

    # 分布式训练检测
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        init_process_group(backend=args.backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = (ddp_rank == 0)
        seed_offset = ddp_rank
    else:
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    device_type = 'cuda' if 'cuda' in device else 'cpu'
    if master_process:
        os.makedirs(args.out_dir, exist_ok=True)

    # 设置随机种子、加速策略
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # 构建 GPT 配置
    config = GPTConfig(
        block_size=args.block_size,
        # 这里可以根据实际 meta.pkl 中 vocab_size 决定
        vocab_size=50257,  
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.1,
        bias=False
    )
    model = GPT(config)
    model.to(device)

    # 是否从 ckpt 恢复
    ckpt_path = os.path.join(args.out_dir, 'ckpt.pt')
    iter_num = 0
    best_val_loss = 1e9
    if args.resume and os.path.exists(ckpt_path):
        print(f"从 {ckpt_path} 恢复训练 ...")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
    else:
        checkpoint = None

    # 优化器
    optimizer = model.configure_optimizers(
        args.weight_decay, args.learning_rate, (args.beta1, args.beta2), device_type
    )
    # 若恢复，需要加载优化器状态
    if checkpoint is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    checkpoint = None  # 释放

    # 是否开启编译
    if args.compile:
        print("[INFO] 使用 torch.compile() 编译模型...")
        model = torch.compile(model)

    # DDP
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    # AMP 训练
    scaler = torch.cuda.amp.GradScaler(enabled=(device_type=='cuda'))

    running_mfu = -1.0
    t0 = time.time()

    while True:
        # 取一个 batch
        X, Y = get_batch('train', args.data_dir, args.block_size, args.batch_size, device, device_type)

        # 前向+反向
        with torch.cuda.amp.autocast(enabled=(device_type=='cuda')):
            logits, loss = model(X, Y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        iter_num += 1

        # 日志打印
        if iter_num % args.log_interval == 0 and master_process:
            dt = time.time() - t0
            t0 = time.time()
            lossf = loss.item()
            if iter_num >= 5:
                # 估算 MFU
                raw_model = model.module if ddp else model
                mfu = raw_model.estimate_mfu(args.batch_size, dt)
                running_mfu = mfu if running_mfu < 0 else 0.9 * running_mfu + 0.1 * mfu
            else:
                running_mfu = -1.0
            print(f"iter {iter_num}, loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")

        # eval & save
        if iter_num % args.eval_interval == 0 and master_process:
            raw_model = model.module if ddp else model
            losses = estimate_loss(raw_model, args.eval_iters, args.block_size,
                                   args.batch_size, args.data_dir, device, device_type)
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            val_loss = losses['val']
            if val_loss < best_val_loss:
                best_val_loss = val_loss

            # 无论是否优于 best_val_loss，都可以保存
            ckpt = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iter_num': iter_num,
                'best_val_loss': best_val_loss
            }
            torch.save(ckpt, ckpt_path)
            print(f"[INFO] checkpoint 已保存到 {ckpt_path}")

        # 终止条件
        if iter_num >= args.max_iters:
            break

    if ddp:
        destroy_process_group()
    print("[INFO] 训练完成！")


if __name__ == "__main__":
    main()