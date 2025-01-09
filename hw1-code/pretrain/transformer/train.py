import copy
import torch.nn as nn
from encoder import MultiHeadedAttention, Encoder, EncoderLayer
from decoder import Decoder, DecoderLayer
from util.embeddings import Embeddings
from transformer import  Transformer, Generator
from util.point_wise import PositionwiseFeedForward
from util.position import PositionalEncoding
import time
import torch 
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
import torch.optim as optim
from util.subsequent import subsequent_mask
import pandas as pd
import torch.nn.functional as F



class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src, tgt=None, pad=2):  # 2 = <blank>
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
            tgt_mask.data
        )
        return tgt_mask


# Tokenizer for English and Chinese
class MedicalTokenizer:
    def __init__(self, src_lang="en", tgt_lang="zh"):
        self.src_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.tgt_tokenizer = AutoTokenizer.from_pretrained("/home/wangyuanda/project/Lession-All/LLM/PKU-LLM/hw1-code/pretrain/transformer/tokenizer/bert-base-chinese")

    def encode_src(self, text):
        return self.src_tokenizer.encode(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)

    def encode_tgt(self, text):
        return self.tgt_tokenizer.encode(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)

# Translation Dataset
class TranslationDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        src_text = self.dataset[idx]["translation"]["en"]
        tgt_text = self.dataset[idx]["translation"]["zh"]
        src = self.tokenizer.encode_src(src_text).squeeze()
        tgt = self.tokenizer.encode_tgt(tgt_text).squeeze()
        return src, tgt


class TrainState:
    """Track number of steps, examples, and tokens processed"""

    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed

class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())

# Training loop
def run_epoch(data_iter, model, loss_compute, optimizer, scheduler, mode="train", accum_iter=1, train_state=None):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        out = model.generator(out)  # Apply the Generator here

        # 调整形状
        out = out.contiguous().view(-1, out.size(-1))  # 形状：(batch_size * seq_len, vocab_size)
        target = batch.tgt_y.contiguous().view(-1)     # 形状：(batch_size * seq_len)

        loss = loss_compute(out, target)
        loss.backward()
        
        if i % accum_iter == 0:
            optimizer.step()
            optimizer.zero_grad()
            n_accum += 1
            scheduler.step()
        
        total_loss += loss.item() 
        total_tokens += batch.ntokens
        tokens += batch.ntokens

        if i % 1000 == 1:
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(f"Step: {i}, Loss: {loss.item() / batch.ntokens:.4f}, Tokens / Sec: {tokens / elapsed:.1f}, LR: {lr:.1e}")
            start = time.time()
            tokens = 0

    return total_loss / total_tokens

def collate_batch(batch, device):
    src_batch, tgt_batch = zip(*batch)
    src_batch = torch.stack(src_batch).to(device)  # 移动到指定设备
    tgt_batch = torch.stack(tgt_batch).to(device)  # 移动到指定设备
    return Batch(src_batch, tgt_batch)

# Training main loop
def train_transformer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载 wmt16  英文到中文翻译数据集
    dataset = load_dataset("parquet", data_files="/home/wangyuanda/project/Lession-All/LLM/PKU-LLM/hw1-code/pretrain/transformer/data/train-00000-of-00013.parquet", split="train")
    # dataset = dataset.select(range(10000))  # 只取前 100,000 个样本
    tokenizer = MedicalTokenizer()
    translation_dataset = TranslationDataset(dataset, tokenizer) # 30000 - > 512 # 512 - > 20000 
    data_loader = DataLoader(
        translation_dataset, batch_size=128, shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, device)  # 将 device 传递到 collate_batch
    )
    
    src_vocab = len(tokenizer.src_tokenizer)  # 根据实际的 tokenizer 设置词汇量
    tgt_vocab = len(tokenizer.tgt_tokenizer)
    model = make_model(src_vocab, tgt_vocab).to(device)  # 将模型移动到指定设备

    criterion = LabelSmoothing(size=tgt_vocab, padding_idx=tokenizer.tgt_tokenizer.pad_token_id, smoothing=0.1).to(device)   
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: rate(step, 512, 2, 4000))  # 将 warmup 减少到 1000
    for epoch in range(5):
        model.train()
        total_loss = run_epoch(data_loader, model, criterion, optimizer, scheduler)
        print(f"Epoch {epoch + 1}: Train Loss: {total_loss:.4f}")
    torch.save(model.state_dict(), 'transformer_model.pth')


def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )
    

def make_model(
    src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1
):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = Transformer(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)), # 输入会先通过 Embeddings 模块，然后通过 c(position) 模块
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


def load_model(src_vocab, tgt_vocab, model_path, device):
    model = make_model(src_vocab, tgt_vocab)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

# 定义贪心解码函数
# def greedy_decode(model, src, src_mask, max_len, start_symbol, tokenizer, device):
#     memory = model.encode(src, src_mask)
#     ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
#     for i in range(max_len-1):
#         tgt_mask = subsequent_mask(ys.size(1)).type_as(src.data).to(device)
#         out = model.decode(memory, src_mask,
#                            ys,
#                            tgt_mask)
#         out = model.generator(out[:, -1])
#         prob = F.log_softmax(out, dim=-1)
#         _, next_word = torch.max(prob, dim=1)
#         next_word = next_word.item()
#         ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
#         if next_word == tokenizer.tgt_tokenizer.sep_token_id:
#             break
#     return ys

def beam_search_decode(model, src, src_mask, max_len, start_symbol, tokenizer, device, beam_width=5):
    # 编码源句子
    memory = model.encode(src, src_mask)
    
    # 初始化 Beam，每个 Beam 保存句子序列、累积得分和掩码
    beams = [(torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device), 0.0)]  # (sequence, score)
    
    for _ in range(max_len - 1):
        # 创建一个临时列表存储每个 Beam 扩展后的候选句子
        new_beams = []
        for ys, score in beams:
            tgt_mask = subsequent_mask(ys.size(1)).type_as(src.data).to(device)
            out = model.decode(memory, src_mask, ys, tgt_mask)
            out = model.generator(out[:, -1])
            log_probs = F.log_softmax(out, dim=-1)

            # 从当前分支的最后一步生成 beam_width 个最高概率的下一个词
            top_log_probs, top_indices = torch.topk(log_probs, beam_width)
            
            # 为每个候选词更新候选句子和累积得分
            for k in range(beam_width):
                next_word = top_indices[0, k].item()
                new_seq = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
                new_score = score + top_log_probs[0, k].item()
                new_beams.append((new_seq, new_score))
        
        # 按照得分排序，并保留 top beam_width 个候选句子
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
        
        # 检查是否所有的序列都以 <SEP> 结束，若是则提前停止
        if all(beam[0][-1].item() == tokenizer.tgt_tokenizer.sep_token_id for beam in beams if beam[0].size(0) > 0):
            break
    
    # 选择得分最高的句子作为最终输出
    best_seq = beams[0][0]
    return best_seq


# 翻译句子
def translate_sentence(model, sentence, tokenizer, device, max_len=128, src_mask=None):
    if isinstance(sentence, torch.Tensor):
        src = sentence.to(device)
        src = src[:, :max_len]
        if src_mask is None:
            src_mask = (src != tokenizer.src_tokenizer.pad_token_id).unsqueeze(-2)
        else:
            src_mask = src_mask.to(device)
    else:
        # 编码源句子
        src = tokenizer.encode_src(sentence).to(device)
        src = src[:, :max_len]
        src_mask = (src != tokenizer.src_tokenizer.pad_token_id).unsqueeze(-2)
    
    # 开始符号（根据目标词汇表的开始符号）
    start_symbol = tokenizer.tgt_tokenizer.cls_token_id
    
    # 使用解码函数生成目标句子
    # tgt_tokens = greedy_decode(model, src, src_mask, max_len, start_symbol, tokenizer, device).flatten()
    tgt_tokens = beam_search_decode(model, src, src_mask, max_len, start_symbol, tokenizer, device).flatten()

    # 将生成的令牌转换回文本
    tgt_sentence = tokenizer.tgt_tokenizer.decode(tgt_tokens.cpu().numpy(), skip_special_tokens=True)
    return tgt_sentence

from sacrebleu import corpus_bleu

def compute_bleu_score(model, tokenizer, device, data_loader):
    model.eval()
    references = []
    hypotheses = []

    with torch.no_grad():
        for batch in data_loader:
            src, tgt = batch.src, batch.tgt_y
            src_mask = batch.src_mask
            for i in range(src.size(0)):
                src_sentence = src[i].unsqueeze(0)
                src_mask_sentence = src_mask[i].unsqueeze(0)
                translation = translate_sentence(
                    model,
                    sentence=src_sentence,
                    tokenizer=tokenizer,
                    device=device,
                    max_len=128,
                    src_mask=src_mask_sentence
                )
                hypothesis = translation.strip()
                reference = tokenizer.tgt_tokenizer.decode(tgt[i].cpu().numpy(), skip_special_tokens=True).strip()
                hypotheses.append(hypothesis)
                references.append([reference])

    bleu = corpus_bleu(hypotheses, references)
    print(f"BLEU Score: {bleu.score:.2f}")
    return bleu.score

# 推理函数
def inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化 tokenizer
    tokenizer = MedicalTokenizer()
    src_vocab = tokenizer.src_tokenizer.vocab_size
    tgt_vocab = tokenizer.tgt_tokenizer.vocab_size
    
    # 加载模型
    model = load_model(src_vocab, tgt_vocab, 'transformer_model.pth', device)
    
    # 输入句子
    # sentence = "Hello, how are you?"
    sentence = "PARIS – As the economic crisis deepens and widens, the world has been searching for historical analogies to help us understand what has been happening." 
    # 生成翻译
    translation = translate_sentence(model, sentence, tokenizer, device)
    print(f"原始句子: {sentence}")
    print(f"翻译结果: {translation}")


# 开始训练
# train_transformer()
# inference()
Trans = make_model(512, 1024)


# device = "cuda" if torch.cuda.is_available() else "cpu"
# dataset = load_dataset("parquet", data_files="/home/wangyuanda/project/Lession-All/LLM/PKU-LLM/hw1-code/pretrain/transformer/data/test-00000-of-00001.parquet", split="train")
# # dataset = dataset.select(range(10000))  # 只取前 100,000 个样本
# tokenizer = MedicalTokenizer()
# translation_dataset = TranslationDataset(dataset, tokenizer)
# data_loader = DataLoader(
#     translation_dataset, batch_size=128, shuffle=True,
#     collate_fn=lambda batch: collate_batch(batch, device)  # 将 device 传递到 collate_batch
# )
# src_vocab = tokenizer.src_tokenizer.vocab_size
# tgt_vocab = tokenizer.tgt_tokenizer.vocab_size
    
# # 加载模型
# model = load_model(src_vocab, tgt_vocab, 'transformer_model.pth', device)
# compute_bleu_score(model, tokenizer,  device, data_loader)