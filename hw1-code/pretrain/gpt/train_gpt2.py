from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # 50000 BPE merges, 256 bytes tokens + 1<endofsentence>
    n_layer: int = 12 # layers
    n_head: int = 12 # heads
    n_embd: int = 768 # embedding dimension

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
         
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class CausalSelfAttention(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        assert  config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
        # not really a bias
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
    
    def forward(self, x):
        B, T, C = x.size()
        
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim = 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0 , float('-inf')) 
        att = F.softmax(att, dim = -1)
        
        y = att @ v # (B. nh, T, T) * (B, nh, T, hs) --> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        y = self.c_proj(y)
        return y

class Block(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    
    def __init__(self, config):
        
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False)
        
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, target=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is {self.config.block_size}"
        
        # forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # shape (T)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb

        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)

        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # shape (B, T, vocab_size)
        
        loss = None
        if target is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        
        return logits, loss
        
    
    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel
        
        print(f"Loading weights from pretrained gpt: {model_type}")
        
        # config
        config_args = {
            "gpt2" : dict(n_layer = 12, n_head=12, n_embd=768), # 124 M
            "gpt2-medium" : dict(n_layer = 24, n_head=16, n_embd=1024), # 350M
            "gpt2-large" : dict(n_layer = 36, n_head=20, n_embd=1280), # 774M
            "gpt2-xl" : dict(n_layer = 48, n_head=25, n_embd=1600), # 1448 M
        }[model_type]
        
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")]
        
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        
        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]  # ignore these
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]  # same, just the masked ones
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear layer
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

class DataLoaderLite:

    def __init__(self, B, T):
        import tiktoken
        self.B = B
        self.T = T
        
        # at init load tokens from disk and store them in memory
        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"Loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")
        
        # state
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)  # inputs
        y = buf[1:].view(B, T)   # targets
        # advance the position in the tensor
        self.current_position += B * T
        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y



if __name__ == "__main__": 
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    print(f"usng device: {device}")
    
    num_return_sequences = 5
    max_length = 30
    
    model = GPT.from_pretrained('gpt2')
    # model = GPT(GPTConfig())
    # model = GPT.from_pretrained('gpt2')
    print("我们第一次实现了gpt2")
    
    model.eval()
    model.to("cuda")
    
    
    
    # import tiktoken

    # enc = tiktoken.get_encoding('gpt2')
    
    ## first
    # tokens = enc.encode("Hello, I'm a language model.")
    # tokens = torch.tensor(tokens, dtype=torch.long)  # (8,)
    # tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  # (5, 8)
    # x = tokens.to('cuda')
    
    
    ## second
    # Get a data batch

    # with open('input.txt', 'r') as f:
    #     text = f.read()
    # text = text[:1000]
    # tokens = enc.encode(text)
    # B, T = 4, 32
    # buf = torch.tensor(tokens[:B*T + 1], device=device)
    # x = buf[:-1].view(B, T)
    # y = buf[1:].view(B, T)
    
    
    # third
    train_loader = DataLoaderLite(B=4, T=32)

    # Get logits
    model = GPT(GPTConfig())
    model.to(device)
    # logits, loss = model(x, y)
    # print(loss)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr = 3e-4)
    for i in range(50):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        print(f"step {i}, loss : {loss.item()}")
    
    # import sys; sys.exit(0)
    
    
    

    # Generate! Right now x is (B, T) where B = 5, T = 8
    # Set the seed to 42
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    while x.size(1) < max_length:
        # Forward the model to get the logits
        with torch.no_grad():
            logits = model(x)  # (B, T, vocab_size)
            # Take the logits at the last position
            logits = logits[:, -1, :]  # (B, vocab_size)
            # Get the probabilities
            probs = F.softmax(logits, dim=-1)

            # Do top-k sampling of 50 (HuggingFace pipeline default)
            # topk_probs here becomes (5, 50), topk_indices is (5, 50)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # Select a token from the top-k probabilities
            ix = torch.multinomial(topk_probs, 1)  # (B, 1)
            # Gather the corresponding indices
            xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
            # Append to the sequence
            x = torch.cat((x, xcol), dim=1)

    import tiktoken
    enc = tiktoken.get_encoding('gpt2') 
    # for i in range(num_return_sequences):
    #     tokens = x[i, :max_length].tolist()
    #     decoded = enc.decode(tokens)
    #     print(">", decoded)
    # 
    for i in range(min(4 , num_return_sequences)):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(">", decoded)
        

