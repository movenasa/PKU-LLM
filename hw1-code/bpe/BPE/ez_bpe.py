from base_bpe import Base_Tokenizer, get_stats, merge


class Tokenizer(Base_Tokenizer):
    def __init__(self):
        super().__init__()
        
    def train(self, text, vocab_size, verbose = False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256
        
        merges = {} # (int, int) --> int for encode
        vocab = {byte : bytes([byte]) for byte in range(256)}
        tokens = text.encode(encoding="utf-8")
        ids = list(tokens)

        for i in range(num_merges):
            # 拿到所有的pair对
            stat = get_stats(ids)
            # 找到次数最多的pair
            pair = max(stat, key=stat.get)
            # 找到新生成的词汇表
            idx = 256 + i
            ids = merge(ids, pair, idx)
            # 更新merge表和词汇表
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
        self.merges = merges # use for encode
        self.vocab = vocab # use for decode
                        
    def encode(self, text):
        # 先转换成tokens
        tokens = text.encode(encoding="utf-8")
        ids = list(tokens)
        # 超过 2 个字符才合并
        while len(ids) >= 2:
            stat = get_stats(ids)
            # 查找字典里面出现最多的bpe对，按照那个进行合并
            pair = min(stat, key = lambda p : self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids
    
    def decode(self, byt):
        # 利用vocab 反向解码
        byts = b"".join(self.vocab[id] for id in byt)
        print(byts)
        text = byts.decode(encoding="utf-8", errors="replace")
        return text

if __name__=='__main__':
    p = Tokenizer()
    with open("../manual.txt", "r", encoding = "utf-8") as f:
        data = f.read()
    p.train(data, 1024)
    print(p.encode("hello_world"))
    print(p.decode(p.encode("hello world")))
    print(p.decode(p.encode("hello world")) == "hello world") 