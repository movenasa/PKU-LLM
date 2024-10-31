import unicodedata

'''
定义基类bpe-tokenizer，后面的都要继承自这个基类tokenizer
'''

'''
获取二元字符数量的统计函数
'''
def get_stats(ids):
    merges = {}
    for pair in zip(ids, ids[1:]):
       merges[pair] = merges.get(pair, 0) + 1 
    return merges

'''
三个输入: 原序列,pair对,合并的idx
操作: 遍历原序列,将pair对替换成idx
输出: 返回新的替换后的序列
'''
def merge(ids, pair, idx):
    newids = []
    for i in range(len(ids)):
        if i + 1 < len(ids) and pair[0] == ids[i] and pair[1] == ids[i + 1]:
            newids.append(idx)
            i += 2
        else :
            newids.append(ids[i])
            i += 1
    return newids

class Base_Tokenizer:
    
    def __init__(self):
        self.pattern = "" # 后续的正则表
        self.special_token = {} # 特殊分隔符表
        self.merges = {} # 相关合并表
        self.vocab = self._build_vocab() # 词汇表
    
    def train(self):
        raise NotImplementedError
    
    def encode(self):
        raise NotImplementedError
    
    def decode(self):
        raise NotImplementedError
    
    def _build_vocab(self):
        vocab = {byte : bytes([byte]) for byte in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1] # case 257 : -> 132,64 vocab[257] = vocab[132] + vacab[64]
        # 处理speical token
        for speical, idx in self.special_token.items():
            vocab[idx] = speical.encode(encoding = "utf-8")
        return vocab
    def load(self):
        pass
        
    def save(self): 
        pass
    
        