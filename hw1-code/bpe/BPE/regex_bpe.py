import regex as re
from base_bpe import Base_Tokenizer, get_stats, merge
from tqdm import tqdm
# 与base_bpe 两处不同
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class RegexTokenizer(Base_Tokenizer):
    def __init__(self, pattern = None):
        super().__init__() # 会继承过来   self.pattern = "" # 后续的正则表 self.special_token = {} # 特殊分隔符表  self.merges = {} # 相关合并表  self.vocab = self._build_vocab() # 词汇表
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.regex_pattern = re.compile(self.pattern)
        self.special_token = {}   # speical_token : str -> {<endoftext> : 100257}
        self.inverse_special_token = {}
        
    def train(self, text, vocab_size, verbose=False): # verbose 为True，是一个调试开关，方便打印一些result 主要是引入正则后的处理
        assert vocab_size >= 256
        # 训练次数
        num_merges = vocab_size - 256
        # 嵌入文本 需要先正则分类一波,然后为每个正则得到嵌入
        chunks = re.findall(self.regex_pattern, text)
        tokens = [list(id.encode("utf-8")) for id in chunks]
        
        merges = {}
        vocab = {byte: bytes([byte]) for byte in range(256)}
        for i in tqdm(range(num_merges), desc="traing the Regextokenizer"):
            # 需要接一下tokens，因为正则完一块一块的，形式发生了变化
            stat = {}
            # print(type(tokens))
            for chunk in tokens:
                mini_stat = get_stats(chunk)
                for key, value in mini_stat.items():
                    # print(f"key , value : {key}, {value}")
                    if key in mini_stat:
                        stat[key] = stat.get(key, 0) + value
                        
            pair = max(stat, key=stat.get)
            #  print(f"pair : {pair}, stat: {stat[(239,188)]}")
            ids = 256 + i

            tokens = [merge(chunk, pair, ids) for chunk in tokens]
            merges[pair] = ids
            vocab[ids] =  vocab[pair[0]] + vocab[pair[1]]
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {ids} ({vocab[ids]}) had {stat[pair]} occurrences")
        self.merges = merges
        self.vocab = vocab
        
    # 定义speical_tokens
    def register_speical_tokens(self, sepcial_tokens):
        # special_tokens is a dictionary of str -> int
        # example: {"<|endoftext|>": 100257}
        self.special_token = sepcial_tokens
        self.inverse_special_token = {k : v for k, v in sepcial_tokens.items()}
    
    def _encode_chunk(self, type_bytes):
        # type_bytes 是一段序列，encode成对应的tokens，采用merges合并
        ids = list(type_bytes)
        while len(ids) >= 2:
            stats = get_stats(type_bytes)
            pair = min(stats, key = lambda p : self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(type_bytes, pair, self.merges[pair])
        return ids 
    
    def encode_text(self, text):
        tokens = []
        text = re.findall(self.regex_pattern, text)
        chunks = [chunk.encode("utf-8") for chunk in text]
        for chunk in chunks:
            ids = self._encode_chunk(chunk)
            tokens.extend(ids)
        return tokens
    
    def decode(self, ids):
        # ids(lists of integers, return Python strings)
        bytes = []
        for idx in ids:
            if idx in self.vocab:
                bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_token:
                bytes.append(self.special_token[idx])
        bytes = b"".join(bytes)
        text = bytes.decode("utf-8", errors="replace")
        return text
    
    def encode(self, text, allow_special = "all"):
        """
        Unlike encode_ordinary, this function handles special tokens.
        allowed_special: can be "all"|"none"|"none_raise" or a custom set of special tokens
        if none_raise, then an error is raised if any special token is encountered in text
        this is the default tiktoken behavior right now as well
        any other behavior is either annoying, or a major footgun
        """
        if allow_special == "all":
            special = self.special_token
        elif allow_special == "none":
            special = {}
        elif allow_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_token)
        '''
        # otherwise, we have to be careful with potential special tokens in text
        # we handle special tokens by splitting the text
        # based on the occurrence of any exact match with any of the special tokens
        # we can use re.split for this. note that surrounding the pattern with ()
        # makes it into a capturing group, so the special tokens will be included
        '''
        # special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_pattern = "(" + "|".join(re.escape(k) for k,v in special.items()) + ")"
        special_chunk = re.split(special_pattern, text)
        
        ids = []
        for id in special_chunk:
            if id in special:
                ids.append(special[id])
            else :
                ids.extend(self.encode_text(id))
        return ids

if __name__ == "__main__":
    special_tokens = {
    '<|endoftext|>': 100257,
    '<|fim_prefix|>': 100258,
    '<|fim_middle|>': 100259,
    '<|fim_suffix|>': 100260,
    '<|endofprompt|>': 100276
    }
    with open("../manual.txt", "r") as f:
        data = f.read()
    llama_text = """
        <|endoftext|>The llama (/ˈlɑːmə/; Spanish pronunciation: [ˈʎama] or [ˈʝama]) (Lama glama) is a domesticated South American camelid, widely used as a meat and pack animal by Andean cultures since the pre-Columbian era.
        Llamas are social animals and live with others as a herd. Their wool is soft and contains only a small amount of lanolin.[2] Llamas can learn simple tasks after a few repetitions. When using a pack, they can carry about 25 to 30% of their body weight for 8 to 13 km (5–8 miles).[3] The name llama (in the past also spelled "lama" or "glama") was adopted by European settlers from native Peruvians.[4]
        The ancestors of llamas are thought to have originated from the Great Plains of North America about 40 million years ago, and subsequently migrated to South America about three million years ago during the Great American Interchange. By the end of the last ice age (10,000–12,000 years ago), camelids were extinct in North America.[3] As of 2007, there were over seven million llamas and alpacas in South America and over 158,000 llamas and 100,000 alpacas, descended from progenitors imported late in the 20th century, in the United States and Canada.[5]
        <|fim_prefix|>In Aymara mythology, llamas are important beings. The Heavenly Llama is said to drink water from the ocean and urinates as it rains.[6] According to Aymara eschatology,<|fim_suffix|> where they come from at the end of time.[6]<|fim_middle|> llamas will return to the water springs and ponds<|endofprompt|>
        """.strip()
    
    tokenizer = RegexTokenizer(pattern=GPT4_SPLIT_PATTERN)
    tokenizer.register_speical_tokens(special_tokens)
    tokenizer.train(data, 1024)
    tokenizer.encode(llama_text)
    
                
                
                