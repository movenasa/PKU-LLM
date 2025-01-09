import os
from os.path import exists
import torch.nn as nn
import torch
from encoder import LayerNorm, clones, SublayerConnection
import pandas as pd
import altair as alt
from util.subsequent import subsequent_mask

def example_mask():
    LS_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "Subsequent Mask" : [subsequent_mask(20)[0][x, y].item()],
                    "Window" : x,
                    "Masking": y,
                }
            )
            for y in range(20)
            for x in range(20)
        ]
    )
    
    return (
        alt.Chart(LS_data)
        .mark_rect()
        .properties(height=250, width=250)
        .encode(
            alt.X("Window:O"),
            alt.Y("Masking:O"),
            alt.Color("Subsequent Mask:Q", scale=alt.Scale(scheme="viridis")),
        )
        .interactive()
    )
# chart = example_mask()
# chart.save("mask_attention.png") 


## As we can see, the decoder is also composed of a stack of N = 6 identical layers
class Decoder(nn.Module):
    """
    Generate N layer decoder with masking
    """
    
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    '''
    Decoder is made of self-attn, src-attn, and feed forward
    '''
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
    
    def forward(self, x, memory, src_mask, tgt_mask):
        '''
        ''' 
        m = memory
        x = self.sublayer[0](x, lambda x : self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x : self.src_attn(x, m, m, src_mask)) # (batch, seq, d_model)
        return self.sublayer[2](x, self.feed_forward)
    
# tokenizer -- > 