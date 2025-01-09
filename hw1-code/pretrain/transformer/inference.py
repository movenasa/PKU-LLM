from train import make_model
import torch
from decoder import subsequent_mask

RUN_EXAMPLES = True

def show_example(fn, args=[]):
    if __name__ == "__main__" and RUN_EXAMPLES:
        return fn(*args)

def inference_test():
    test_model = make_model(11, 11, 2) 
    test_model.eval()
    
    #  a b c d e f g h i j k l
    src = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]) # (1, 11) batch, seq
    src_mask = torch.ones(1, 1, 11) # (1, 1, 10)

    memory = test_model.encode(src, src_mask)
    # print(memory.shape) 
    ys = torch.zeros(1, 1).type_as(src) #  ys 是用于存储解码器生成的输出序列的张量

    for i in range(10):
        out = test_model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = test_model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )

    print("Example Untrained Model Prediction:", ys)


def run_tests():
    for _ in range(10):
        inference_test()


show_example(run_tests)