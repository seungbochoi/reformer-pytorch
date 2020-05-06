from reformer_pytorch import ReformerLM
from reformer_pytorch.generative_tools import TrainingWrapper

import random
import tqdm
import gzip
import numpy as np
import torch
import shutil
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
# import apex

# constants

NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 1e-4
VALIDATE_EVERY = 3
GENERATE_EVERY = 500  # validation
GENERATE_LENGTH = 512  # validation
SEQ_LEN = 4096


# helpers

def cycle(loader):
    while True:
        for data in loader:
            yield data


def decode_token(token):
    return str(chr(max(32, token)))


def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))


# instantiate model

model = ReformerLM(
    dim=512,
    depth=6,
    max_seq_len=SEQ_LEN,
    num_tokens=256,
    heads=8,
    bucket_size=64,
    n_hashes=4,
    ff_chunks=10,
    lsh_dropout=0.1,
    weight_tie=True,
    causal=True,
    n_local_attn_heads=4,
    use_full_attn=False  # set this to true for comparison with full attention
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 파이토치식으로 리턴이 뭐나올지 정해주는거야.
# if torch.cuda.device_count() > 1:
#     print("USE", torch.cuda.device_count(), "GPUs!!")
#     model = torch.nn.DataParallel(model)

model = TrainingWrapper(model)
model.to(device)
# model.cuda()

# prepare enwik8 data

with gzip.open('./data/enwik8.gz') as file:
    X = np.fromstring(file.read(int(95e6)), dtype=np.uint8)
    trX, vaX = np.split(X, [int(90e6)])
    data_train, data_val = torch.from_numpy(trX), torch.from_numpy(vaX)


class TextSamplerDataset(Dataset):  # Dataset class를 상속하는것
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len - 1, (1,))
        # 4096 +1
        # 그래서 나중에 로쓰구할때
        # 0-4095 로 1 - 4096 만들고 트루랑 비교해서 퍼플렉시티 구할꺼야
        full_seq = self.data[rand_start: rand_start + self.seq_len + 1].long()  # float64 국룰!
        return full_seq.cuda()  # gpu로 넘기는것
        # return full_seq

    def __len__(self):  # 전체 데이타 사이즈(4096), 에폭 몇번돌릴지 이거로 측정한다
        return self.data.size(0) // self.seq_len


train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset = TextSamplerDataset(data_val, SEQ_LEN)
train_loader = cycle(DataLoader(train_dataset, batch_size=BATCH_SIZE))
val_loader = cycle(DataLoader(val_dataset, batch_size=BATCH_SIZE))  # 4 by 4096

# optimizer

optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


def save_ckp(state, is_best, checkpoint_path, best_model_path):
    '''
    :param state: checkpoint we want to save
    :param is_best: indicator to save the best checkpoint; min valid loss
    :param checkpoint_path: path to save checkpoint per epoch
    :param best_model_path:  path to save BEST model param
    '''
    torch.save(state, checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, best_model_path)


def save_losses_history(train_loss, val_loss):
    f = open("/home/bizon/param_reformer/losses_history_0505_B1_without_clamp.txt", 'a')
    f.write(f'train_loss: {train_loss}, val_loss:{val_loss}\n')
    f.close()


# training
# NUM_BATCHES <- 이거걍 이폭으로봐라

prev_best_loss = float('inf')
checkpoint_path = "/home/bizon/param_reformer/model0505_B1_without_clamp.pth"
best_model_path = "/home/bizon/param_reformer/best_model0505_B1_without_clamp.pth"
save_loss_history_path  = "/home/bizon/param_reformer/losses_history_0505_B1_without_clamp.txt"
prev_best_loss_epoch = -1

#
# checkpoint = torch.load("/home/bizon/param_reformer/best_model0505_B1_without_clamp.pth")
# model.load_state_dict(checkpoint['state_dict'])
# optim.load_state_dict(checkpoint['optimizer'])
# prev_best_loss_epoch = checkpoint['epoch']
# prev_best_loss = checkpoint['valid_loss_min']
# print(checkpoint.keys())
print(f'best val loss was {prev_best_loss}, at epoch {prev_best_loss_epoch}')
for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
    if i <= prev_best_loss_epoch: continue
    model.train()  # 트레인전 세팅해주는겨

    for __ in range(GRADIENT_ACCUMULATE_EVERY):  # 한 이터당 4번할껴 ㅎㅎ 이것도 논문인듯?
        train_loss = model(next(train_loader), return_loss=True)  # 매번 랜덤하게 다른 4*4096 셋이 들어온다.
        # 그라디언트가 4번 어큠된다
        # train_loss = train_loss.mean()  # 이래도되나?? train_loss.shape = [2]
        train_loss.backward()  # calc derivative ! (weight update는 아직안함) - 그라디언트 구해라

    print(f'training loss: {train_loss.item()}')
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # 그라디언트 클리핑
    optim.step()  # 웨이트 업데이트 W2 -= W1 +lr*(...)
    optim.zero_grad()  # 그라디언트 초기화!

    if i % VALIDATE_EVERY == 0:
        model.eval()
        with torch.no_grad():
            loss = model(next(val_loader), return_loss=True)
            # loss = loss.mean()
            checkpoint = {
                'epoch': i,
                'valid_loss_min': loss,
                'state_dict': model.state_dict(),
                'optimizer': optim.state_dict(),
                'dim': 512,
                'depth': 6,
                'max_seq_len': SEQ_LEN,
                'num_tokens': 256,
                'heads': 8,
                'bucket_size': 64,
                'n_hashes': 4,
                'ff_chunks': 10,
                'lsh_dropout': 0.1,
                'weight_tie': True,
                'causal': True,
                'n_local_attn_heads': 4,
                'use_full_attn': False,
                }
            save_ckp(checkpoint, False, checkpoint_path, best_model_path)

            # save the best param !
            if prev_best_loss > loss:
                # store the param
                print('  found lower val.loss!! from {:.6f} -> {:.6f}.'.format(prev_best_loss, loss))
                print('  save model...')
                save_ckp(checkpoint, True, checkpoint_path, best_model_path)
                prev_best_loss = loss
            print(f'curr validation loss: {loss.item()}, best val loss so far: {prev_best_loss}')

            # save val_loss, train_loss @ every 3 epoch
            save_losses_history(train_loss.item(), loss.item())

    if i % GENERATE_EVERY == 0:
        model.eval()
        inp = random.choice(val_dataset)[:-1]
        prime = decode_tokens(inp)
        print(f'%s \n\n %s', (prime, '*' * 100))

        sample = model.generate(inp, GENERATE_LENGTH)
        output_str = decode_tokens(sample)
        print(output_str)

if __name__ == "__main__":
    pass
