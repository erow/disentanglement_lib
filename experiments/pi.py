from fastai.text.all import *
from pl_bolts.models.rl.common.networks import MLP

file = open('pi.txt', 'r')
pi = file.read()
file.close()
pi = torch.LongTensor(list(map(int, pi)))

block_size = 32


class Dataset:
    def __len__(self):
        return len(pi) - block_size + 1

    def __getitem__(self, i):
        return pi[i:i + block_size - 1], pi[i + block_size - 1]


defaults.use_cuda = 1


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = MLP([block_size - 1, 1], 10)

    def forward(self, x):
        # x = input[:, :block_size - 1]
        x = x.float()
        return self.ln(x)

    def loss(self, preds, y):
        # x, y = torch.split(input, [block_size - 1, 1], 1)
        return F.cross_entropy(preds, y)


dl = DataLoader(Dataset(), 128)
model = Model()
learner = Learner(DataLoaders(dl, DataLoader([])), model, model.loss)
learner.fit(2)
