import numpy
import pandas as pd
import torch

from disentanglement_lib.data.ground_truth import dsprites

data = []
bce = torch.nn.BCELoss(reduction='sum')
for i in range(1, 6):
    action = dsprites.DSprites([i])
    num = action.factors_num_values[0]
    images = torch.FloatTensor([action[idx][0] for idx in range(num)])
    images = torch.FloatTensor(images.reshape(num, -1))
    p = torch.mean(images, 0, keepdim=True).repeat([num, 1])
    H = bce(images, p).item() / num
    data.append([i, H])

df = pd.DataFrame(data, columns=['action', 'entropy'])
df.to_csv('action_entropy.csv')
print(df)
