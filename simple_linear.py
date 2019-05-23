import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

df = pd.read_csv('data/train.csv')
df.fillna(method='backfill', inplace=True)
x = torch.from_numpy(df.x.values).float()
y = torch.from_numpy(df.y.values).float()

class SimpleLinear(nn.Module):

    def __init__(self, in_size, out_size):
        super(SimpleLinear, self).__init__()
        self.linear = nn.Linear(in_size, out_size)
    
    def forward(self,x):
        out = self.linear(x)
        return out

model = SimpleLinear(700,700)
criterion = nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(150):
    y_hat = model(x)
    loss = criterion(y_hat, y)
    print(loss)
    loss.backward()
    optimiser.step()
    optimiser.zero_grad()

