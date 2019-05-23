#%%
import torch
import torch.nn as nn
import pandas as pd

df = pd.read_csv('data/train.csv')
df.fillna(method='backfill', inplace=True)
x = torch.tensor(df.x, dtype=torch.float)
y = torch.tensor(df.y, dtype=torch.float)
X = torch.ones(len(x), 2)
X[:, 1] = x

# normal equiation
coefs = (X.t() @ X).inverse() @ X.t() @ y
coefs
# %%
# SGD
def mse(y_hat, y):
    return ((y_hat - y) ** 2).mean()

a = torch.tensor([-1.,1], requires_grad=True)

lr_rate = 1e-1

def gradient_descent(X, y):
    y_hat = X @ a
    loss = mse(y_hat, y)
    print(loss)
    loss.backward()
    with torch.no_grad():
        print(a)     
        a.sub_(a.grad * lr_rate)
        print(a)
        a.grad.zero_()

#%%
gradient_descent(X, y)
