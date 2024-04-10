import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, TensorDataset

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model year', 'Origin']

df = pd.read_csv(
    url, 
    names=column_names, 
    na_values='?', 
    comment='\t', 
    sep=" ", 
    skipinitialspace=True
)

## Drop NA rows ##
df = df.dropna()
df = df.reset_index(drop=True)

## train/test split ##
df_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

train_stats = df_train.describe().transpose()
numeric_cols = ['Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration']

df_train_norm, df_test_norm = df_train.copy(), df_test.copy()

for col_name in numeric_cols:
    mean = train_stats.loc[col_name, 'mean']
    std = train_stats.loc[col_name, 'std']
    df_train_norm.loc[:, col_name] = (df_train_norm.loc[:, col_name] - mean) / std
    df_test_norm.loc[:, col_name] = (df_test_norm.loc[:, col_name] - mean) / std

## Bucketize model years ##
boundaries = torch.tensor([73, 76, 79])
values = torch.tensor(df_train_norm['Model year'].values)
model_year_bucketed = 'Model year bucketed'
df_train_norm[model_year_bucketed] = torch.bucketize(
    values, 
    boundaries, 
    right=True
)

values = torch.tensor(df_test_norm['Model year'].values)

df_test_norm[model_year_bucketed] = torch.bucketize(
    values, 
    boundaries, 
    right=True
)
numeric_cols.append(model_year_bucketed)

## Onehot encode Origin column ##
total_origins = len(set(df_train_norm['Origin']))

# x_train encoding
origin_encoded = one_hot(torch.from_numpy(df_train_norm['Origin'].values) % total_origins)
x_train_numeric = torch.tensor(df_train_norm[numeric_cols].values)
x_train = torch.cat([x_train_numeric, origin_encoded], 1).float()
print(origin_encoded)
# x_test encoding
origin_encoded = one_hot(torch.from_numpy(df_test_norm['Origin'].values) % total_origins)
x_test_numeric = torch.tensor(df_test_norm[numeric_cols].values)
x_test = torch.cat([x_test_numeric, origin_encoded], 1).float()

# y labels
y_train = torch.tensor(df_train_norm['MPG'].values).float()
y_test = torch.tensor(df_test_norm['MPG'].values).float()

train_ds = TensorDataset(x_train, y_train)
batch_size = 8
torch.manual_seed(1)

train_dl = DataLoader(train_ds, batch_size, shuffle=True)

hidden_units = [8, 4]
input_size = x_train.shape[1]

all_layers = []

for hidden_unit in hidden_units:
    layer = nn.Linear(input_size, hidden_unit)
    all_layers.append(layer)
    all_layers.append(nn.ReLU())
    input_size = hidden_unit

all_layers.append(nn.Linear(hidden_units[-1], 1))

model = nn.Sequential(*all_layers)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

torch.manual_seed(1)
num_epochs = 200
log_epochs = 20

for epoch in range(num_epochs):
    loss_hist_train = 0
    for x_batch, y_batch in train_dl:
        pred = model(x_batch)[:, 0]
        loss = loss_fn(pred, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_hist_train += loss.item()

    if epoch % log_epochs == 0:
        print(f'Epoch {epoch} Loss '
              f'{loss_hist_train/len(train_dl):.4f}')
        
with torch.no_grad():
    pred = model(x_test)[:, 0]
    loss = loss_fn(pred, y_test)
    print(f'Test MSE: {loss.item():.4f}')
    print(f'Test MAE: {nn.L1Loss()(pred, y_test).item():.4f}')