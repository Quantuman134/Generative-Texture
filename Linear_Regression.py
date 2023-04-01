import torch
import torch.nn as nn
import numpy as np

x_values = [i for i in range(11)]
x_train = np.array(x_values, dtype=np.float32)
x_train = x_train.reshape(-1, 1)
y_values = [2*i+1 for i in x_values]
y_train = np.array(y_values, dtype=np.float32)

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out
    
input_dim = 1
output_dim = 1
model = LinearRegressionModel(input_dim, output_dim)

epochs = 1000
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

for epoch in range(epochs):
    epoch +=1
    
    #transform to tensor
    inputs = torch.from_numpy(x_train)
    labels = torch.from_numpy(y_train)

    #clear grad buffer
    optimizer.zero_grad()

    #forward propogation
    outputs = model(inputs)

    #loss computation
    loss = criterion(outputs, labels)

    #back propogation
    loss.backward()

    #update parameters of model
    optimizer.step()
    if epoch%50 == 0:
        print('epoch = {}, loss = {}'.format(epoch, loss.item()))