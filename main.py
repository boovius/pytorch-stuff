import torch
from torch import nn
from device import device
from models import NeuralNetwork
from data_loader import load_training_data, load_testing_data, labels_map
from train_test_utils import train, test
from plot import plot

model = NeuralNetwork().to(device)

print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

epochs = 2
training_data_loader = load_training_data()
test_data_loader = load_testing_data()

plot(training_data_loader.dataset, labels_map)
plot(test_data_loader.dataset, labels_map)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(training_data, model, loss_fn, optimizer, device)
    test(test_data, model, loss_fn, device)
print("Done!")
