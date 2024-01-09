import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from simple_pinn import (
    SchrodingersEqDataset,
    SimplePINN,
    evalAndCompare,
    calculateComplexDerivatives,
    calculateOperatorLoss,
    mse_loss,
)
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# load the data
dataset = SchrodingersEqDataset("../PINNs/main/Data/NLS.mat")

dl = DataLoader(dataset, batch_size=10_000, shuffle=True)

# create the model
model = SimplePINN(input_size=2, hidden_layers=[20, 20, 20, 20], output_size=2)
model.to(device)
print(model)

optimizer = optim.Adam(model.parameters(), lr=0.01)

n_epochs = 500
for epoch in range(n_epochs):
    for inputs, targets in dl:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        breakpoint()
        data_loss = mse_loss(outputs, targets)
        f_loss = calculateOperatorLoss(outputs, inputs)
        loss = f_loss + data_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # print out epoch, loss, data_loss, and f_loss
    print(
        f"epoch {epoch}: loss={loss:.4f}, data_loss={data_loss:.4f}, f_loss={f_loss:.4f}"
    )
