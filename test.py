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
    mseLoss,
    calculateBoundaryLoss,
)
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# load the data
dataset = SchrodingersEqDataset("../PINNs/main/Data/NLS.mat", points_to_sample=10_000)

dl = DataLoader(dataset, batch_size=10_000, shuffle=True)

# create the model
model = SimplePINN(input_size=2, hidden_layers=[100, 100, 100, 100, 100], output_size=2)
model.to(device)
print(model)

optimizer = optim.Adam(model.parameters(), lr=0.0001)

n_epochs = 500
for epoch in range(n_epochs):
    epoch_loss = 0.0
    epoch_f_loss = 0.0
    epoch_init_loss = 0.0
    epoch_bnd_loss = 0.0
    for inputs in dl:
        # calculate the model output on this batch's collocation points
        inputs = inputs.to(device)
        outputs = model(inputs)
        f_loss = calculateOperatorLoss(outputs, inputs)
        # calculate the model output on this batch's initial condition data
        inputs, targets = dataset.getRandomInitSoln(100)
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        init_loss = mseLoss(outputs, targets)
        # calculate the model output on this batch's boundary conditions
        left_inputs, right_inputs = dataset.getRandomBoundaryPoints(100)
        left_inputs, right_inputs = (
            left_inputs.to(device),
            right_inputs.to(device),
        )
        left_outputs = model(left_inputs)
        right_outputs = model(right_inputs)
        bnd_loss = calculateBoundaryLoss(
            left_inputs, left_outputs, right_inputs, right_outputs
        )
        # sum up the losses
        loss = f_loss + init_loss + bnd_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_f_loss += f_loss.item()
        epoch_init_loss += init_loss.item()
        epoch_bnd_loss += bnd_loss.item()
    # print out epoch, loss, init_loss, and f_loss
    epoch_loss /= len(dl)
    epoch_f_loss /= len(dl)
    epoch_init_loss /= len(dl)
    epoch_bnd_loss /= len(dl)
    print(
        f"epoch {epoch}: loss={epoch_loss:.4f}, f_loss={epoch_f_loss:.4f}, init_loss={init_loss:.4f}, bnd_loss={epoch_bnd_loss:.4f}",
    )
