import torch.optim as optim
import torch.nn as nn
import torch


def get_optimizer(optimizer_name, model, learning_rate):
    if optimizer_name == "SGD":
        return optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_name == "Momentum":
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_name == "Adagrad":
        return optim.Adagrad(model.parameters(), lr=learning_rate)
    elif optimizer_name == "RMSprop":
        return optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.99)
    elif optimizer_name == "Adam":
        return optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    elif optimizer_name == "AdamW":
        return optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    elif optimizer_name == "Adadelta":
        return optim.Adadelta(model.parameters(), rho=0.95)
    elif optimizer_name == "Adamax":
        return optim.Adamax(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

def get_loss_function(loss_name):
    if loss_name == "MSELoss":
        return nn.MSELoss()
    elif loss_name == "CrossEntropyLoss":
        return nn.CrossEntropyLoss()
    elif loss_name == "BCELoss":
        return nn.BCELoss()
    elif loss_name == "L1Loss":
        return nn.L1Loss()
    elif loss_name == "SmoothL1Loss":
        return nn.SmoothL1Loss()
    elif loss_name == "NLLLoss":
        return nn.NLLLoss()
    elif loss_name == "BCEWithLogitsLoss":
        return nn.BCEWithLogitsLoss()
    elif loss_name == "HingeEmbeddingLoss":
        return nn.HingeEmbeddingLoss()
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")

def train_model(model, train_loader, learning_rate, loss_name, optimizer_name, epoch_number):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimzer = get_optimizer(optimizer_name, model, learning_rate)
    criterion = get_loss_function(loss_name)

    print("Start training...")
    print(f"Training on {device}.")
    step = 0

    for epoch in range(epoch_number):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimzer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimzer.step()
            step += 1
            if step % 500 == 0:
                print(f'loss: {float(loss.sum().item()):f} %')
        print(f'epoch: {epoch + 1}/{epoch_number}')

