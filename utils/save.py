import torch
import os

def save_model(model, model_name, folder_path):
    file_path = f"{folder_path}/{model_name}.pth"
    torch.save(model.state_dict(), file_path)

