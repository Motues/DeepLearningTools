import torch
from torch.xpu import device


def evaluate_model(model, test_loader):
    model.eval()
    correct_num = 0
    wrong_num = 0
    total_num = 0
    step = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    with torch.no_grad():
        for inputs, real_label in test_loader:
            inputs = inputs.to(device)
            real_label = real_label.to(device)
            predicted_labels = model(inputs)
            for i, predicted_label in enumerate(predicted_labels):
                tar = torch.argmax(predicted_label)
                if tar == real_label[i]:
                    correct_num += 1
                else:
                    wrong_num += 1
                    # utils.save_image(x[i], f'../data/MNIST/error/{y[i]}_{tar}_{wrong_num}.png')
                total_num += 1
            step += 1
            if step % 100 == 0:
                print(f'accuracy: {float(correct_num / total_num * 100):f} %')
    print(f'Final accuracy: {float(correct_num / total_num * 100):f} %')

