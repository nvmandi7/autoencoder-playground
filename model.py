import torch
from torch import nn

# Model
class BirbVAE(nn.Module):
    def __init__(self, input_size):
        super(BirbVAE, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device\n")
    
    model = BirbVAE(input_size=48).to(device)
    print(model, "\n")


    X = torch.rand(48, device=device)
    logits = model(X)
    pred_probab = nn.Softmax(dim=0)(logits)
    y_pred = pred_probab.argmax()
    print(f"Predicted class: {y_pred}")



