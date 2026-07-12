"""
Teacher MLP: modelo denso que servirá como "profesor" para destilar.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class TeacherMLP(nn.Module):
    """
    MLP denso de 3 capas ocultas. El "profesor" que luego destilaremos.
    """

    def __init__(self, hidden_dims=(784, 512, 256, 10)):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc2 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.fc3 = nn.Linear(hidden_dims[2], hidden_dims[3])

    def forward(self, x: torch.Tensor, return_activations: bool = False):
        x = x.view(x.size(0), -1)  # flatten
        h1 = torch.relu(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        out = self.fc3(h2)
        if return_activations:
            return out, [h1, h2]
        return out

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def load_mnist(batch_size: int = 128, max_samples: int | None = None):
    """Carga MNIST, opcionalmente limitado a `max_samples`."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    if max_samples is not None:
        from torch.utils.data import Subset
        train_dataset = Subset(train_dataset, range(min(max_samples, len(train_dataset))))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train_teacher(model: TeacherMLP, train_loader: DataLoader,
                  epochs: int = 5, lr: float = 1e-3, device: str = 'cpu'):
    """Entrena TeacherMLP en MNIST."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)

        acc = 100.0 * correct / total
        print(f"  Epoch {epoch + 1:2d}/{epochs}  |  loss: {total_loss / total:.4f}  |  acc: {acc:.2f}%")

    return model


@torch.inference_mode()
def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: str = 'cpu'):
    """Evalúa accuracy de un modelo."""
    model.to(device)
    model.eval()
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x = x.view(x.size(0), -1)
        logits = model(x)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)

    return 100.0 * correct / total
