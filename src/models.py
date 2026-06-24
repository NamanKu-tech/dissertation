"""Custom models not shipped with ByzFL, registered into byzfl's model namespace.

Import this module BEFORE constructing any ByzFL Client/Server that uses a
model defined here. The registration is a one-time side-effect at import time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import byzfl.fed_framework.models as _byz_models


class mlp3_mnist(nn.Module):
    """Three-layer MLP for MNIST: 784 → 200 → 100 → 10, ReLU + log_softmax.

    Matches the architecture class used in the FedLAW paper (arXiv 2511.03529)
    for MNIST experiments. ByzFL ships only fc_mnist (2-layer) and cnn_mnist;
    we inject this one so ByzFL's Client/Server can look it up by name.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(self.fc3(x), dim=1)


# Register in ByzFL's model namespace so Client/Server can find it by name.
if not hasattr(_byz_models, "mlp3_mnist"):
    _byz_models.mlp3_mnist = mlp3_mnist
