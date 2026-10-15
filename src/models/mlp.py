"""A multilayer perceptron on raw pixels.

TASK 2. Build `MLP`: a fully connected network that maps one flattened,
standardised photograph (`in_dim` grey levels) to one number in [0, 1].

* `hidden` is a list of layer widths from the config, for example [256, 64];
  build one Linear + nonlinearity + Dropout block per entry.
* `dropout` is the probability from the config. Where it goes, and whether
  it belongs after the last hidden layer, is your call.
* The output must lie in [0, 1]. `prior` is the mean target of the training
  rows; it is passed to you for a reason you have to find yourself.
* `forward` returns shape (batch,), not (batch, 1).

It trains in seconds on a CPU. That is the point of starting here: every
decision in it can be tested in the time it takes to read this docstring.
Count its parameters against the number of training photographs before you
decide how much dropout it needs.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: list[int], dropout: float = 0.3, prior: float = 0.5):
        super().__init__()
        layers = []
        current_dim = in_dim
        
        # Build one Linear + nonlinearity + Dropout block per entry in hidden
        for h_dim in hidden:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            current_dim = h_dim
            
        self.hidden_layers = nn.Sequential(*layers)
        
        # Final output layer mapping to 1 scalar logit/value
        self.output_layer = nn.Linear(current_dim, 1)
        
        # Initialize the output bias using the logit of the prior training mean.
        # Starting the bias at logit(prior) ensures the untrained network's 
        # initial predictions sit right at the prior mean.
        # Clipped slightly away from 0 and 1 to prevent log(0) or log(inf).
        clipped_prior = max(min(prior, 1.0 - 1e-7), 1e-7)
        initial_bias = torch.logit(torch.tensor(clipped_prior))
        self.output_layer.bias.data.fill_(initial_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.hidden_layers(x)
        
        # Project to final layer and squeeze to shape (batch,) instead of (batch, 1)
        logits = self.output_layer(x).squeeze(-1)
        
        # Bound the output in [0, 1] via Sigmoid
        return torch.sigmoid(logits)
