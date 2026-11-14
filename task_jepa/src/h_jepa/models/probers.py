from torch import nn

from h_jepa.models.module import MLP


def build_prober(*, input_dim, output_dim: int) -> nn.Module:
    return MLP(input_dim=int(input_dim), hidden_dim=512, output_dim=int(output_dim))
