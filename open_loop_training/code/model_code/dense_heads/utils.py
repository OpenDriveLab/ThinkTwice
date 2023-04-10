import torch
import torch.nn as nn
import torch.nn.functional as F


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "none",
) -> torch.Tensor:
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss

def init_weights(m):
    with torch.no_grad():
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif classname.find('LayerNorm') != -1:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif classname.find('BatchNorm') != -1:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif classname.find('GroupNorm') != -1:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif classname.find('Linear') != -1:
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            nn.init.xavier_normal_(m.weight)
        elif classname.find('Embedding') != -1:
            nn.init.trunc_normal_(m.weight, mean=0, std=0.02)





class SpatialGRU(nn.Module):
    """A GRU cell that takes an input tensor [BxTxCxHxW] and an optional previous state and passes a
    convolutional gated recurrent unit over the data"""
    def __init__(self, input_size, hidden_size, act, gru_bias_init=0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru_bias_init = gru_bias_init
        self.act = act
        self.conv_update = nn.Sequential(
            nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1),
            self.act(),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)
        )
        self.conv_reset = nn.Sequential(
            nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1),
            self.act(),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)
        )
        self.conv_state_tilde = nn.Sequential(
            nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1),
            self.act(),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)
        )
        self.conv_decoder = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, bias=True, padding=1),
            self.act(),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)
        )
    def forward(self, x, state=None):
        # Check size
        assert len(x.size()) == 5, 'Input tensor must be BxTxCxHxW.'
        # recurrent layers
        rnn_output = []
        b, timesteps, c, h, w = x.size()
        rnn_state = torch.zeros(b, self.hidden_size, h, w, device=x.device) if state is None else state
        for t in range(timesteps):
            x_t = x[:, t]
            rnn_state = self.gru_cell(x_t, rnn_state)
            rnn_output.append(self.conv_decoder(rnn_state))
        # reshape rnn output to batch tensor
        return torch.stack(rnn_output, dim=1)
    def gru_cell(self, x, state):
        # Compute gates
        x_and_state = torch.cat([x, state], dim=1)
        update_gate = self.conv_update(x_and_state)
        reset_gate = self.conv_reset(x_and_state)
        # Add bias to initialise gate as close to identity function
        update_gate = torch.sigmoid(update_gate + self.gru_bias_init)
        reset_gate = torch.sigmoid(reset_gate + self.gru_bias_init)
        # Compute proposal state, activation is defined in norm_act_config (can be tanh, ReLU etc)
        state_tilde = self.conv_state_tilde(torch.cat([x, (1.0 - reset_gate) * state], dim=1))
        output = (1.0 - update_gate) * state + update_gate * state_tilde
        return output
