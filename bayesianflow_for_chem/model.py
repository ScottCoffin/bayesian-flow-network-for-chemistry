# -*- coding: utf-8 -*-
# Author: Nianze A. Tao (Omozawa Sueno)
"""
Define Bayesian Flow Network for Chemistry (ChemBFN) model.
"""
from pathlib import Path
from typing import List, Tuple, Optional, Union
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import softmax, linear, dropout
from typing_extensions import Self


class Linear(nn.Linear):
    # Modified from https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
    # We made it simpler and compatible with both `loralib` and `TorchScript`.
    def __init__(self, in_features: int, out_features: int, bias: bool = True, **kargs):
        """
        LoRA implemented in a dense layer.

        :param in_features: number of input features
        :param out_features: number of output features
        :param bias: whether to use additional bias
        :param device: device
        :param dtype: PyTorch data type
        :type in_features: int
        :type out_features: int
        :type bias: bool
        :type device: torch.device | str | None
        :type dtype: torch.dtype
        """
        nn.Linear.__init__(self, in_features, out_features, bias, **kargs)
        self.lora_enabled: bool = False
        self.lora_A: Optional[nn.Parameter] = None
        self.lora_B: Optional[nn.Parameter] = None
        self.scaling: Optional[float] = None
        self.lora_dropout: Optional[float] = None
        nn.Linear.reset_parameters(self)

    def enable_lora(
        self, r: int = 8, lora_alpha: int = 1, lora_dropout: float = 0.0
    ) -> None:
        """
        Enable LoRA parameters.

        :param r: rank
        :param lora_alpha: LoRA alpha value
        :param lora_dropout: dropout frequency in LoRA layer
        :type r: int
        :type lora_alpha: float
        :type lora_dropout: float
        :return:
        :rtype: None
        """
        assert r > 0, "Rank should be larger than 0."
        self.lora_A = nn.Parameter(self.weight.new_zeros((r, self.in_features)))
        self.lora_B = nn.Parameter(self.weight.new_zeros((self.out_features, r)))
        self.scaling = lora_alpha / r
        self.lora_dropout = lora_dropout
        self.lora_enabled = True
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)
        self.weight.requires_grad_(False)

    def forward(self, x: Tensor) -> Tensor:
        result = linear(x, self.weight, self.bias)
        if self.lora_enabled and isinstance(self.lora_dropout, float):
            result += (
                dropout(x, self.lora_dropout, self.training)
                @ self.lora_A.transpose(0, 1)
                @ self.lora_B.transpose(0, 1)
            ) * self.scaling
        return result


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale) + shift


class RoPE(nn.Module):
    def __init__(self, channel: int = 512, num_head: int = 8) -> None:
        """
        Rotary position embedding block with XPOS method.

        :param channel: hidden layer features
        :param num_head: number of heads
        :type channel: int
        :type num_head: int
        """
        super().__init__()
        d = channel // num_head
        assert d % 2 == 0
        self.channel = channel
        i = torch.arange(0, d, 2)[None, :] / d
        theta_half = torch.pow(10000, -i)
        zeta_half = (i + 0.4) / 1.4
        theta, zeta = torch.zeros((1, d)), torch.zeros((1, d))
        theta[:, 0::2] = theta_half
        theta[:, 1::2] = theta_half
        zeta[:, 0::2] = zeta_half
        zeta[:, 1::2] = zeta_half
        self.register_buffer("theta", theta)
        self.register_buffer("zeta", zeta)

    def forward(self, size: int) -> Tuple[Tensor, Tensor, Tensor]:
        """
        :param size: maximum length of sequence in the batch
        :type size: int
        :return: cos part of position encoding;  shape: (1, 1, n_t, n_h) \n
                 sin part of position encoding;  shape: (1, 1, n_t, n_h) \n
                 scaling coefficients;           shape: (1, 1, n_t, n_h)
        :rtype: tuple
        """
        pos = torch.arange(size, device=self.theta.device)[:, None]
        cos, sin = torch.cos(pos * self.theta), torch.sin(pos * self.theta)
        zeta = torch.pow(self.zeta, pos / self.channel)
        return cos[None, None, ...], sin[None, None, ...], zeta[None, None, ...]


class Attention(nn.Module):
    def __init__(self, channel: int = 512, num_head: int = 8) -> None:
        """
        Multi-head self-attention block.

        :param channel: hidden layer features
        :param num_head: number of heads
        :type channel: int
        :type num_head: int
        """
        super().__init__()
        assert channel % num_head == 0
        self.d = channel // num_head  # head dimension
        self.nh = num_head  # number of heads
        self.tp = (2 * self.d) ** 0.5  # attention temperature
        self.qkv = Linear(channel, channel * 3)

    @staticmethod
    def _rotate(
        q: Tensor, k: Tensor, pe: Tuple[Tensor, Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        q_rotate, k_rotate = torch.zeros_like(q), torch.zeros_like(k)
        q_rotate[..., 0::2] = -q[..., 1::2]
        q_rotate[..., 1::2] = q[..., 0::2]
        q = (q * pe[0] + q_rotate * pe[1]) * pe[2]
        k_rotate[..., 0::2] = -k[..., 1::2]
        k_rotate[..., 1::2] = k[..., 0::2]
        k = (k * pe[0] + k_rotate * pe[1]) / pe[2]
        return q, k

    def forward(
        self, x: Tensor, pe: Tuple[Tensor, Tensor, Tensor], mask: Optional[Tensor]
    ) -> Tensor:
        """
        :param x: output tensor;       shape: (n_b, n_t, n_f)
        :param pe: position encoding;  shape: (1, 1, n_t, n_h) * 3
        :param mask: attention mask;   shape: (1, n_b, n_t, n_t)
        :type x: torch.Tensor
        :type pe: tuple
        :type mask: torch.Tensor | None
        :return: attentioned output;   shape: (n_b, n_t, n_f)
        :rtype: torch.Tensor
        """
        n_b, n_a, _ = shape = x.shape
        split = (n_b, n_a, self.nh, self.d)
        q, k, v = self.qkv(x).chunk(3, -1)
        q = q.view(split).permute(2, 0, 1, 3).contiguous()
        k = k.view(split).permute(2, 0, 1, 3).contiguous()
        v = v.view(split).permute(2, 0, 1, 3).contiguous()
        q, k = self._rotate(q, k, pe)  # position embedding
        """
        # Original code. Maybe using `nn.functional.scaled_dot_product_attention(...)` is better.

        k_t = k.transpose(-2, -1)
        if mask is not None:
            alpha = softmax((q @ k_t / self.tp).masked_fill_(mask, -torch.inf), -1)
        else:
            alpha = softmax(q @ k_t / self.tp, -1)
        atten_out = (alpha @ v).permute(1, 2, 0, 3).contiguous().view(shape)
        """
        atten_out = nn.functional.scaled_dot_product_attention(
            q, k, v, mask, 0.0, False, scale=1 / self.tp
        )
        atten_out = atten_out.permute(1, 2, 0, 3).contiguous().view(shape)
        return atten_out

    def enable_lora(
        self, r: int = 4, lora_alpha: int = 1, lora_dropout: float = 0.0
    ) -> None:
        """
        Enable LoRA parameters.

        :param r: rank
        :param lora_alpha: LoRA alpha value
        :param lora_dropout: dropout frequency in LoRA layer
        :type r: int
        :type lora_alpha: float
        :type lora_dropout: float
        :return:
        :rtype: None
        """
        self.qkv.enable_lora(r, lora_alpha, lora_dropout)


class TransformerLayer(nn.Module):
    def __init__(
        self, channel: int = 512, num_head: int = 8, dropout: float = 0.01
    ) -> None:
        """
        Transfomer layer block.

        :param channel: hidden layer features
        :param num_head: number of attention heads
        :param dropout: dropout frequency
        :type channel: int
        :type num_head: int
        :type dropout: float
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(channel, 1e-6, False)
        self.attention = Attention(channel, num_head)
        self.norm2 = nn.LayerNorm(channel, 1e-6, False)
        self.ffn = nn.Sequential(
            nn.Linear(channel, channel * 4),
            nn.SELU(),
            nn.Linear(channel * 4, channel),
            nn.Dropout(dropout),
        )
        self.adaln_modulation = nn.Sequential(nn.SELU(), Linear(channel, 6 * channel))
        # zero-out adaLN layer
        nn.init.constant_(self.adaln_modulation[1].weight, 0)
        nn.init.constant_(self.adaln_modulation[1].bias, 0)

    def forward(
        self,
        x: Tensor,
        pe: Tuple[Tensor, Tensor, Tensor],
        c: Tensor,
        mask: Optional[Tensor],
    ) -> Tensor:
        """
        :param x: input tensor;        shape: (n_b, n_t, n_f)
        :param pe: position encoding;  shape: (1, 1, n_t, n_h) * 3
        :param c: conditioning;        shape: (n_b, 1, n_f)
        :param mask: attention mask;   shape: (1, n_b, n_t, n_t)
        :type x: torch.Tensor
        :type pe: tuple
        :type c: torch.Tensor
        :type mask: torch.Tensor | None
        :return: output tensor;        shape: (n_b, n_t, n_f)
        :rtype: torch.Tensor
        """
        c = self.adaln_modulation(c)
        shift, scale, gate, shift_ffn, scale_ffn, gate_ffn = c.chunk(6, -1)
        x = x + gate * self.attention(modulate(self.norm1(x), shift, scale), pe, mask)
        x = x + gate_ffn * self.ffn(modulate(self.norm2(x), shift_ffn, scale_ffn))
        return x

    def enable_lora(
        self, r: int = 4, lora_alpha: int = 1, lora_dropout: float = 0.0
    ) -> None:
        """
        Enable LoRA parameters.

        :param r: rank
        :param lora_alpha: LoRA alpha value
        :param lora_dropout: dropout frequency in LoRA layer
        :type r: int
        :type lora_alpha: float
        :type lora_dropout: float
        :return:
        :rtype: None
        """
        self.attention.enable_lora(r, lora_alpha, lora_dropout)
        self.adaln_modulation[1].enable_lora(r, lora_alpha, lora_dropout)


class FinalLayer(nn.Module):
    def __init__(self, num_vocab: int, channel: int = 512) -> None:
        """
        The final layer of model.

        :param num_vocab: number of vocabulary
        :param channel: hidden layer features
        :type num_vocab: int
        :type channel: int
        """
        super().__init__()
        self.norm_final = nn.LayerNorm(channel, 1e-6, False)
        self.linear = Linear(channel, num_vocab)
        self.adaln_modulation = nn.Sequential(nn.SELU(), Linear(channel, 2 * channel))
        # zero-out this layer
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)
        nn.init.constant_(self.adaln_modulation[-1].weight, 0)
        nn.init.constant_(self.adaln_modulation[-1].bias, 0)

    def forward(self, x: Tensor, c: Tensor, return_logits: bool = True) -> Tensor:
        """
        :param x: input tensor;                 shape: (n_b, n_t, n_f)
        :param c: conditioning;                 shape: (n_b, 1, n_f)
        :param return_logits: whether to return unnormalised output logits
        :type x: torch.Tensor
        :type c: torch.Tensor
        :type return_logits: bool
        :return: output logits (unnormalised);  shape: (n_b, n_t, n_vocab)
                 or token embeddings;           shape: (n_b, n_t, n_f)
        :rtype: torch.Tensor
        """
        shift, scale = self.adaln_modulation(c).chunk(2, -1)
        x = modulate(self.norm_final(x), shift, scale)
        if return_logits:
            return self.linear(x)
        return x

    def enable_lora(
        self, r: int = 4, lora_alpha: int = 1, lora_dropout: float = 0.0
    ) -> None:
        """
        Enable LoRA parameters.

        :param r: rank
        :param lora_alpha: LoRA alpha value
        :param lora_dropout: dropout frequency in LoRA layer
        :type r: int
        :type lora_alpha: float
        :type lora_dropout: float
        :return:
        :rtype: None
        """
        self.linear.enable_lora(r, lora_alpha, lora_dropout)
        self.adaln_modulation[1].enable_lora(r, lora_alpha, lora_dropout)


class ChemBFN(nn.Module):
    def __init__(
        self,
        num_vocab: int,
        channel: int = 512,
        num_layer: int = 12,
        num_head: int = 8,
        dropout: float = 0.01,
    ) -> None:
        r"""
        Bayesian Flow Network for Chemistry model representation.

        Enable semi-autoregressive sampling by setting
        `ChemBFN(...).semi_autoregressive = True`.

        :param num_vocab: number of vocabulary
        :param channel: hidden layer features
        :param num_layer: number of transformer layers
        :param num_head: number of heads
        :param dropout: dropout frequency
        :type num_vocab: int
        :type channel: int
        :type num_layer: int
        :type num_head: int
        :type dropout: float
        """
        super().__init__()
        self.K = num_vocab
        self.lora_enabled: bool = False
        self.semi_autoregressive: bool = False
        self.embedding = Linear(num_vocab, channel)
        self.time_embed = nn.Sequential(
            nn.Linear(1, channel // 2), nn.SELU(), nn.Linear(channel // 2, channel)
        )
        self.position = RoPE(channel, num_head)
        self.encoder_layers = nn.ModuleList(
            [TransformerLayer(channel, num_head, dropout) for _ in range(num_layer)]
        )
        self.final_layer = FinalLayer(num_vocab, channel)
        self.register_buffer("beta", torch.scalar_tensor(20.4054 / self.K))
        self.hparam = dict(
            num_vocab=num_vocab,
            channel=channel,
            num_layer=num_layer,
            num_head=num_head,
            dropout=dropout,
        )
        self.lora_param = {}

    def enable_lora(
        self, r: int = 4, lora_alpha: int = 1, lora_dropout: float = 0.0
    ) -> None:
        """
        Enable LoRA parameters.

        :param r: rank
        :param lora_alpha: LoRA alpha value
        :param lora_dropout: dropout frequency in LoRA layer
        :type r: int
        :type lora_alpha: float
        :type lora_dropout: float
        :return:
        :rtype: None
        """
        self.lora_enabled = True
        self.lora_param = dict(r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        self.embedding.enable_lora(r, lora_alpha, lora_dropout)
        for layer in self.encoder_layers:
            layer.enable_lora(r, lora_alpha, lora_dropout)
        self.final_layer.enable_lora(r, lora_alpha, lora_dropout)

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        mask: Optional[Tensor] = None,
        y: Optional[Tensor] = None,
    ) -> Tensor:
        """
        :param x: input probabilities;                       shape: (n_b, n_t, n_vocab)
        :param t: time;                                      shape: (n_b, 1, 1)
        :param mask: input mask;                             shape: (n_b, n_t, 1)
        :param y: conditioning vector;                       shape: (n_b, 1, n_f)
        :type x: torch.Tensor
        :type t: torch.Tensor
        :type mask: torch.Tensor | None
        :type y: torch.Tensor | None
        :return: probability distribution (before softmax);  shape: (n_b, n_t, n_vocab)
                 or token embeddings;                        shape: (n_b, n_t, n_f)
        :rtype: torch.Tensor
        """
        n_b, n_t, _ = x.shape
        c = self.time_embed(t)
        if y is not None:
            c += y
        pe = self.position(x.shape[1])
        x = self.embedding(x)
        attn_mask: Optional[Tensor] = None
        if self.semi_autoregressive:
            attn_mask = torch.tril(
                torch.ones((1, n_b, n_t, n_t), device=self.beta.device), diagonal=0
            )
        else:
            if mask is not None:
                """
                # Original Code.

                attn_mask = mask.transpose(-2, -1).repeat(1, x.shape[1], 1)[None, ...] == 0
                """
                attn_mask = mask.transpose(-2, -1).repeat(1, n_t, 1)[None, ...] != 0
        for layer in self.encoder_layers:
            x = layer(x, pe, c, attn_mask)
        return self.final_layer(x, c, mask is None)

    def calc_beta(self, t: Tensor) -> Tensor:
        r"""
        Calculate beta(t) value.

        .. math::
        ```
        \begin{equation}
            \beta(t) = %
            -\frac{4\ln{(1 - t + te^{-\frac{K}{4}\beta(1)})}}{K}
        \end{equation}
        ```

        :param t: continuous time in [0, 1];  shape: (n_b, 1, 1)
        :type t: torch.Tensor
        :return: beta(t);                     shape: (n_b, 1, 1)
        :rtype: torch.Tensor
        """
        return -4 * (1 - t + t * (-self.K * self.beta / 4).exp()).log() / self.K

    def calc_discrete_alpha(self, t1: Tensor, t2: Tensor) -> Tensor:
        r"""
        Calculate alpha(i) value.

        .. math:: $\alpha(i) = \bate(t_{i}) - \beta(t_{i - 1})$

        :param t1: discrete time (i - 1) / n;  shape: (n_b, 1, 1)
        :param t2: discrete time i / n;        shape: (n_b, 1, 1)
        :type t1: torch.Tensor
        :type t2: torch.Tensor
        :return: alpha(i);                     shape: (n_b, 1, 1)
        :rtype: torch.Tensor
        """
        # assert t2 > t1
        return self.calc_beta(t2) - self.calc_beta(t1)

    def calc_cts_alpha(self, t: Tensor) -> Tensor:
        r"""
        Calculate alpha(t) / 2 value.

        .. math::
        ```
        \begin{equation}
            \alpha(t) = %
            \frac{d\beta(t)}{dt} = %
            \frac{4}{K}%
            \frac{1 - e^{-\frac{K}{4}\beta(1)}}%
            {1 - t + te^{-\frac{K}{4}\beta(1)}}
        \end{equation}
        ```

        :param t: continuous time in [0, 1];  shape: (n_b, 1, 1)
        :type t: torch.Tensor
        :return: alpha(t);                    shape: (n_b, 1, 1)
        :rtype: torch.Tensor
        """
        a = 1 - (-self.K * self.beta / 4).exp()
        b = 1 - t + t * (-self.K * self.beta / 4).exp()
        return 2 * a / b / self.K

    def discrete_output_distribution(
        self, theta: Tensor, t: Tensor, y: Optional[Tensor], w: Optional[float]
    ) -> Tensor:
        """
        :param theta: input distribution;     shape: (n_b, n_t, n_vocab)
        :param t: continuous time in [0, 1];  shape: (n_b, 1, 1)
        :param y: conditioning vector;        shape: (n_b, 1, n_f)
        :param w: guidance strength controlling the conditional generation
        :type theta: torch.Tensor
        :type t: torch.Tensor
        :type y: torch.Tensor | None
        :type w: float | None
        :return: output distribution;         shape: (n_b, n_t, n_vocab)
        :rtype: torch.Tensor
        """
        theta = 2 * theta - 1  # rescale to [-1, 1]
        if w is None:
            return softmax(self.forward(theta, t, None, y), -1)
        elif y is None:
            return softmax(self.forward(theta, t, None, None), -1)
        else:
            p_cond = self.forward(theta, t, None, y)
            p_uncond = self.forward(theta, t, None, None)
            return softmax((1 + w) * p_cond - w * p_uncond, -1)

    def cts_loss(
        self,
        x: Tensor,
        t: Tensor,
        y: Optional[Tensor],
        mask: Optional[Tensor] = None,
        return_output_dist: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Compute continuous-time loss.

        :param x: target data;                shape: (n_b, n_t)
        :param t: continuous time in [0, 1);  shape: (n_b, 1, 1)
        :param y: conditioning vector;        shape: (n_b, 1, n_f)
        :param mask: in-text mask;            shape: (n_b, n_t)
        :param return_output_dist: whether to return the output distribution
        :type x: torch.Tensor
        :type t: torch.Tensor
        :type y: torch.Tensor | None
        :type mask: torch.Tensor | None
        :type return_output_dist: bool
        :returns: continuous-time loss;       shape: () \n
                  output distribution;        shape: (n_b, n_t, n_vocab) or `None`
        :rtype: tuple
        """
        beta = self.calc_beta(t)  # shape: (n_b, 1, 1)
        e_x = nn.functional.one_hot(x, self.K).float()
        mu = beta * (self.K * e_x - 1)
        sigma = (beta * self.K).sqrt()
        theta = softmax(mu + sigma * torch.randn_like(mu), -1)
        if mask is not None:
            mask = mask[..., None]
            theta = e_x * mask + (1 - mask) * theta
        e_hat = self.discrete_output_distribution(theta, t, y, None)
        cts_loss = self.K * (e_x - e_hat).pow(2) * self.calc_cts_alpha(t)
        if return_output_dist:
            return cts_loss.mean(), e_hat
        return cts_loss.mean(), None

    @torch.inference_mode()
    def reconstruction_loss(self, x: Tensor, t: Tensor, y: Optional[Tensor]) -> Tensor:
        """
        Compute reconstruction loss.

        :param x: target data;                shape: (n_b, n_t)
        :param t: continuous time in [0, 1];  shape: (n_b, 1, 1)
        :param y: conditioning vector;        shape: (n_b, 1, n_f)
        :type x: torch.Tensor
        :type t: torch.Tensor
        :type y: torch.Tensor | None
        :return: reconstruction loss;         shape: ()
        :rtype: torch.Tensor
        """
        beta = self.calc_beta(t)
        mu = beta * (self.K * nn.functional.one_hot(x, self.K).float() - 1)
        sigma = (beta * self.K).sqrt()
        theta = softmax(mu + sigma * torch.randn_like(mu), -1)
        logits = self.forward(2 * theta - 1, t, None, y)
        # compute negative log probability
        x, logits = torch.broadcast_tensors(x[..., None], logits)
        return (-logits.gather(-1, x[..., :1]).squeeze(-1)).mean()

    @torch.jit.export
    def sample(
        self,
        batch_size: int,
        sequence_size: int,
        y: Optional[Tensor],
        sample_step: int = 100,
        guidance_strength: float = 4.0,
        token_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Sample from a piror distribution.

        :param batch_size: batch size
        :param sequence_size: max sequence length
        :param y: conditioning vector;      shape: (n_b, 1, n_f)
        :param sample_step: number of sampling steps
        :param guidance_strength: strength of conditional generation. It is not used if y is null.
        :param token_mask: token mask;      shape: (1, 1, n_vocab)
        :type batch_size: int
        :type sequence_size: int
        :type y: torch.Tensor | None
        :type sample_step: int
        :type guidance_strength: float
        :type token_mask: torch.Tensor | None
        :return: sampled token indices;     shape: (n_b, n_t) \n
                 entropy of the tokens;     shape: (n_b)
        :rtype: tuple
        """
        theta = (
            torch.ones((batch_size, sequence_size, self.K), device=self.beta.device)
            / self.K
        )
        if y is not None:
            assert y.dim() == 3  # this doesn't work if the model is frezen in JIT.
            if y.shape[0] == 1:
                y = y.repeat(batch_size, 1, 1)
        for i in torch.linspace(1, sample_step, sample_step, device=self.beta.device):
            t = (i - 1).view(1, 1, 1).repeat(batch_size, 1, 1) / sample_step
            p = self.discrete_output_distribution(theta, t, y, guidance_strength)
            if token_mask is not None:
                p = p.masked_fill_(token_mask, 0.0)
            alpha = self.calc_discrete_alpha(t, t + 1 / sample_step)
            e_k = nn.functional.one_hot(torch.argmax(p, -1), self.K).float()
            mu = alpha * (self.K * e_k - 1)
            sigma = (alpha * self.K).sqrt()
            theta = (mu + sigma * torch.randn_like(mu)).exp() * theta
            theta = theta / theta.sum(-1, True)
        t_final = torch.ones((batch_size, 1, 1), device=self.beta.device)
        p = self.discrete_output_distribution(theta, t_final, y, guidance_strength)
        entropy = -(p * p.log()).sum(-1).mean(-1)
        if token_mask is not None:
            p = p.masked_fill_(token_mask, 0.0)
        return torch.argmax(p, -1), entropy

    @torch.jit.export
    def ode_sample(
        self,
        batch_size: int,
        sequence_size: int,
        y: Optional[Tensor],
        sample_step: int = 100,
        guidance_strength: float = 4.0,
        token_mask: Optional[Tensor] = None,
        temperature: float = 0.5,
    ) -> Tuple[Tensor, Tensor]:
        """
        ODE-based sampling.

        :param batch_size: batch size
        :param sequence_size: max sequence length
        :param y: conditioning vector;      shape: (n_b, 1, n_f)
        :param sample_step: number of sampling steps
        :param guidance_strength: strength of conditional generation. It is not used if y is null.
        :param token_mask: token mask;      shape: (1, 1, n_vocab)
        :param temperature: sampling temperature
        :type batch_size: int
        :type sequence_size: int
        :type y: torch.Tensor | None
        :type sample_step: int
        :type guidance_strength: float
        :type token_mask: torch.Tensor | None
        :type temperature: float
        :return: sampled token indices;     shape: (n_b, n_t) \n
                 entropy of the tokens;     shape: (n_b)
        :rtype: tuple
        """
        z = torch.zeros((batch_size, sequence_size, self.K), device=self.beta.device)
        if y is not None:
            assert y.dim() == 3  # this doesn't work if the model is frezen in JIT.
            if y.shape[0] == 1:
                y = y.repeat(batch_size, 1, 1)
        for i in torch.linspace(1, sample_step, sample_step, device=self.beta.device):
            t = (i - 1).view(1, 1, 1).repeat(batch_size, 1, 1) / sample_step
            theta = torch.softmax(z, -1)
            beta = self.calc_beta(t + 1 / sample_step)
            p = self.discrete_output_distribution(theta, t, y, guidance_strength)
            if token_mask is not None:
                p = p.masked_fill_(token_mask, 0.0)
            u = torch.randn_like(z)
            z = (self.K * p - 1) * beta + (self.K * beta * temperature).sqrt() * u
        t_final = torch.ones((batch_size, 1, 1), device=self.beta.device)
        theta = torch.softmax(z, -1)
        p = self.discrete_output_distribution(theta, t_final, y, guidance_strength)
        entropy = -(p * p.log()).sum(-1).mean(-1)
        if token_mask is not None:
            p = p.masked_fill_(token_mask, 0.0)
        return torch.argmax(p, -1), entropy

    @torch.jit.export
    def inpaint(
        self,
        x: Tensor,
        y: Optional[Tensor] = None,
        sample_step: int = 100,
        guidance_strength: float = 4.0,
        token_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Molecule inpaint functionality.

        :param x: categorical indices of scaffold;  shape: (n_b, n_t)
        :param y: conditioning vector;              shape: (n_b, 1, n_f)
        :param sample_step: number of sampling steps
        :param guidance_strength: strength of conditional generation. It is not used if y is null.
        :param token_mask: token mask;              shape: (1, 1, n_vocab)
        :type x: torch.Tensor
        :type y: torch.Tensor | None
        :type sample_step: int
        :type guidance_strength: float
        :type token_mask: torch.Tensor | None
        :return: sampled token indices;             shape: (n_b, n_t) \n
                 entropy of the tokens;             shape: (n_b)
        :rtype: tuple
        """
        n_b, n_t = x.shape
        mask = (x != 0).float()[..., None]
        theta = torch.ones((n_b, n_t, self.K), device=x.device) / self.K
        x_onehot = nn.functional.one_hot(x, self.K) * mask
        theta = x_onehot + (1 - mask) * theta
        if y is not None:
            assert y.dim() == 3  # this doesn't work if the model is frezen in JIT.
            if y.shape[0] == 1:
                y = y.repeat(n_b, 1, 1)
        for i in torch.linspace(1, sample_step, sample_step, device=x.device):
            t = (i - 1).view(1, 1, 1).repeat(n_b, 1, 1) / sample_step
            p = self.discrete_output_distribution(theta, t, y, guidance_strength)
            if token_mask is not None:
                p = p.masked_fill_(token_mask, 0.0)
            alpha = self.calc_discrete_alpha(t, t + 1 / sample_step)
            e_k = nn.functional.one_hot(torch.argmax(p, -1), self.K).float()
            mu = alpha * (self.K * e_k - 1)
            sigma = (alpha * self.K).sqrt()
            theta = (mu + sigma * torch.randn_like(mu)).exp() * theta
            theta = theta / theta.sum(-1, True)
            theta = x_onehot + (1 - mask) * theta
        t_final = torch.ones((n_b, 1, 1), device=x.device)
        p = self.discrete_output_distribution(theta, t_final, y, guidance_strength)
        entropy = -(p * p.log()).sum(-1).mean(-1)
        if token_mask is not None:
            p = p.masked_fill_(token_mask, 0.0)
        return torch.argmax(p, -1), entropy

    @torch.jit.export
    def ode_inpaint(
        self,
        x: Tensor,
        y: Optional[Tensor] = None,
        sample_step: int = 100,
        guidance_strength: float = 4.0,
        token_mask: Optional[Tensor] = None,
        temperature: float = 0.5,
    ) -> Tuple[Tensor, Tensor]:
        """
        ODE inpainting.

        :param x: categorical indices of scaffold;  shape: (n_b, n_t)
        :param y: conditioning vector;              shape: (n_b, 1, n_f)
        :param sample_step: number of sampling steps
        :param guidance_strength: strength of conditional generation. It is not used if y is null.
        :param token_mask: token mask;              shape: (1, 1, n_vocab)
        :param temperature: sampling temperature
        :type x: torch.Tensor
        :type y: torch.Tensor | None
        :type sample_step: int
        :type guidance_strength: float
        :type token_mask: torch.Tensor | None
        :type temperature: float
        :return: sampled token indices;             shape: (n_b, n_t) \n
                 entropy of the tokens;             shape: (n_b)
        :rtype: tuple
        """
        n_b, n_t = x.shape
        mask = (x != 0).float()[..., None]
        x_onehot = nn.functional.one_hot(x, self.K) * mask
        z = torch.zeros((n_b, n_t, self.K), device=self.beta.device)
        if y is not None:
            assert y.dim() == 3  # this doesn't work if the model is frezen in JIT.
            if y.shape[0] == 1:
                y = y.repeat(n_b, 1, 1)
        for i in torch.linspace(1, sample_step, sample_step, device=self.beta.device):
            t = (i - 1).view(1, 1, 1).repeat(n_b, 1, 1) / sample_step
            theta = torch.softmax(z, -1)
            theta = x_onehot + (1 - mask) * theta
            beta = self.calc_beta(t + 1 / sample_step)
            p = self.discrete_output_distribution(theta, t, y, guidance_strength)
            if token_mask is not None:
                p = p.masked_fill_(token_mask, 0.0)
            u = torch.randn_like(z)
            z = (self.K * p - 1) * beta + (self.K * beta * temperature).sqrt() * u
        t_final = torch.ones((n_b, 1, 1), device=self.beta.device)
        theta = torch.softmax(z, -1)
        theta = x_onehot + (1 - mask) * theta
        p = self.discrete_output_distribution(theta, t_final, y, guidance_strength)
        entropy = -(p * p.log()).sum(-1).mean(-1)
        if token_mask is not None:
            p = p.masked_fill_(token_mask, 0.0)
        return torch.argmax(p, -1), entropy

    def inference(self, x: Tensor, mlp: nn.Module) -> Tensor:
        """
        Predict from SMILES tokens.

        :param x: input tokens;  shape: (n_b, n_t)
        :param mlp: MLP module
        :type x: torch.Tensor
        :type mlp: torch.nn.Module
        :return: output values;  shape: (n_b, n_task)
        :rtype: torch.Tensor
        """
        t = torch.ones((x.shape[0], 1, 1), device=x.device)
        mask = (x != 0).float()[..., None]
        theta = 2 * torch.nn.functional.one_hot(x, self.K).float() - 1
        z = self.forward(theta, t, mask, None)
        if self.semi_autoregressive:
            return mlp.forward(z[x == 2].view(z.shape[0], -1))
        return mlp.forward(z[::, 0])

    @classmethod
    def from_checkpoint(
        cls, ckpt: Union[str, Path], ckpt_lora: Union[str, Path, None] = None
    ) -> Self:
        """
        Load model weight from a checkpoint.

        :param ckpt: checkpoint file
        :param ckpt_lora: LoRA checkpoint file which is optional
        :type ckpt: str | pathlib.Path
        :type ckpt_lora: str | pathlib.Path | None
        :return: Bayesian Flow Network for Chemistry model
        :rtype: bayesianflow_for_chem.model.ChemBNF
        """
        with open(ckpt, "rb") as f:
            state = torch.load(f, "cpu", weights_only=True)
        nn, hparam = state["nn"], state["hparam"]
        model = cls(
            hparam["num_vocab"],
            hparam["channel"],
            hparam["num_layer"],
            hparam["num_head"],
            hparam["dropout"],
        )
        model.load_state_dict(nn, False)
        if ckpt_lora:
            with open(ckpt_lora, "rb") as g:
                lora_state = torch.load(g, "cpu", weights_only=True)
            lora_nn, lora_param = lora_state["lora_nn"], lora_state["lora_param"]
            model.enable_lora(**lora_param)
            model.load_state_dict(lora_nn, False)
        return model


class MLP(nn.Module):
    def __init__(
        self, size: List[int], class_input: bool = False, dropout: float = 0.0
    ) -> None:
        """
        MLP module.
        e.g.

        ```python
        mlp = MLP(size=[512, 256, 1])
        mlp = MLP(size=[10, 256, 512], True)  # embedding 10 classes
        ```

        :param size: hidden feature sizes
        :param class_input: whether the input is class indices
        :param dropout: dropout frequency
        :type size: list
        :type class_input: bool
        :type dropout: float
        """
        super().__init__()
        assert len(size) >= 2
        self.class_input = class_input
        self.dropout = nn.Dropout(dropout if not class_input else 0.0)
        self.layers = nn.ModuleList(
            [nn.Linear(i, size[key + 1]) for key, i in enumerate(size[:-2])]
        )
        if class_input:
            self.layers[0] = nn.Embedding(size[0], size[1])
        self.layers.append(nn.Linear(size[-2], size[-1]))
        self.hparam = dict(size=size, class_input=class_input, dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: input tensor;  shape: (n_b, n_input)
        :return: output tensor;  shape: (n_b, n_output) if not class_input;
                                        (n_b, 1, n_output) if class_input
        :type x: torch.Tensor
        :rtype: torch.Tensor
        """
        x = self.dropout(x)
        if self.class_input:
            x = x.to(dtype=torch.long)
        for layer in self.layers[:-1]:
            x = torch.selu(layer.forward(x))
        return self.layers[-1](x)

    @classmethod
    def from_checkpoint(cls, ckpt: Union[str, Path], strict: bool = True) -> Self:
        """
        Load model weight from a checkpoint.

        :param ckpt: checkpoint file
        :param strict: whether to strictly match `state_dict`
        :type ckpt: str | pathlib.Path
        :type strict: bool
        :return: MLP
        :rtype: bayesianflow_for_chem.model.MLP
        """
        with open(ckpt, "rb") as f:
            state = torch.load(f, "cpu", weights_only=True)
        nn, hparam = state["nn"], state["hparam"]
        model = cls(hparam["size"], hparam["class_input"], hparam["dropout"])
        model.load_state_dict(nn, strict)
        return model


if __name__ == "__main__":
    ...
