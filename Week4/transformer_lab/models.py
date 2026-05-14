import math

import torch
from torch import nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    """The fixed sine/cosine position encoding from Attention Is All You Need."""

    def __init__(self, d_model, max_len=512):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, qkv_mode="full"):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if qkv_mode not in {"full", "qk_only"}:
            raise ValueError("qkv_mode must be 'full' or 'qk_only'")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv_mode = qkv_mode
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model) if qkv_mode == "full" else None
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        bsz, seq_len, _ = x.shape
        x = x.view(bsz, seq_len, self.n_heads, self.d_head)
        return x.transpose(1, 2)

    def forward(self, x, context=None, causal=False):
        if context is None:
            context = x
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(context))
        # Ablation 2.2: merge K and V into one projection, so attention both
        # addresses and transports the same representation.
        v = k if self.qkv_mode == "qk_only" else self._split_heads(self.v_proj(context))
        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.d_head)
        if causal:
            seq_len = x.size(1)
            mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device).triu(1)
            scores = scores.masked_fill(mask.view(1, 1, seq_len, seq_len), float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(x.size(0), x.size(1), self.d_model)
        return self.out_proj(out)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, qkv_mode="full"):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout, qkv_mode)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm1(x + self.dropout(self.attn(x)))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, qkv_mode="full"):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout, qkv_mode)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout, qkv_mode)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory):
        x = self.norm1(x + self.dropout(self.self_attn(x, causal=True)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, context=memory)))
        x = self.norm3(x + self.dropout(self.ff(x)))
        return x


class TransformerSeq2SeqModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        seq_len,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=256,
        dropout=0.1,
        qkv_mode="full",
    ):
        super().__init__()
        self.autoregressive = True
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len=seq_len)
        self.encoder = nn.ModuleList(
            [
                TransformerEncoderBlock(d_model, n_heads, d_ff, dropout, qkv_mode)
                for _ in range(n_layers)
            ]
        )
        self.decoder = nn.ModuleList(
            [
                TransformerDecoderBlock(d_model, n_heads, d_ff, dropout, qkv_mode)
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size)
        self.scale = math.sqrt(d_model)

    def embed(self, tokens):
        return self.pos(self.embedding(tokens) * self.scale)

    def forward(self, src_tokens, decoder_tokens):
        memory = self.embed(src_tokens)
        for layer in self.encoder:
            memory = layer(memory)
        x = self.embed(decoder_tokens)
        for layer in self.decoder:
            x = layer(x, memory)
        return self.out(self.norm(x))


class ConvBlock(nn.Module):
    def __init__(self, d_model, kernel_size=5, dropout=0.1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(d_model, d_model, kernel_size, padding=padding)
        self.ff = nn.Conv1d(d_model, d_model, 1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        y = x.transpose(1, 2)
        y = self.conv(y)
        y = F.gelu(y).transpose(1, 2)
        x = self.norm1(x + self.dropout(y))
        y = F.gelu(self.ff(x.transpose(1, 2))).transpose(1, 2)
        return self.norm2(x + self.dropout(y))


class CNNSequenceModel(nn.Module):
    """A positional-encoding + CNN sequence model for ablation 2.4."""

    def __init__(
        self,
        vocab_size,
        seq_len,
        d_model=128,
        n_layers=6,
        kernel_size=5,
        dropout=0.1,
    ):
        super().__init__()
        self.autoregressive = False
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len=seq_len)
        self.layers = nn.ModuleList(
            [ConvBlock(d_model, kernel_size, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size)
        self.scale = math.sqrt(d_model)

    def forward(self, tokens):
        x = self.embedding(tokens) * self.scale
        x = self.pos(x)
        for layer in self.layers:
            x = layer(x)
        return self.out(self.norm(x))


def build_model(args):
    if args.model == "transformer":
        return TransformerSeq2SeqModel(
            vocab_size=args.vocab_size,
            seq_len=args.seq_len,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.layers,
            d_ff=args.d_ff,
            dropout=args.dropout,
            qkv_mode="full",
        )
    if args.model == "qk_only":
        return TransformerSeq2SeqModel(
            vocab_size=args.vocab_size,
            seq_len=args.seq_len,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.layers,
            d_ff=args.d_ff,
            dropout=args.dropout,
            qkv_mode="qk_only",
        )
    if args.model == "cnn":
        return CNNSequenceModel(
            vocab_size=args.vocab_size,
            seq_len=args.seq_len,
            d_model=args.d_model,
            n_layers=args.layers,
            kernel_size=args.kernel_size,
            dropout=args.dropout,
        )
    raise ValueError(f"unknown model: {args.model}")
