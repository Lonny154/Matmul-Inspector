import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniTransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_size: int):
        super().__init__()

        assert hidden_size % num_heads == 0

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.ln1 = nn.LayerNorm(hidden_size)

        # Combined Q, K, V projection
        self.qkv = nn.Linear(
            hidden_size,
            3 * hidden_size,
            bias=False,
        )

        self.attn_out = nn.Linear(
            hidden_size,
            hidden_size,
            bias=False,
        )

        self.ln2 = nn.LayerNorm(hidden_size)

        self.fc1 = nn.Linear(
            hidden_size,
            mlp_size,
            bias=False,
        )

        self.fc2 = nn.Linear(
            mlp_size,
            hidden_size,
            bias=False,
        )

    def forward(self, x):
        batch, seq_len, hidden = x.shape

        residual = x

        with torch.cuda.nvtx.range("layernorm_1"):
            x = self.ln1(x)

        with torch.cuda.nvtx.range("qkv_projection"):
            qkv = self.qkv(x)

        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(
            batch,
            seq_len,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        k = k.view(
            batch,
            seq_len,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        v = v.view(
            batch,
            seq_len,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        with torch.cuda.nvtx.range("attention"):
            attention = F.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=True,
            )

        attention = attention.transpose(1, 2).contiguous()
        attention = attention.view(batch, seq_len, hidden)

        with torch.cuda.nvtx.range("attention_output_projection"):
            x = self.attn_out(attention)

        x = x + residual

        residual = x

        with torch.cuda.nvtx.range("layernorm_2"):
            x = self.ln2(x)

        with torch.cuda.nvtx.range("mlp_fc1"):
            x = self.fc1(x)

        with torch.cuda.nvtx.range("gelu"):
            x = F.gelu(x)

        with torch.cuda.nvtx.range("mlp_fc2"):
            x = self.fc2(x)

        return x + residual

def benchmark(
    model,
    x,
    warmup: int,
    iterations: int,
):
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)

        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        for _ in range(iterations):
            model(x)

        end.record()

        torch.cuda.synchronize()

        elapsed_ms = start.elapsed_time(end)

    mean_ms = elapsed_ms / iterations

    return mean_ms


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=768)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--mlp", type=int, default=3072)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    device = torch.device("cuda")

    model = MiniTransformerBlock(
        hidden_size=args.hidden,
        num_heads=args.heads,
        mlp_size=args.mlp,
    ).to(
        device=device,
        dtype=torch.float16,
    )

    model.eval()

    x = torch.randn(
        args.batch,
        args.seq,
        args.hidden,
        device=device,
        dtype=torch.float16,
    )

    print("Mini Transformer workload")
    print("-------------------------")
    print(f"GPU:          {torch.cuda.get_device_name()}")
    print(f"Batch:        {args.batch}")
    print(f"Sequence:     {args.seq}")
    print(f"Hidden:       {args.hidden}")
    print(f"Heads:        {args.heads}")
    print(f"MLP:          {args.mlp}")
    print(f"Dtype:        {x.dtype}")
    print()

    mean_ms = benchmark(
        model,
        x,
        args.warmup,
        args.iterations,
    )

    blocks_per_second = 1000.0 / mean_ms
    tokens_per_second = (
        args.batch * args.seq * blocks_per_second
    )

    print(f"Mean block latency: {mean_ms:.4f} ms")
    print(f"Blocks/sec:         {blocks_per_second:.2f}")
    print(f"Tokens/sec:         {tokens_per_second:.2f}")


if __name__ == "__main__":
    main()