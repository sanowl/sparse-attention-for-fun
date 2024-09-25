import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from einops import rearrange, repeat

class SparseSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_type: str = "longformer",
        attention_window: int = 512,
        block_size: int = 64,
        num_random_blocks: int = 3,
        use_separate_projections: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.attention_type = attention_type
        self.attention_window = attention_window
        self.block_size = block_size
        self.num_random_blocks = num_random_blocks

        if use_separate_projections:
            self.query = nn.Linear(hidden_size, hidden_size)
            self.key = nn.Linear(hidden_size, hidden_size)
            self.value = nn.Linear(hidden_size, hidden_size)
        else:
            self.qkv = nn.Linear(hidden_size, 3 * hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, hidden_size)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def longformer_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_heads, seq_length, head_dim = query.size()

        # Compute local attention
        padding_size = (self.attention_window // 2, self.attention_window // 2)
        padded_key = F.pad(key, padding_size)
        padded_value = F.pad(value, padding_size)

        # Use unfold for efficient sliding window computation
        local_key = padded_key.unfold(2, self.attention_window, 1)
        local_value = padded_value.unfold(2, self.attention_window, 1)

        attention_scores = torch.einsum('bhqd,bhkwd->bhqw', query, local_key)
        attention_scores = attention_scores / math.sqrt(head_dim)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.einsum('bhqw,bhwvd->bhqd', attention_probs, local_value)

        return context_layer, attention_probs

    def bigbird_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_heads, seq_length, head_dim = query.size()
        num_blocks = seq_length // self.block_size

        # Global attention (first and last blocks)
        global_attn_indices = torch.cat([
            torch.arange(self.block_size),
            torch.arange(seq_length - self.block_size, seq_length)
        ]).to(query.device)

        # Random attention
        random_attn_indices = torch.randint(
            self.block_size,
            seq_length - self.block_size,
            (self.num_random_blocks * self.block_size,),
            device=query.device
        )

        # Combine global and random attention indices
        attn_indices = torch.cat([global_attn_indices, random_attn_indices]).unique()

        # Compute attention for selected indices
        selected_query = query[:, :, attn_indices]
        attention_scores = torch.einsum('bhid,bhjd->bhij', selected_query, key)
        attention_scores = attention_scores / math.sqrt(head_dim)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask[:, :, attn_indices, :]

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.einsum('bhij,bhjd->bhid', attention_probs, value)

        # Scatter the computed contexts back to their original positions
        output = torch.zeros_like(query)
        output[:, :, attn_indices] = context_layer

        return output, attention_probs

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_length, _ = hidden_states.size()

        if hasattr(self, 'qkv'):
            qkv = self.qkv(hidden_states)
            query, key, value = rearrange(qkv, 'b n (three h d) -> three b h n d', three=3, h=self.num_attention_heads)
        else:
            query = self.transpose_for_scores(self.query(hidden_states))
            key = self.transpose_for_scores(self.key(hidden_states))
            value = self.transpose_for_scores(self.value(hidden_states))

        if self.attention_type == "longformer":
            context_layer, attention_probs = self.longformer_attention(query, key, value, attention_mask)
        elif self.attention_type == "bigbird":
            context_layer, attention_probs = self.bigbird_attention(query, key, value, attention_mask)
        else:
            raise ValueError(f"Unsupported attention type: {self.attention_type}")

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        output = self.output(context_layer)
        output = self.dropout(output)

        return output, attention_probs

class SparseSelfAttentionBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_type: str = "longformer",
        attention_window: int = 512,
        block_size: int = 64,
        num_random_blocks: int = 3,
        use_separate_projections: bool = True,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-12
    ):
        super().__init__()
        self.attention = SparseSelfAttention(
            hidden_size,
            num_attention_heads,
            attention_type,
            attention_window,
            block_size,
            num_random_blocks,
            use_separate_projections,
            dropout
        )
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attention_output, attention_probs = self.attention(self.layer_norm1(hidden_states), attention_mask)
        hidden_states = hidden_states + attention_output
        feed_forward_output = self.feed_forward(self.layer_norm2(hidden_states))
        hidden_states = hidden_states + feed_forward_output
        return hidden_states, attention_probs

# Example usage
if __name__ == "__main__":
    hidden_size = 768
    num_attention_heads = 12
    seq_length = 4096
    batch_size = 2

    # Test Longformer attention
    longformer_block = SparseSelfAttentionBlock(
        hidden_size,
        num_attention_heads,
        attention_type="longformer",
        attention_window=512
    )
    input_tensor = torch.rand(batch_size, seq_length, hidden_size)
    output, _ = longformer_block(input_tensor)
    print("Longformer output shape:", output.shape)

    # Test BigBird attention
    bigbird_block = SparseSelfAttentionBlock(
        hidden_size,
        num_attention_heads,
        attention_type="bigbird",
        block_size=64,
        num_random_blocks=3
    )
    output, _ = bigbird_block(input_tensor)
    print("BigBird output shape:", output.shape)

    # Memory usage
    print(f"Input tensor memory: {input_tensor.element_size() * input_tensor.nelement() / 1024 / 1024:.2f} MB")
    print(f"Output tensor memory: {output.element_size() * output.nelement() / 1024 / 1024:.2f} MB")