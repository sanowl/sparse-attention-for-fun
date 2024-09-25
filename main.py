import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
from einops import rearrange

class SparseSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_type: str = "longformer",
        attention_window: int = 512,
        block_size: Optional[int] = 64,
        num_random_blocks: Optional[int] = 3,
        use_separate_projections: bool = True,
        dropout: float = 0.1,
        global_tokens: List[int] = None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.attention_type = attention_type.lower()
        
        if self.attention_type == "longformer":
            self.attention_window = attention_window
        elif self.attention_type == "bigbird":
            if block_size is None or num_random_blocks is None:
                raise ValueError("block_size and num_random_blocks must be specified for BigBird attention.")
            self.block_size = block_size
            self.num_random_blocks = num_random_blocks
        else:
            raise ValueError(f"Unsupported attention type: {self.attention_type}")
        
        self.global_tokens = global_tokens or []
        self.use_separate_projections = use_separate_projections

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
        w = self.attention_window

        # Pad the sequence for sliding window attention
        padding = (w // 2, w // 2)
        padded_query = F.pad(query, (0, 0, *padding))
        padded_key = F.pad(key, (0, 0, *padding))
        padded_value = F.pad(value, (0, 0, *padding))

        # Unfold the padded tensors into overlapping chunks
        chunks_count = seq_length
        chunk_query = padded_query.unfold(2, w, 1)
        chunk_key = padded_key.unfold(2, w, 1)
        chunk_value = padded_value.unfold(2, w, 1)

        # Compute attention scores
        attention_scores = torch.einsum('bhcxd,bhcyd->bhcxy', chunk_query, chunk_key) / math.sqrt(head_dim)

        # Apply attention mask if provided
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(-1)
            attention_mask = F.pad(attention_mask, (w//2, w//2)).unfold(2, w, 1)
            attention_scores = attention_scores + (attention_mask * -1e9)

        # Compute attention probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Compute context layer
        context_layer = torch.einsum('bhcxy,bhcyd->bhcxd', attention_probs, chunk_value)

        # Fold the context layer back to the original sequence shape
        context_layer = context_layer.view(batch_size, num_heads, seq_length, head_dim)

        return context_layer, attention_probs
    
    def bigbird_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_heads, seq_length, head_dim = query.size()
        num_blocks = math.ceil(seq_length / self.block_size)
        global_attn_indices = torch.tensor(self.global_tokens, device=query.device)
        if len(global_attn_indices) == 0:
            if seq_length < 2 * self.block_size:
                global_attn_indices = torch.arange(seq_length, device=query.device)
            else:
                global_attn_indices = torch.cat([
                    torch.arange(self.block_size, device=query.device),
                    torch.arange(seq_length - self.block_size, seq_length, device=query.device)
                ])
        random_attn_indices = []
        for i in range(num_blocks):
            block_start = i * self.block_size
            block_end = min(block_start + self.block_size, seq_length)
            current_block_size = block_end - block_start
            if current_block_size == 0:
                continue
            num_rand = min(self.num_random_blocks, current_block_size)
            rand_indices = torch.randperm(current_block_size, device=query.device)[:num_rand] + block_start
            random_attn_indices.append(rand_indices)
        if random_attn_indices:
            random_attn_indices = torch.cat(random_attn_indices)
        else:
            random_attn_indices = torch.tensor([], device=query.device, dtype=torch.long)
        attn_indices = torch.cat([global_attn_indices, random_attn_indices]).unique()
        selected_query = query[:, :, attn_indices]
        attention_scores = torch.matmul(selected_query, key.transpose(-1, -2)) / math.sqrt(head_dim)
        if attention_mask is not None:
            expanded_mask = attention_mask[:, None, attn_indices].unsqueeze(1)
            attention_scores = attention_scores + (expanded_mask * -1e9)
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value)
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
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        output = self.output(context_layer)
        return output, attention_probs

class SparseSelfAttentionBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_type: str = "longformer",
        attention_window: int = 512,
        block_size: Optional[int] = 64,
        num_random_blocks: Optional[int] = 3,
        use_separate_projections: bool = True,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-12,
        pre_ln: bool = True,
        global_tokens: List[int] = None
    ):
        super().__init__()
        self.pre_ln = pre_ln
        self.attention = SparseSelfAttention(
            hidden_size,
            num_attention_heads,
            attention_type,
            attention_window,
            block_size,
            num_random_blocks,
            use_separate_projections,
            dropout,
            global_tokens
        )
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size)
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states
        if self.pre_ln:
            hidden_states = self.layer_norm1(hidden_states)
        attention_output, attention_probs = self.attention(hidden_states, attention_mask)
        hidden_states = residual + self.dropout1(attention_output)
        if not self.pre_ln:
            hidden_states = self.layer_norm1(hidden_states)
        residual = hidden_states
        if self.pre_ln:
            hidden_states = self.layer_norm2(hidden_states)
        feed_forward_output = self.feed_forward(hidden_states)
        hidden_states = residual + self.dropout2(feed_forward_output)
        if not self.pre_ln:
            hidden_states = self.layer_norm2(hidden_states)
        return hidden_states, attention_probs

def test_sparse_attention(attention_type: str, seq_length: int, batch_size: int = 2):
    hidden_size = 768
    num_attention_heads = 12
    block_size = 64 if attention_type == "bigbird" else None
    attention_window = 512 if attention_type == "longformer" else None
    num_random_blocks = 3 if attention_type == "bigbird" else None
    block = SparseSelfAttentionBlock(
        hidden_size,
        num_attention_heads,
        attention_type=attention_type,
        attention_window=attention_window,
        block_size=block_size,
        num_random_blocks=num_random_blocks,
        pre_ln=True
    )
    input_tensor = torch.rand(batch_size, seq_length, hidden_size)
    attention_mask = torch.ones(batch_size, seq_length)
    output, attention_probs = block(input_tensor, attention_mask)
    print(f"{attention_type.capitalize()} output shape:", output.shape)
    print(f"{attention_type.capitalize()} attention probs shape:", attention_probs.shape)
    print(f"Input tensor memory: {input_tensor.element_size() * input_tensor.nelement() / 1024 / 1024:.2f} MB")
    print(f"Output tensor memory: {output.element_size() * output.nelement() / 1024 / 1024:.2f} MB")
    assert output.shape == input_tensor.shape, f"Output shape mismatch: {output.shape} != {input_tensor.shape}"
    return output, attention_probs

if __name__ == "__main__":
    print("Testing Longformer attention:")
    test_sparse_attention("longformer", seq_length=4096)
    print("\nTesting BigBird attention:")
    test_sparse_attention("bigbird", seq_length=4096)
    print("\nMemory-efficient test with longer sequence:")
    test_sparse_attention("longformer", seq_length=16384, batch_size=1)
    print("\nAll tests passed successfully!")