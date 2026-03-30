import torch
import torch.nn as nn
import torch.nn.functional as F
# PRT Attention — CORE ARCHITECTURE
# Key idea:
#   - Token i (patch) can only attend to token j if j is within spatial range R
#   - CLS token (index 0) attends to ALL tokens — global classification signal
#   - ALL tokens can attend back to the CLS token

def build_prt_mask(num_patches_side: int, range_r: int) -> torch.Tensor:
    """
    Build the PRT attention mask.
    Returns mask of shape (N+1, N+1) where N = num_patches_side^2
    Mask value = 0.0 means ALLOWED, -inf means BLOCKED
    
    Rules:
      - CLS (row 0): can attend to everything → all 0
      - Patch i (row i+1): can attend to CLS (col 0) → 0
      - Patch i (row i+1): can attend to patch j (col j+1) if
        chebyshev_distance(i, j) <= range_r → 0, else -inf
    """
    N = num_patches_side * num_patches_side
    total = N + 1  # +1 for CLS
    mask = torch.full((total, total), float('-inf'))
    
    # CLS row — attend to everything
    mask[0, :] = 0.0
    # All patches attend to CLS column
    mask[:, 0] = 0.0
    
    # Patch-to-patch: within range R (Chebyshev distance)
    for i in range(N):
        ri, ci = divmod(i, num_patches_side)
        for j in range(N):
            rj, cj = divmod(j, num_patches_side)
            if max(abs(ri - rj), abs(ci - cj)) <= range_r:
                mask[i + 1, j + 1] = 0.0
    
    return mask  # (N+1, N+1)


class PRTAttention(nn.Module):
    """Patch Range Transformer Attention with CLS global bridge."""
    
    def __init__(self, embed_dim: int, num_heads: int,
                 num_patches_side: int, range_r: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = self.head_dim ** -0.5
        
        self.qkv     = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj    = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Pre-compute and register the mask (not a parameter)
        mask = build_prt_mask(num_patches_side, range_r)
        self.register_buffer('attn_mask', mask)  # (N+1, N+1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape  # N = num_patches + 1 (CLS)
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)            # each: (B, heads, N, head_dim)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)
        
        # Apply PRT mask: broadcast over batch and heads
        attn = attn + self.attn_mask.unsqueeze(0).unsqueeze(0)  # (1,1,N,N)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


# Cell 8: PRT Transformer Block and full model

class PRTBlock(nn.Module):
    """Single PRT Transformer encoder block."""
    
    def __init__(self, embed_dim, num_heads, num_patches_side, range_r,
                 mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = PRTAttention(embed_dim, num_heads,
                                  num_patches_side, range_r, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchRangeTransformer(nn.Module):
    """
    PRT: Patch Range Transformer for Image Classification
    
    Architecture:
    1. Patch embedding (like ViT)
    2. CLS token prepended
    3. Positional embeddings added
    4. N x PRTBlock (range-limited attention + global CLS)
    5. Classification head on CLS token
    """
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 num_classes=4, embed_dim=256, num_heads=8,
                 num_layers=6, range_r=2, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        
        assert img_size % patch_size == 0
        self.num_patches_side = img_size // patch_size
        self.num_patches = self.num_patches_side ** 2
        self.patch_dim = in_channels * patch_size * patch_size
        self.embed_dim = embed_dim
        
        # Patch embedding: conv layer for efficiency
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim,
                      kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2),  # (B, embed_dim, num_patches)
        )
        
        # CLS token and positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop  = nn.Dropout(dropout)
        
        # PRT blocks
        self.blocks = nn.ModuleList([
            PRTBlock(embed_dim, num_heads, self.num_patches_side,
                     range_r, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, num_classes)
        )
                
        # Weight initialization
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        cls_out    = x[:, 0]
        patch_mean = x[:, 1:].mean(dim=1)
        combined   = cls_out + patch_mean      # fuse global + local
        return self.head(combined)
        
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


