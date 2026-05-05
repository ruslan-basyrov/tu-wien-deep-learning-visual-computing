import torch
import torch.nn as nn

# Adapted from https://github.com/lucidrains/vit-pytorch (MIT License)
# Key ViT concepts:
#   - PatchEmbedding: split image into fixed-size patches, project each to embed_dim
#   - cls_token: learnable classification token prepended to patch sequence
#   - pos_embed: learned positional embeddings (no spatial inductive bias in ViT)
#   - TransformerBlock: multi-head self-attention lets every patch attend globally


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=192):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                      # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)      # (B, num_patches, embed_dim)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        normed = self.norm1(x)
        x = x + self.attn(normed, normed, normed)[0]   # self-attention with residual
        x = x + self.mlp(self.norm2(x))                # feed-forward with residual
        return x


class ViT(nn.Module):
    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 4,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x = self.patch_embed(x)                        # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)         # (B, 1, D)
        x = torch.cat([cls, x], dim=1)                 # (B, N+1, D)
        x = self.dropout(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x[:, 0])                         # cls token output only
        return self.head(x)
