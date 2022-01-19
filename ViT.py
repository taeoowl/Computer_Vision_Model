import torch
import torch.nn as nn
import torch.nn.functional as F

from einops.layers.torch import Rearrange

class InputEmbed(nn.Module):
    def __init__(self, img, patch_size, input_dim):
        super().__init__()
        b, c, h, w = img.shape
        patch_flatten_dim = c * patch_size**2
        num_patches = h*w // patch_size**2
        
        self.patch_embed = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_flatten_dim, input_dim)
        )
        
        self.cls_embed = nn.Parameter(torch.rand(1, 1, input_dim))
        self.pos_embed = nn.Parameter(torch.rand(1, num_patches+1, input_dim))
        
    def forward(self, x):
        patch_embedding = self.patch_embed(x)
        cls_embedding = torch.concat([self.cls_embed, patch_embedding], dim=1)
        input_embedding = self.pos_embed + cls_embedding
        return input_embedding

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, num_patches, input_dim):
        super().__init__()
        qkv_dim = input_dim // heads
        
        self.layernorm = nn.LayerNorm(input_dim)
        self.scale = qkv_dim ** -0.5
        self.to_q = nn.Linear(input_dim, qkv_dim, bias = False)
        self.to_k = nn.Linear(input_dim, qkv_dim, bias = False)
        self.to_v = nn.Linear(input_dim, qkv_dim, bias = False)
        self.attend = nn.Softmax(dim=-1)
        self.block_out = nn.Linear(qkv_dim, input_dim)

    def forward(self, x):
        x = self.layernorm(x)
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        dot_product = torch.matmul(q, k.transpose(2, 1)) * self.scale
        att_weight = self.attend(dot_product)
        att_result = torch.matmul(att_weight, v)
        att_block_output = self.block_out(att_result)
        return att_block_output

class MLPBlock(nn.Module):
    def __init__(self, input_dim, mlp_dim):
        super().__init__()
        
        self.mlp_block = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, input_dim)
        )
        
    def forward(self, x):
        mlp_block_output = self.mlp_block(x)
        return mlp_block_output
    
class Encoder(nn.Module):
    def __init__(self, heads, num_patches, input_dim, mlp_dim, layers):
        super().__init__()
        
        self.layers = layers
        self.att_block = MultiHeadAttention(heads, num_patches, input_dim)
        self.mlp_block = MLPBlock(input_dim, mlp_dim)
    
    def forward(self, x):
        for i in range(layers):
            x = self.att_block(x) + x
            x = self.mlp_block(x) + x
        return x
        
    
class ClassificationHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cls_head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, num_classes)
        )
        
    def forward(self, x):
        cls_embedding = x[:, 0]
        cls_pred = self.cls_head(cls_embedding)
        return cls_pred

class ViT(nn.Module):
    def __init__(self, img, patch_size, input_dim, heads, num_patches, mlp_dim, layers, num_classes):
        super().__init__()
        self.input_embed = InputEmbed(img, patch_size, input_dim)
        self.encoder = Encoder(heads, num_patches, input_dim, mlp_dim, layers)
        self.cls_head = ClassificationHead(num_classes)
        
    def forward(self, x):
        input_embeddings = self.input_embed(x)
        encoder_output = self.encoder(input_embeddings)
        cls_pred = self.cls_head(encoder_output)
        return cls_pred
    
# TEST CODE 
# img = torch.randn(1, 3, 48, 48)
# vit = ViT(img, 16, 1024, 8, 9, 2048, 8, 1000)
# vit(img)
