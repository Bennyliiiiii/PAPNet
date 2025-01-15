import torch
import torch.nn as nn

class MultiStageCrossAttentionFusion(nn.Module):
    def __init__(self, pillar_dim, pointcloud_dims, num_heads, dropout=0.1):
        """
        Args:
            pillar_dim: Pillar 
            pointcloud_dims
            num_heads
            dropout
        """
        super(MultiStageCrossAttentionFusion, self).__init__()
        self.num_stages = len(pointcloud_dims)
        self.pillar_dim = pillar_dim


        self.self_attention = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=pillar_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(self.num_stages)
        ])
        self.cross_attention = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=pillar_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(self.num_stages)
        ])
        

        self.projectors = nn.ModuleList([
            nn.Linear(dim, pillar_dim)
            for dim in pointcloud_dims
        ])


        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(pillar_dim, pillar_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(pillar_dim, pillar_dim)
            ) for _ in range(self.num_stages)
        ])
        self.norm1 = nn.ModuleList([nn.LayerNorm(pillar_dim) for _ in range(self.num_stages)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(pillar_dim) for _ in range(self.num_stages)])

    def forward(self, pillar_features, pointcloud_features_list):
        """
        Args:
            pillar_features: Tensor of shape (num_pillars, batch_size, pillar_dim)
            pointcloud_features_list: List of Tensors, each of shape (num_points_i, batch_size, pointcloud_dim_i)
        Returns:
            fused_features: Tensor of shape (num_pillars, batch_size, pillar_dim)
        """
        assert len(pointcloud_features_list) == self.num_stages, 
        fused_features = pillar_features

        for i in range(self.num_stages):
            pointcloud_features = pointcloud_features_list[i]


            fused_features, _ = self.self_attention[i](fused_features, fused_features, fused_features)
            fused_features = self.norm1[i](fused_features)

            pointcloud_features = self.projectors[i](pointcloud_features)
            fused_features, _ = self.cross_attention[i](fused_features, pointcloud_features, pointcloud_features)
            fused_features = self.norm2[i](fused_features)

            fused_features = fused_features + self.ffns[i](fused_features)

        return fused_features


# 测试代码
if __name__ == "__main__":
    batch_size = 4
    num_pillars = 128
    pillar_dim = 256
    pointcloud_dims = [128, 64, 32]
    num_heads = 8


    pillar_features = torch.rand(num_pillars, batch_size, pillar_dim) 
    pointcloud_features_list = [
        torch.rand(1024, batch_size, dim) for dim in pointcloud_dims

    model = MultiStageCrossAttentionFusion(pillar_dim=pillar_dim, pointcloud_dims=pointcloud_dims, num_heads=num_heads)


    fused_features = model(pillar_features, pointcloud_features_list)

    print(f"Fused features shape: {fused_features.shape}")  

