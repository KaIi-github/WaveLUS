import torch.nn as nn
import torch
import math
import torch
import torchvision.models as models
import torch.nn.functional as F


def haar_dwt2(x):
    """Haar Wavelet Transform"""
    ll = x[:, :, ::2, ::2] + x[:, :, 1::2, ::2] + x[:, :, ::2, 1::2] + x[:, :, 1::2, 1::2]
    lh = x[:, :, ::2, ::2] + x[:, :, 1::2, ::2] - x[:, :, ::2, 1::2] - x[:, :, 1::2, 1::2]
    hl = x[:, :, ::2, ::2] - x[:, :, 1::2, ::2] + x[:, :, ::2, 1::2] - x[:, :, 1::2, 1::2]
    hh = x[:, :, ::2, ::2] - x[:, :, 1::2, ::2] - x[:, :, ::2, 1::2] + x[:, :, 1::2, 1::2]
    return ll / 4, (lh / 4, hl / 4, hh / 4)


class WaveLUS(nn.Module):
    def __init__(self, num_heads=16, num_out1=4, num_out2=4, drop_rate=0.5, device="cuda:0"):
        super(WaveLUS, self).__init__()
        self.encoder = models.resnet18(pretrained=True)
        self.num_features = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()

        self.num_heads = num_heads
        self.num_out1 = num_out1
        self.num_out2 = num_out2
        self.drop_rate = drop_rate
        self.device = device

        self.encoder_layers = list(self.encoder.children())

        if self.num_features % self.num_heads != 0:
            raise ValueError(f"num_features ({self.num_features}) must be divisible by num_heads ({self.num_heads})")

        self.subspace_size = self.num_features // self.num_heads
        self._scale = math.sqrt(self.subspace_size)
        
        self.attn_query_vecs = nn.Parameter(torch.randn(self.num_heads, self.num_features // self.num_heads))

        self.fc_out_b = nn.Sequential(nn.Dropout(p=self.drop_rate), nn.Linear(self.num_features * 4, self.num_out1))
        self.fc_out_p = nn.Sequential(nn.Dropout(p=self.drop_rate), nn.Linear(self.num_features * 4, self.num_out2))
        
    def attention_pool(self, h):
        """
        Attention Pooling
        Input:
        - h: Input features, shape (BS, N, num_features)
        Output:
        - h_vid: Pool features, shape (BS, num_features)
        - attn: Attention score, shape (BS, N, num_heads)
        """
        BS, N, D = h.shape
        
        h = h.view(BS, N, self.num_heads, self.subspace_size)  # shape: (BS, N, num_heads, subspace_size)

        # attn score
        alpha = torch.einsum('bnhi,hi->bnh', h, self.attn_query_vecs) / self._scale  # shape: (BS, N, num_heads)
        attn = torch.softmax(alpha, dim=1)  # shape: (BS, N, num_heads)

        # pooling
        h_vid = torch.einsum('bnh,bnhi->bhi', attn, h)  # shape: (BS, num_heads, subspace_size)
        h_vid = h_vid.reshape(BS, -1)  # shape: (BS, num_features)

        return h_vid, attn


    def forward(self, x):
        BS, N, C, H, W = x.shape

        x = x.view(BS * N, C, H, W)
        _, (cH, cV, cD) = haar_dwt2(x)

        h = self.encoder(x)
        h_wtch = self.encoder(cH)
        h_wtcv = self.encoder(cV)

        h = h.view(BS, N, -1)
        h_wtch = h_wtch.view(BS, N, -1)
        h_wtcv = h_wtcv.view(BS, N, -1)

        h_before = h.mean(dim=1)
        h_wtch_before = h_wtch.mean(dim=1)
        h_wtcv_before = h_wtcv.mean(dim=1)


        # attention_pool
        h_after, attn = self.attention_pool(h)
        h_wtcv_after, attn_wtc = self.attention_pool(h_wtcv)
        h_wtch_after, attn_wtc = self.attention_pool(h_wtch)

        # bline classification and regression
        p_vid = torch.cat((h_before, h_after, h_wtch_before, h_wtch_after), dim=1)
        output_p = self.fc_out_p(p_vid)

        # bline classification and regression
        b_vid = torch.cat((h_before, h_after, h_wtcv_before, h_wtcv_after), dim=1)
        output_b = self.fc_out_b(b_vid)

        return output_b, output_p, b_vid, p_vid


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vidnet_kwargs = {
        'num_heads': 16,
        'num_out1': 4,
        'num_out2': 4,
        'drop_rate': 0.5,
        'device': device,
    }
    medvidnet = WaveLUS(**vidnet_kwargs)

    n_parameters = sum(p.numel() for p in medvidnet.parameters() if p.requires_grad)
    print('number of params (M): %.2f' % (n_parameters / 1.e6))
    BS, N, H, W, C = 4, 64, 224, 224, 3
    input_data = torch.randn(BS, N, C, H, W)

    output_b, output_p, regression_output_b, regression_output_p = medvidnet(input_data)

    print("Output shape:", output_b.shape)  # [BS, num_out]