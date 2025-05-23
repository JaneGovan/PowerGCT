import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.Attention import AttentionLayer, RotaryAttention, FullAttention
from layers.Embeding import PowerGCTEmbedding, PowerGCTEmbedding_wo_chunk, PowerGCTEmbedding_wo_rope, PowerGCTEmbedding_wo_chunk_and_rope
from layers.Dense import DenseNet


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):
    def __init__(self, configs, patch_len=16, stride=16):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        padding = stride

        # patching and embedding
        self.patch_embedding = PowerGCTEmbedding_wo_rope(
            configs.d_model, patch_len, stride, padding, configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model)
        )

        # Prediction Head
        self.head_nf = configs.d_model * \
                       int((configs.seq_len - patch_len) / stride + 2)

        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(configs.dropout)

        self.type_projection = nn.Sequential(
            nn.LayerNorm(self.head_nf * configs.enc_in),
            nn.Linear(self.head_nf * configs.enc_in, self.head_nf * configs.enc_in // 2),
            nn.GELU(),
            nn.LayerNorm(self.head_nf * configs.enc_in // 2),
            nn.Linear(self.head_nf * configs.enc_in // 2, configs.num_class)
        )
        self.loc_projection = DenseNet(layer_num=(4, 8, 4, 2), growth_rate=1, in_channels=1, classes=configs.num_class)

    def classification(self, x_enc, x_mark_enc):
        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        output = self.flatten(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output1 = self.type_projection(output)  # (batch_size, num_classes)
        # output = output.unsqueeze(dim=1)
        # output2 = self.loc_projection(output)
        return output1

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.classification(x_enc, x_mark_enc)
        return dec_out  # [B, N]
