import torch
import torch.nn as nn

# 定义卷积网络 (CNN)
class SimpleCNN(nn.Module):
    def __init__(self,in_channels=12, out_channels=144, dropout=0.5):
        super(SimpleCNN, self).__init__()
        self.blocks = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout(0.5),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout(0.5),
            nn.Conv2d(in_channels=128, out_channels=in_channels*out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
        )

        # self.nor_blocks = nn.Sequential(
        #     nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     # nn.Dropout(0.5),
        #     nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     # nn.Dropout(0.5),
        #     nn.Conv2d(in_channels=128, out_channels=in_channels*out_channels, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Dropout(0.5),
        # )

        # 初始化权重
        for m in self.blocks.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.blocks(x)
    
class AttentionCNN(nn.Module):
    def __init__(self,in_channels=12, out_channels=144, att_heads = 1, dropout=0.5):
        super(AttentionCNN, self).__init__()
        self.blocks = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout(0.5),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout(0.5),
            nn.Conv2d(in_channels=128, out_channels=in_channels*out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
            MHSAttention(embed_dim=out_channels, num_heads=att_heads),
            nn.Linear()
        )
        # 初始化权重
        for m in self.blocks.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.blocks(x)

class AttentionCorrModule(nn.Module):
    def __init__(self, in_channels=12, out_channels=144, att_heads =1, dropout=0.5):
        super(AttentionCorrModule, self).__init__()
        self.blocks = nn.Sequential(
            SimpleCNN(in_channels, out_channels, dropout),
            nn.Flatten(start_dim=1),
            nn.Unflatten(1, (in_channels, -1)),
            MHSAttention(embed_dim=out_channels, num_heads=att_heads),
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=in_channels*out_channels,out_features=in_channels*out_channels),
            nn.Unflatten(1, (in_channels, -1)),
        )
        # output (batch_size, seq_len, input_size)
    def forward(self, x):
        x1 = self.blocks(x)
        # concat attention cnn output vector with corresponding TM vector
        x = torch.cat((x.view(x1.shape), x1), 2)
        return x
    
class CorrModule(nn.Module):
    def __init__(self, in_channels=12, out_channels=144, dropout=0.5):
        super(CorrModule, self).__init__()
        self.blocks = nn.Sequential(
            SimpleCNN(in_channels, out_channels, dropout),
            nn.Flatten(start_dim=1),
            nn.Unflatten(1, (in_channels, -1)),
        )
        # output (batch_size, seq_len, input_size)
    def forward(self, x):
        x1 = self.blocks(x)

        # concat cnn output vector with corresponding TM vector

        # print(f'x.shape={x.shape}')
        # print(f'x.view(out.shape).shape={x.view(out.shape).shape}')
        # print(f'out.shape={out.shape}')
        x = torch.cat((x.view(x1.shape), x1), 2)
        # print(f'concat x shape={x.shape}')
        return x

# 定义双层LSTM
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.5):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        return out

# 定义注意力机制
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        # Compute attention scores
        scores = torch.matmul(x, x.transpose(-2, -1)) / (self.hidden_size ** 0.5)
        # Apply softmax to get the attention weights
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)
        # Compute the attention output
        context = torch.matmul(attn_weights, x)
        return context

class MHSAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MHSAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        attn_output, attn_output_weights = self.attention(x,x,x)
        return attn_output
    

# 定义整个模型
class CRNN(nn.Module):
    def __init__(self, in_channels=12, out_channels=144, step=1, lstm_hidden=288, lstm_layer=2, att_hidden=144):
        super(CRNN, self).__init__()
        self.blocks = nn.Sequential(
            CorrModule(in_channels, out_channels),
            LSTM(input_size=2*out_channels, hidden_size=lstm_hidden, num_layers=lstm_layer),
        )
        self.fc = nn.Linear(lstm_hidden, step*in_channels*in_channels)

    def forward(self, x):
        x = self.blocks(x)[:, -1, :] # return the last value
        print(x.shape)
        return self.fc(x)

class Model(nn.Module):
    def __init__(self, configs):
        #in_channels=12, out_channels=144, step=3, lstm_hidden=288, lstm_layer=2, att_hidden=144, att_heads=12
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.n_heads = configs.n_heads
        self.dropout = configs.dropout
        self.lstm_layer_num = configs.lstm_layer_num
        self.lstm_hidden_dims = configs.lstm_hidden_dims
        self.matrix_dim = configs.matrix_dim
        self.c_out = configs.c_out

        self.blocks = nn.Sequential(
            nn.Unflatten(2, (configs.matrix_dim, configs.matrix_dim)),
            AttentionCorrModule(configs.seq_len, configs.c_out, configs.n_heads, configs.dropout),
            LSTM(input_size=2*configs.c_out, hidden_size=configs.lstm_hidden_dims, num_layers=configs.lstm_layer_num),
            nn.Sigmoid(),
            MHSAttention(embed_dim=configs.lstm_hidden_dims, num_heads=configs.n_heads),
            # nn.Flatten(start_dim=1),
            nn.Linear(2*configs.c_out, configs.c_out),
            # nn.Unflatten(1, (configs.pred_len, -1)),
        )
        # self.attention = nn.MultiheadAttention(embed_dim=2*out_channels, num_heads=att_heads, batch_first=True)
        # self.fc = nn.Linear(2*in_channels*out_channels, step*out_channels)


    def forecast(self, x_enc):
        return self.blocks(x_enc)
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # print(f'MACRNN input shape={x_enc.shape}')
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            out = self.forecast(x_enc)
            return out #dec_out[:, -self.pred_len:, :]  # [B, L, D]
        return None