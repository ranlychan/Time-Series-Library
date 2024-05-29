import torch.nn as nn

class Model(nn.Module):
    def __init__(self, configs): #input_size, hidden_size, num_layers
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.lstm_layer_num = configs.lstm_layer_num
        self.lstm_hidden_dims = configs.lstm_hidden_dims
        self.c_out = configs.c_out

        self.rnn = nn.GRU(
            input_size=configs.enc_in,
            hidden_size=configs.lstm_hidden_dims,
            num_layers=configs.lstm_layer_num,
            batch_first=True,
        )
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(configs.lstm_hidden_dims, configs.c_out)

    def forecast(self, x_enc):
        r_out, _ = self.rnn(x_enc, None)  # None represents zero initial hidden state
        out = self.out(r_out)  # return the last value #[:, -1, :]
        out = self.sigmoid(out)
        return out
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            out = self.forecast(x_enc)
            return out #dec_out[:, -self.pred_len:, :]  # [B, L, D]
        return None