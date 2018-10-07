import torch.nn as nn


class TCNN(nn.Module):
    def __init__(self, params):
        super(TCNN, self).__init__()
        self.embedding = nn.Embedding(params['DICT_SIZE'], params['EMBED_DIM'])
        self.conv = nn.Conv1d(in_channels=params['TOKEN_SIZE'],  # 本问题in_channel就是1
                              out_channels=params['N_FILTER'],  # Filter的个数
                              kernel_size=params['FILTER_SIZE'],  # Filter的大小
                              stride=params['STRIDE'],  # Filter的步长
                              padding=params['PADDING']  # suggested padding = (kernel_size-stride)/2
                              )
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=params['POOL_SIZE'])  # pool_size

        # new dim is (EMBED_DIM - FILTER_SIZE + 2 * padding) / stride + 1
        new_dim = (params['EMBED_DIM'] - params['FILTER_SIZE'] + 2 * params['PADDING']) / params['STRIDE'] + 1
        new_dim = new_dim / params['POOL_SIZE']
        new_dim = int(new_dim)

        self.fc = nn.Linear(params['N_FILTER'] * new_dim, params['N_HIDDEN_NODE'])
        self.out = nn.Linear(params['N_HIDDEN_NODE'], 2)

    def forward(self, x_src, x_tar):
        x_src = self.embedding(x_src)
        x_tar = self.embedding(x_tar)
        x_src = self.conv(x_src)
        x_tar = self.conv(x_tar)
        x_src = self.relu(x_src)
        x_tar = self.relu(x_tar)
        x_src = self.pool(x_src)
        x_tar = self.pool(x_tar)
        x_src = x_src.view(x_src.size(0), -1)
        x_tar = x_tar.view(x_tar.size(0), -1)

        # mmd计算用哪一层的请考虑
        x_src_mmd = x_src
        x_tar_mmd = x_tar

        x_src = self.fc(x_src)
        y_src = self.out(x_src)
        return y_src, x_src_mmd, x_tar_mmd
