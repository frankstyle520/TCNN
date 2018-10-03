import torch.nn as nn

# CNN遗留问题
# 不均衡的处理
# 代码的检查
# 复现论文的结果

# TCNN遗留问题
# MMD无法计算，它要求源域和目标域的个数一致

class CNN(nn.Module):
    def __init__(self, params):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(params['DICT_SIZE'], params['EMBED_DIM'])
        self.conv = nn.Conv1d(in_channels=params['TOKEN_SIZE'],    # 本问题in_channel就是1
                              out_channels=params['N_FILTER'],     # Filter的个数
                              kernel_size=params['FILTER_SIZE'],   # Filter的大小
                              stride=params['STRIDE'],             # Filter的步长
                              padding=params['PADDING']            # suggested padding = (kernel_size-stride)/2
                             )
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=params['POOL_SIZE']) # pool_size

        # new dim is (EMBED_DIM - FILTER_SIZE + 2 * padding) / stride + 1
        new_dim = (params['EMBED_DIM'] - params['FILTER_SIZE'] + 2 * params['PADDING']) / params['STRIDE'] + 1
        new_dim = new_dim / params['POOL_SIZE']
        new_dim = int(new_dim)

        self.fc = nn.Linear(params['N_FILTER'] * new_dim, params['N_HIDDEN_NODE'])
        self.out = nn.Linear(params['N_HIDDEN_NODE'], 2)

    def forward(self, x):
        x = self.embedding(x)
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        y = self.out(x)
        return y
