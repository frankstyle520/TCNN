from ParsingSourceCapsule import *
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
from sklearn.metrics import matthews_corrcoef, roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
import CNN
from tool.imblearn.under_sampling import RandomUnderSampler
from tool.imblearn.over_sampling import RandomOverSampler

# 遗留问题
# 模型的输入输出需检查
# 不均衡的处理

# TCNN遗留问题
# MMD无法度量

# 声明各种参数
LOOP_SIZE = 1
REGENERATE = True
IMBALANCE_PROCESSOR = RandomUnderSampler() # RandomOverSampler(), RandomUnderSampler(), None, 'cost'
HANDCRAFT_DIM = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
root_path_source = 'data/projects/'
root_path_csv = 'data/csvs/'

# 超参
cnn_params = {'EMBED_DIM': 30, 'N_FILTER': 50, 'FILTER_SIZE': 5, 'N_HIDDEN_NODE': 100, 'GAMMA': 10 ^ 3,
              'LEARNING_RATE': 0.02, 'MOMEMTUN': 0.05, 'L2_WEIGHT': 0.003, 'DROPOUT': 0.5, 'N_EPOCH': 15,
              'STRIDE': 1, 'PADDING': 0, 'BATCH_SIZE': 32, 'POOL_SIZE': 2, 'DICT_SIZE': 0, 'TOKEN_SIZE': 0}

# 解析源项目和目标项目
path_train_and_test = []
with open('data/pairs-one.txt', 'r') as file_obj:
    for line in file_obj.readlines():
        line = line.strip('\n')
        line = line.strip(' ')
        path_train_and_test.append(line.split(','))

# 循环每一对组合
for path in path_train_and_test:
    path_train_source = root_path_source + path[0]
    path_train_handcraft = root_path_csv + path[0] + '.csv'
    path_test_source = root_path_source + path[1]
    path_test_handcraft = root_path_csv + path[1] + '.csv'
    package_heads = path[2:]

    # 获取训练集和测试集的实例名单
    train_file_instances = extract_handcraft_instances(path_train_handcraft)
    test_file_instances = extract_handcraft_instances(path_test_handcraft)

    # 获取tokens
    dict_token_train = parse_source(path_train_source, train_file_instances, package_heads)
    dict_token_test = parse_source(path_test_source, test_file_instances, package_heads)

    # 把tokens转成数字
    list_dict, vector_len, vocabulary_size = transform_token_to_number([dict_token_train, dict_token_test])
    dict_encoding_train = list_dict[0]
    dict_encoding_test = list_dict[1]
    cnn_params['DICT_SIZE'] = vocabulary_size + 1
    cnn_params['TOKEN_SIZE'] = vector_len

    # 取出可用于训练的数据
    train_ast, train_hand_craft, train_label = extract_data(path_train_handcraft, dict_encoding_train)
    test_ast, test_hand_craft, test_label = extract_data(path_test_handcraft, dict_encoding_test)

    # 不均衡处理
    # train_ast, train_hand_craft, train_label = imbalance_process(train_ast, train_hand_craft, train_label, IMBALANCE_PROCESSOR)

    # 数据从numpy转tensor
    train_ast = torch.from_numpy(train_ast)
    train_label = torch.from_numpy(train_label)
    test_ast = torch.from_numpy(test_ast)

    # 数据集
    train_dataset = Data.TensorDataset(train_ast, train_label)
    loader = Data.DataLoader(dataset=train_dataset, batch_size=cnn_params['BATCH_SIZE'], shuffle=True)

    # 模型
    model = CNN.CNN(cnn_params)

    # 优化器用Momentum
    optimizer = optim.SGD(model.parameters(), lr=cnn_params['LEARNING_RATE'], momentum=cnn_params['MOMEMTUN'], weight_decay=cnn_params['L2_WEIGHT'])

    # 训练
    for epoch in range(cnn_params['N_EPOCH']):
        print('epoch: ' + str(epoch))
        for step, (batch_x, batch_y) in enumerate(loader):
            model.train()
            y_src = model(batch_x)

            # 代价敏感损失函数
            weight = generate_weight_for_class(batch_y)
            weight = torch.FloatTensor(weight)
            criterion = nn.CrossEntropyLoss(weight=weight)

            loss = criterion(y_src, batch_y.squeeze())
            print('loss: ' + str(loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 预测
    test_output = model(test_ast)
    pred_y = test_output.data.max(1)[1]

    # 评估结果
    acc = accuracy_score(y_true=test_label, y_pred=pred_y)
    auc = roc_auc_score(y_true=test_label, y_score=pred_y)
    f1 = f1_score(y_true=test_label, y_pred=pred_y)
    mcc = matthews_corrcoef(y_true=test_label, y_pred=pred_y)

    print('acc: ' + str(acc))
    print('auc: ' + str(auc))
    print('f1: ' + str(f1))
    print('mcc: ' + str(mcc))
