from ParsingSourceCapsule import *
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
from sklearn.metrics import matthews_corrcoef, roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
import CNN
from tool.imblearn.under_sampling import RandomUnderSampler
from tool.imblearn.over_sampling import RandomOverSampler
import itertools


# -------辅助方法--------start
def insert_param(training, test, cnn_params, loop_size, data):
    data['Training Set'].append(training)
    data['Test Set'].append(test)
    data['Embedding Dim'].append(cnn_params['EMBED_DIM'])
    data['Number of Filter'].append(cnn_params['N_FILTER'])
    data['Filter Size'].append(cnn_params['FILTER_SIZE'])
    data['Number of Hidden Nodes'].append(cnn_params['N_HIDDEN_NODE'])
    data['Learning Rate'].append(cnn_params['LEARNING_RATE'])
    data['Momentun'].append(cnn_params['MOMEMTUN'])
    data['L2 Weight'].append(cnn_params['L2_WEIGHT'])
    data['Dropout'].append(cnn_params['DROPOUT'])
    data['Number of Epoch'].append(cnn_params['N_EPOCH'])
    data['Stride'].append(cnn_params['STRIDE'])
    data['Padding'].append(cnn_params['PADDING'])
    data['Batch Size'].append(cnn_params['BATCH_SIZE'])
    data['Pool Size'].append(cnn_params['POOL_SIZE'])
    data['Loop Size'].append(str(loop_size))


def insert_result(acc_m, acc_s, auc_m, auc_s, f1_m, f1_s, mcc_m, mcc_s, data):
    data['accuracy_mean'].append(round(acc_m, 3))
    data['accuracy_std'].append(round(acc_s, 3))
    data['AUC_mean'].append(round(auc_m, 3))
    data['AUC_std'].append(round(auc_s, 3))
    data['F-measure_mean'].append(round(f1_m, 3))
    data['F-measure_std'].append(round(f1_s, 3))
    data['MCC_mean'].append(round(mcc_m, 3))
    data['MCC_std'].append(round(mcc_s, 3))

# -------辅助方法--------end

# TCNN遗留问题
# MMD无法度量

# 声明各种参数
LOOP_SIZE = 3
REGENERATE = True
IMBALANCE_PROCESSOR = RandomUnderSampler() # RandomOverSampler(), RandomUnderSampler(), None, 'cost'
HANDCRAFT_DIM = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
root_path_source = 'data/projects/'
root_path_csv = 'data/csvs/'

# 超参
init_cnn_params = {'EMBED_DIM': 30, 'N_FILTER': 50, 'FILTER_SIZE': 5, 'N_HIDDEN_NODE': 100,
                   'LEARNING_RATE': 0.02, 'MOMEMTUN': 0.05, 'L2_WEIGHT': 0.003, 'DROPOUT': 0.5, 'N_EPOCH': 15,
                   'STRIDE': 1, 'PADDING': 0, 'BATCH_SIZE': 32, 'POOL_SIZE': 2, 'DICT_SIZE': 0, 'TOKEN_SIZE': 0}

opt_n_filter = [10, 15]
opt_filter_size = [5]
opt_n_hidden_node = [50]
opt_n_epoch = [10]

# 解析源项目和目标项目
path_train_and_test = []
with open('data/pairs-one.txt', 'r') as file_obj:
    for line in file_obj.readlines():
        line = line.strip('\n')
        line = line.strip(' ')
        path_train_and_test.append(line.split(','))

# 循环每一对组合
for path in path_train_and_test:

    # 记录数据
    record_data = {'Training Set': [], 'Test Set': [], 'Embedding Dim': [], 'Number of Filter': [], 'Filter Size': [],
                   'Number of Hidden Nodes': [], 'Learning Rate': [], 'Momentun': [], 'L2 Weight': [],
                   'Dropout': [], 'Number of Epoch': [], 'Stride': [], 'Padding': [], 'Batch Size': [],
                   'Pool Size': [], 'Loop Size': [], 'accuracy_mean': [], 'accuracy_std': [],
                   'AUC_mean': [], 'AUC_std': [], 'F-measure_mean': [], 'F-measure_std': [], 'MCC_mean': [], 'MCC_std': []}

    # 取文件
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
    # criterion = nn.CrossEntropyLoss()

    # 数据从numpy转tensor
    train_ast = torch.cuda.LongTensor(train_ast)
    train_label = torch.cuda.LongTensor(train_label)
    test_ast = torch.cuda.LongTensor(test_ast)

    # 循环每一种参数
    for params in itertools.product(opt_n_filter, opt_filter_size, opt_n_hidden_node, opt_n_epoch):
        cnn_params = init_cnn_params.copy()
        cnn_params['N_FILTER'] = params[0]
        cnn_params['FILTER_SIZE'] = params[1]
        cnn_params['N_HIDDEN_NODE'] = params[2]
        cnn_params['N_EPOCH'] = params[3]

        # 数据集
        train_dataset = Data.TensorDataset(train_ast, train_label)
        loader = Data.DataLoader(dataset=train_dataset, batch_size=cnn_params['BATCH_SIZE'], shuffle=True)

        # 循环次数
        acc, auc, f1, mcc = [], [], [], []
        for l in range(LOOP_SIZE):

            # 模型
            model = CNN.CNN(cnn_params)
            model.to(DEVICE)

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
                    weight = torch.cuda.FloatTensor(weight)
                    criterion = nn.CrossEntropyLoss(weight=weight)

                    loss = criterion(y_src, batch_y.squeeze())
                    print('loss: ' + str(loss))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # 预测
            test_output = model(test_ast)
            pred_y = test_output.data.max(1)[1]

            # 记录评估结果
            acc.append(accuracy_score(y_true=test_label, y_pred=pred_y))
            auc.append(roc_auc_score(y_true=test_label, y_score=pred_y))
            f1.append(f1_score(y_true=test_label, y_pred=pred_y))
            mcc.append(matthews_corrcoef(y_true=test_label, y_pred=pred_y))

        # 计算循环后的平均值和标准差
        acc, auc, f1, mcc = np.array(acc), np.array(auc), np.array(f1), np.array(mcc)
        acc_m, acc_s, auc_m, auc_s, f1_m, f1_s, mcc_m, mcc_s = acc.mean(), acc.std(), auc.mean(), auc.std(), f1.mean(), f1.std(), mcc.mean(), mcc.std()

        # 将结果存入文件中
        insert_result(acc_m, acc_s, auc_m, auc_s, f1_m, f1_s, mcc_m, mcc_s, data=record_data)
        insert_param(training=path[0], test=path[1], cnn_params=cnn_params, loop_size=LOOP_SIZE, data=record_data)
        df = pd.DataFrame(data=record_data, columns=['Training Set', 'Test Set', 'Embedding Dim', 'Number of Filter', 'Filter Size',
                                                             'Number of Hidden Nodes', 'Learning Rate', 'Momentun', 'L2 Weight',
                                                             'Dropout', 'Number of Epoch', 'Stride', 'Padding', 'Batch Size', 'Pool Size', 'Loop Size',
                                                             'accuracy_mean', 'accuracy_std', 'AUC_mean', 'AUC_std', 'F-measure_mean',
                                                             'F-measure_std', 'MCC_mean', 'MCC_std'])

        df.to_csv('result/pair_' + path[0] + '_' + path[1] + '.csv', index=False)