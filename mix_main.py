from ParsingSourceCapsule import *
from TrainATest import *
import torch
import torch.utils.data as Data
import datetime


import TCNN
from tool.imblearn.under_sampling import RandomUnderSampler
from tool.imblearn.over_sampling import RandomOverSampler
import itertools


# -------辅助方法--------start
def insert_param(training, test, model_name, cnn_params, mmd_params, loop_size, data):
    data['Training Set'].append(training)
    data['Test Set'].append(test)
    data['Model'].append(model_name)

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

    data['MMD Lambda'].append(mmd_params['MMD_LAMBDA'])
    data['MMD Gamma'].append(mmd_params['MMD_GAMMA'])

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


def save_data(training, test, data):
    df = pd.DataFrame(data=data,
                      columns=['Training Set', 'Test Set', 'Model', 'Embedding Dim', 'Number of Filter', 'Filter Size',
                               'Number of Hidden Nodes', 'Learning Rate', 'Momentun', 'L2 Weight',
                               'Dropout', 'Number of Epoch', 'Stride', 'Padding', 'Batch Size', 'Pool Size',
                               'MMD Lambda', 'MMD Gamma', 'Loop Size',
                               'accuracy_mean', 'accuracy_std', 'AUC_mean', 'AUC_std', 'F-measure_mean',
                               'F-measure_std', 'MCC_mean', 'MCC_std'])

    df.to_csv('result/pair_' + training + '_' + test + '.csv', index=False)

# -------辅助方法--------end

# TCNN遗留问题
# MMD无法度量

# 声明各种参数
LOOP_SIZE = 5
REGENERATE = True
IMBALANCE_PROCESSOR = RandomUnderSampler() # RandomOverSampler(), RandomUnderSampler(), None, 'cost'
HANDCRAFT_DIM = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
root_path_source = 'data/projects/'
root_path_csv = 'data/csvs/'

# 卷积网络的超参
init_cnn_params = {'EMBED_DIM': 30, 'N_FILTER': 50, 'FILTER_SIZE': 5, 'N_HIDDEN_NODE': 100,
                   'LEARNING_RATE': 0.02, 'MOMEMTUN': 0.05, 'L2_WEIGHT': 0.003, 'DROPOUT': 0.5, 'N_EPOCH': 15,
                   'STRIDE': 1, 'PADDING': 0, 'BATCH_SIZE': 32, 'POOL_SIZE': 2, 'DICT_SIZE': 0, 'TOKEN_SIZE': 0}

# MMD的超参
init_mmd_params = {'MMD_LAMBDA': 0.5, 'MMD_GAMMA': 10e3}

# EMBED_DIM、N_EPOCH、STRIDE、PADDING、BATCH_SIZE、POOL_SIZE用的都是经验值
opt_lambda = [0.1, 0.25, 0.5, 0.75, 1]
opt_gamma = [10e0, 10e1, 10e2, 10e3, 10e4]

# opt_mometun
# opt_l2_weight

opt_n_filter = [1, 2, 3, 5, 10, 20, 50, 100, 150, 200]
opt_filter_size = [2, 3, 5, 10, 20, 50, 100]
opt_n_hidden_node = [10, 20, 30, 50, 100, 150, 200, 250]
opt_learning_rate = [10e-4, 10e-3, 10e-2, 10e-1]

# 打印开始时间
print(datetime.datetime.now())

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
    record_data = {'Training Set': [], 'Test Set': [], 'Model':[], 'Embedding Dim': [], 'Number of Filter': [], 'Filter Size': [],
                   'Number of Hidden Nodes': [], 'Learning Rate': [], 'Momentun': [], 'L2 Weight': [],
                   'Dropout': [], 'Number of Epoch': [], 'Stride': [], 'Padding': [], 'Batch Size': [],
                   'Pool Size': [], 'MMD Lambda': [], 'MMD Gamma': [], 'Loop Size': [], 'accuracy_mean': [], 'accuracy_std': [],
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

    # 循环每一种nn参数
    for params_i in itertools.product(opt_n_filter, opt_filter_size, opt_n_hidden_node, opt_learning_rate):
        nn_params = init_cnn_params.copy()
        nn_params['N_FILTER'] = params_i[0]
        nn_params['FILTER_SIZE'] = params_i[1]
        nn_params['N_HIDDEN_NODE'] = params_i[2]
        nn_params['LEARNING_RATE'] = params_i[3]
        nn_params['DICT_SIZE'] = vocabulary_size + 1
        nn_params['TOKEN_SIZE'] = vector_len

        # 数据集
        train_dataset = Data.TensorDataset(train_ast, train_label)
        loader = Data.DataLoader(dataset=train_dataset, batch_size=nn_params['BATCH_SIZE'], shuffle=True)

        # CNN
        model_name = 'CNN'
        acc_m, acc_s, auc_m, auc_s, f1_m, f1_s, mcc_m, mcc_s = CNN_train_and_test(LOOP_SIZE, nn_params, loader, test_ast, test_label, DEVICE)
        # 将结果存入文件中
        insert_result(acc_m, acc_s, auc_m, auc_s, f1_m, f1_s, mcc_m, mcc_s, data=record_data)
        insert_param(training=path[0], test=path[1], model_name=model_name, cnn_params=nn_params, mmd_params=init_mmd_params, loop_size=LOOP_SIZE, data=record_data)
        save_data(path[0], path[1], record_data)

        # TCNN
        # model_name = 'TCNN'
        # # 循环每一种mmd参数
        # for params_j in itertools.product(opt_lambda, opt_gamma):
        #     mmd_params = init_mmd_params.copy()
        #     mmd_params['MMD_LAMBDA'] = params_j[0]
        #     mmd_params['MMD_GAMMA'] = params_j[1]
        #
        #     acc_m, acc_s, auc_m, auc_s, f1_m, f1_s, mcc_m, mcc_s = TCNN_train_and_test(LOOP_SIZE, nn_params, mmd_params, loader, test_ast, test_label, DEVICE)
        #     # 将结果存入文件中
        #     insert_result(acc_m, acc_s, auc_m, auc_s, f1_m, f1_s, mcc_m, mcc_s, data=record_data)
        #     insert_param(training=path[0], test=path[1], model_name=model_name, cnn_params=nn_params, mmd_params=mmd_params, loop_size=LOOP_SIZE, data=record_data)
        #     save_data(path[0], path[1], record_data)

# 打印结束时间
print(datetime.datetime.now())
