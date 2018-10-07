from TCNN_keras_p import get_tcnn_model
from ParsingSourceCapsule import *
from tool.imblearn.under_sampling import RandomUnderSampler
from tool.imblearn.over_sampling import RandomOverSampler
import itertools
from keras import backend as K
from sklearn.metrics import matthews_corrcoef, roc_auc_score, f1_score, precision_score, recall_score, accuracy_score


# 设置参数
init_cnn_params = {'input_dim': 3709, 'output_dim': 30, 'input_length': 2405, 'filters': 50, 'kernel_size': 20,
              'pool_size': 2, 'hidden_units': 150, 'hand_craft_input_dim': 20, 'metrics': ['acc'],
              'batch_size': 32, 'epochs': 15, 'imbalance': 'none', 'regenerate': True}
# num_filter = [1, 2, 3, 5, 10, 20, 50, 100, 150, 200]
# len_filter = [5, 10, 20, 50, 100]
# num_hidden_node = [10, 20, 30, 50, 100, 150, 200, 250]
# epoch = [15]
# loopLen = 20
# mmd_lambda = [1]
# imbalance = ['none', 'cost', 'ros', 'rus']
num_filter = [10]
len_filter = [5]
num_hidden_node = [100]
epoch = [15]
loopLen = 5
mmd_lambda = [1]
imbalance = ['none']
path_train_and_test = []

root_path_source = 'data/projects/'
root_path_csv = 'data/csvs/'


def insert_data(source_train, source_test, dict_params, loop_count, data):
    data['training'].append(source_train)
    data['test'].append(source_test)
    data['Number of Filter'].append(dict_params['filters'])
    data['Filter Length'].append(dict_params['kernel_size'])
    data['Number of Hidden Nodes'].append(dict_params['hidden_units'])
    data['epoch'].append(dict_params['epochs'])
    data['embedding'].append(dict_params['output_dim'])
    data['ImbMethod'].append(dict_params['imbalance'])
    data['loop'].append(str(loop_count))


def insert_score(acc_m, acc_s, pre_m, pre_s, rec_m, rec_s, auc_m, auc_s, f1_m, f1_s, mcc_m, mcc_s, data):
    data['accuracy_mean'].append(round(acc_m, 3))
    data['accuracy_std'].append(round(acc_s, 3))
    data['precision_mean'].append(round(pre_m, 3))
    data['precision_std'].append(round(pre_s, 3))
    data['recall_mean'].append(round(rec_m, 3))
    data['recall_std'].append(round(rec_s, 3))
    data['AUC_mean'].append(round(auc_m, 3))
    data['AUC_std'].append(round(auc_s, 3))
    data['F-measure_mean'].append(round(f1_m, 3))
    data['F-measure_std'].append(round(f1_s, 3))
    data['MCC_mean'].append(round(mcc_m, 3))
    data['MCC_std'].append(round(mcc_s, 3))


def train_and_test_model_tcnn(train_ast, train_hand_craft, train_label, test_ast, test_label, loopLen, dict_params):
    acc = []
    pre = []
    rec = []
    auc = []
    f1 = []
    mcc = []

    weight = None
    for l in range(loopLen):
        if 'cost' == dict_params['imbalance']:
            weight = generate_weight(train_label)
        else:
            if 'none' == dict_params['imbalance']:
                processor = None
            elif 'rus' == dict_params['imbalance']:
                processor = RandomUnderSampler()
            elif 'ros' == dict_params['imbalance']:
                processor = RandomOverSampler()
            train_ast, train_hand_craft, train_label = imbalance_process(train_ast, train_hand_craft, train_label, processor)

        model = get_tcnn_model(dict_params, train_ast, test_ast, mmd_lambda)
        model.fit(train_ast, train_label, batch_size=dict_params['batch_size'], sample_weight=weight, epochs=dict_params['epochs'])
        y_predict = round_self(model.predict(test_ast, batch_size=dict_params['batch_size']))
        K.clear_session()

        acc.append(accuracy_score(y_true=test_label, y_pred=y_predict))
        pre.append(precision_score(y_true=test_label, y_pred=y_predict))
        rec.append(recall_score(y_true=test_label, y_pred=y_predict))
        auc.append(roc_auc_score(y_true=test_label, y_score=y_predict))
        f1.append(f1_score(y_true=test_label, y_pred=y_predict))
        mcc.append(matthews_corrcoef(y_true=test_label, y_pred=y_predict))
    acc = np.array(acc)
    pre = np.array(pre)
    rec = np.array(rec)
    auc = np.array(auc)
    f1 = np.array(f1)
    mcc = np.array(mcc)

    return acc.mean(), acc.std(), pre.mean(), pre.std(), rec.mean(), rec.std(), auc.mean(), auc.std(), f1.mean(), f1.std(), mcc.mean(), mcc.std()


with open('data/pairs.txt', 'r') as file_obj:
    for line in file_obj.readlines():
        line = line.strip('\n')
        line = line.strip(' ')
        path_train_and_test.append(line.split(','))

# 循环每一对组合
for path in path_train_and_test:
    path_train_source = root_path_source + path[0]
    path_train_hand_craft = root_path_csv + path[0] + '.csv'
    path_test_source = root_path_source + path[1]
    path_test_hand_craft = root_path_csv + path[1] + '.csv'
    package_heads = path[2:]

    train_file_names, test_file_names = extract_hand_craft_file_name(path_train_hand_craft, path_test_hand_craft)
    dict_token_train = parse_source(path_train_source, train_file_names, package_heads)
    dict_token_test = parse_source(path_test_source, test_file_names, package_heads)

    list_dict, vector_len, vocabulary_size = transform_token_to_number([dict_token_train, dict_token_test])
    dict_encoding_train = list_dict[0]
    dict_encoding_test = list_dict[1]
    train_ast, train_hand_craft, train_label = extract_data(path_train_hand_craft, dict_encoding_train)
    test_ast, test_hand_craft, test_label = extract_data(path_test_hand_craft, dict_encoding_test)

    data = {'training': [], 'test': [], 'NN': [], 'Number of Filter': [], 'Filter Length': [],
             'Number of Hidden Nodes': [], 'epoch': [], 'embedding': [], 'ImbMethod': [], 'loop': [],
             'accuracy_mean': [], 'accuracy_std': [], 'precision_mean': [], 'precision_std': [], 'recall_mean': [],
             'recall_std': [], 'AUC_mean': [], 'AUC_std': [], 'F-measure_mean': [], 'F-measure_std': [],
             'MCC_mean': [], 'MCC_std': []}

    # 循环每一种参数
    for params in itertools.product(num_filter, len_filter, num_hidden_node, epoch, imbalance):
        dict_params = init_cnn_params.copy()
        dict_params['filters'] = params[0]
        dict_params['kernel_size'] = params[1]
        dict_params['hidden_units'] = params[2]
        dict_params['epochs'] = params[3]
        dict_params['imbalance'] = params[4];
        dict_params['input_dim'] = vocabulary_size + 1
        dict_params['input_length'] = vector_len

        insert_data(path[0], path[1], dict_params, loopLen, data)
        data['NN'].append('tcnn')
        acc_m, acc_s, pre_m, pre_s, rec_m, rec_s, auc_m, auc_s, f1_m, f1_s, mcc_m, mcc_s = train_and_test_model_tcnn(
            train_ast, train_hand_craft, train_label,
            test_ast, test_label, loopLen, dict_params)

        insert_score(acc_m, acc_s, pre_m, pre_s, rec_m, rec_s, auc_m, auc_s, f1_m, f1_s, mcc_m, mcc_s, data=data)
        df = pd.DataFrame(data=data, columns=['training', 'test', 'NN', 'Number of Filter', 'Filter Length',
                                                'Number of Hidden Nodes', 'epoch', 'embedding', 'ImbMethod', 'loop',
                                                'accuracy_mean', 'accuracy_std', 'precision_mean', 'precision_std',
                                                'recall_mean', 'recall_std', 'AUC_mean', 'AUC_std', 'F-measure_mean',
                                                'F-measure_std', 'MCC_mean', 'MCC_std'])
        df.to_csv('result/pair_' + path[0] + '_' + path[1] + '.csv', index=False)
