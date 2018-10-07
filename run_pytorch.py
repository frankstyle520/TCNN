from TCNN import get_tcnn_model
from ParsingSourceCapsule import *
from tool.imblearn.under_sampling import RandomUnderSampler
from tool.imblearn.over_sampling import RandomOverSampler
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import matthews_corrcoef, roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
import mmd

# 设置参数
init_cnn_params = {'input_dim': 3709, 'output_dim': 30, 'input_length': 2405, 'filters': 50, 'kernel_size': 20,
              'pool_size': 2, 'hidden_units': 150, 'hand_craft_input_dim': 20, 'metrics': ['acc'],
              'batch_size': 32, 'epochs': 15, 'imbalance': 'none', 'regenerate': True}

REGENERATE = True
MMD_LAMBDA = 1
IMBALANCE = 'none'
N_FILTER = 10
N_HIDDEN_NODE = 100
FILTER_SIZE = 5
LOOP_SIZE = 1
LAMBDA = 1
GAMMA = 10 ^ 3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEARNING_RATE = 0.02
MOMEMTUN = 0.05
L2_WEIGHT = 0.003
DROPOUT = 0.5
N_EPOCH = 900
BATCH_SIZE = 15

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


def mmd_loss(x_src, x_tar):
    return mmd.mix_rbf_mmd2(x_src, x_tar, [GAMMA])


def train(model, optimizer, epoch, data_src, data_tar):
    total_loss_train = 0
    criterion = nn.CrossEntropyLoss()
    correct = 0
    batch_j = 0
    list_src, list_tar = list(enumerate(data_src)), list(enumerate(data_tar))
    for batch_id, (data, target) in enumerate(data_src):
        _, (x_tar, y_target) = list_tar[batch_j]
        data, target = data.data.view(-1, 28 * 28).to(DEVICE), target.to(DEVICE)
        x_tar, y_target = x_tar.view(-1, 28 * 28).to(DEVICE), y_target.to(DEVICE)
        model.train()
        y_src, x_src_mmd, x_tar_mmd = model(data, x_tar)

        loss_c = criterion(y_src, target)
        loss_mmd = mmd_loss(x_src_mmd, x_tar_mmd)
        pred = y_src.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        loss = loss_c + LAMBDA * loss_mmd
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss_train += loss.data
        res_i = 'Epoch: [{}/{}], Batch: [{}/{}], loss: {:.6f}'.format(
            epoch, N_EPOCH, batch_id + 1, len(data_src), loss.data
        )
    total_loss_train /= len(data_src)
    acc = correct * 100. / len(data_src.dataset)
    res_e = 'Epoch: [{}/{}], training loss: {:.6f}, correct: [{}/{}], training accuracy: {:.4f}%'.format(
        epoch, N_EPOCH, total_loss_train, correct, len(data_src.dataset), acc
    )
    tqdm.write(res_e)
    log_train.write(res_e + '\n')
    RESULT_TRAIN.append([epoch, total_loss_train, acc])
    return model


def test(model, data_tar, e):
    total_loss_test = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_id, (data, target) in enumerate(data_tar):
            data, target = data.view(-1,28 * 28).to(DEVICE),target.to(DEVICE)
            model.eval()
            ypred, _, _ = model(data, data)
            loss = criterion(ypred, target)
            pred = ypred.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            total_loss_test += loss.data
        accuracy = correct * 100. / len(data_tar.dataset)
        res = 'Test: total loss: {:.6f}, correct: [{}/{}], testing accuracy: {:.4f}%'.format(
            total_loss_test, correct, len(data_tar.dataset), accuracy
        )
    tqdm.write(res)
    RESULT_TEST.append([e, total_loss_test, accuracy])
    log_test.write(res + '\n')


if __name__ == '__main__':
    rootdir = '../../../data/office_caltech_10/'
    torch.manual_seed(1)
    data_src = data_loader.load_data(
        root_dir=rootdir, domain='amazon', batch_size=BATCH_SIZE[0])
    data_tar = data_loader.load_test(
        root_dir=rootdir, domain='webcam', batch_size=BATCH_SIZE[1])
    model = DaNN.DaNN(n_input=28 * 28, n_hidden=256, n_class=10)
    model = model.to(DEVICE)
    optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMEMTUN,
        weight_decay=L2_WEIGHT
    )
    for e in tqdm(range(1, N_EPOCH + 1)):
        model = train(model=model, optimizer=optimizer,
                      epoch=e, data_src=data_src, data_tar=data_tar)
        test(model, data_tar, e)
    torch.save(model, 'model_dann.pkl')
    log_train.close()
    log_test.close()
    res_train = np.asarray(RESULT_TRAIN)
    res_test = np.asarray(RESULT_TEST)
    np.savetxt('res_train_a-w.csv', res_train, fmt='%.6f', delimiter=',')
    np.savetxt('res_test_a-w.csv', res_test, fmt='%.6f', delimiter=',')


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
