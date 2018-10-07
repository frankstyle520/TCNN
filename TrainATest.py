import torch.optim as optim
from sklearn.metrics import matthews_corrcoef, roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from ParsingSourceCapsule import *
import mmd
import torch
import TCNN
import CNN
import torch.nn as nn


def TCNN_train_and_test(loop_size, cnn_params, mmd_params, loader, test_ast, test_label, device):
    # 循环次数
    acc, auc, f1, mcc = [], [], [], []
    for l in range(loop_size):

        # 模型
        model = TCNN.TCNN(cnn_params)
        model.to(device)

        # 优化器用Momentum
        optimizer = optim.SGD(model.parameters(), lr=cnn_params['LEARNING_RATE'], momentum=cnn_params['MOMEMTUN'],
                              weight_decay=cnn_params['L2_WEIGHT'])

        # 训练
        for epoch in range(cnn_params['N_EPOCH']):
            print('epoch: ' + str(epoch))
            for step, (batch_x, batch_y) in enumerate(loader):
                model.train()
                y_src, x_src_mmd, x_tar_mmd = model(batch_x, test_ast)

                # 代价敏感损失函数
                weight = generate_weight_for_class(batch_y)
                weight = torch.cuda.FloatTensor(weight)
                criterion = nn.CrossEntropyLoss(weight=weight)

                loss_c = criterion(y_src, batch_y.squeeze())
                loss_mmd = mmd.mmd_loss(x_src_mmd, x_tar_mmd, mmd_params['MMD_GAMMA'])
                loss = loss_c + mmd_params['MMD_LAMBDA'] * loss_mmd
                print('loss_c: ' + str(loss_c))
                print('loss_mmd: ' + str(loss_mmd))
                print('loss: ' + str(loss))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # 预测
        test_output, _, _ = model(test_ast, test_ast)
        pred_y = test_output.data.max(1)[1]

        # 记录评估结果
        acc.append(accuracy_score(y_true=test_label, y_pred=pred_y))
        auc.append(roc_auc_score(y_true=test_label, y_score=pred_y))
        f1.append(f1_score(y_true=test_label, y_pred=pred_y))
        mcc.append(matthews_corrcoef(y_true=test_label, y_pred=pred_y))

    # 计算循环后的平均值和标准差
    acc, auc, f1, mcc = np.array(acc), np.array(auc), np.array(f1), np.array(mcc)
    acc_m, acc_s, auc_m, auc_s, f1_m, f1_s, mcc_m, mcc_s = acc.mean(), acc.std(), auc.mean(), auc.std(), f1.mean(), f1.std(), mcc.mean(), mcc.std()
    return acc_m, acc_s, auc_m, auc_s, f1_m, f1_s, mcc_m, mcc_s

def CNN_train_and_test(loop_size, cnn_params, loader, test_ast, test_label, device):
    # 循环次数
    acc, auc, f1, mcc = [], [], [], []
    for l in range(loop_size):

        # 模型
        model = CNN.CNN(cnn_params)
        model.to(device)

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
    return acc_m, acc_s, auc_m, auc_s, f1_m, f1_s, mcc_m, mcc_s