from ParsingSourceCapsule import *
from keras.layers import Dense, Flatten, Embedding, MaxPooling1D
from keras.layers.convolutional import Conv1D
from keras import Sequential
import mmd
import torch


def get_tcnn_model(dict_params, x_src, x_tar, mmd_lambda):
    """
    :param dict_params: 模型参数字典
    :return:    编译好的模型
    """
    model = Sequential()
    model.add(Embedding(input_dim=dict_params['input_dim'], output_dim=dict_params['output_dim'],
                            input_length=dict_params['input_length']))
    model.add(Conv1D(filters=dict_params['filters'], kernel_size=dict_params['kernel_size'], activation='relu'))
    model.add(MaxPooling1D(pool_size=dict_params['pool_size'], padding='valid'))
    model.add(Flatten())
    model.add(Dense(units=dict_params['hidden_units'], activation='relu'))

    print(len(model.layers))

    get_5th_layer_output = K.function([model.layers[0].input],
                                      [model.layers[4].output])
    # 得到source的x_src_mmd
    x_src_mmd = get_5th_layer_output([x_src])[0]
    # 得到target的x_tar_mmd
    x_tar_mmd = get_5th_layer_output([x_tar])[0]

    def tcnn_loss(y_true, y_pred):
        loss_c = K.binary_crossentropy(y_true, y_pred)
        loss_mmd = mmd_loss(x_src_mmd, x_tar_mmd)
        loss_mmd = loss_mmd.numpy()

        print(loss_c)
        print(mmd_lambda)
        print(loss_mmd)

        loss = loss_c + mmd_lambda * loss_mmd
        return loss

    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(loss=tcnn_loss, optimizer='rmsprop', metrics=dict_params['metrics'])
    return model


def mmd_loss(x_src, x_tar):
    x_src_mmd = torch.from_numpy(x_src)
    x_tar_mmd = torch.from_numpy(x_tar)
    return mmd.mix_rbf_mmd2(x_src_mmd, x_tar_mmd, [1])



