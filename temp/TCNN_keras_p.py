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
    model.add(Conv1D(filters=dict_params['filters'], kernel_size=dict_params['kernel_size'], activation='relu'))
    model.add(MaxPooling1D(pool_size=dict_params['pool_size'], padding='valid'))
    model.add(Flatten())
    model.add(Dense(units=dict_params['hidden_units'], activation='relu'))

    print(len(model.layers))
    get_7th_layer_output = K.function([model.layers[0].input],
                                      [model.layers[6].output])
    # 得到source的x_src_mmd
    x_src_mmd = get_7th_layer_output([x_src])[0]
    # 得到target的x_tar_mmd
    x_tar_mmd = get_7th_layer_output([x_tar])[0]

    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(loss=None, optimizer='rmsprop', metrics=dict_params['metrics'])
    return model


def mmd_loss(x_src, x_tar):
    x_src_mmd = torch.from_numpy(x_src)
    x_tar_mmd = torch.from_numpy(x_tar)
    return mmd.mix_rbf_mmd2(x_src_mmd, x_tar_mmd, [1])


class CustomVariationalLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)

    def vae_loss(self, x, x_decoded_mean_squash):
        x = K.flatten(x)
        x_decoded_mean_squash = K.flatten(x_decoded_mean_squash)
        xent_loss = img_rows * img_cols * metrics.binary_crossentropy(x, x_decoded_mean_squash)
        kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean_squash = inputs[1]
        loss = self.vae_loss(x, x_decoded_mean_squash)
        self.add_loss(loss, inputs=inputs)
        # We don't use this output.
        return x


