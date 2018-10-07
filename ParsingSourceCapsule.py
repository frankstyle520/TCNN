import tool.javalang as jl
import os
import tool.javalang.tree as jlt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
import pickle

types = [jlt.FormalParameter, jlt.BasicType, jlt.PackageDeclaration, jlt.InterfaceDeclaration, jlt.CatchClauseParameter,
         jlt.ClassDeclaration,
         jlt.MemberReference, jlt.SuperMemberReference, jlt.ConstructorDeclaration, jlt.ReferenceType,
         jlt.MethodDeclaration, jlt.VariableDeclarator, jlt.IfStatement, jlt.WhileStatement, jlt.DoStatement,
         jlt.ForStatement, jlt.AssertStatement, jlt.BreakStatement, jlt.ContinueStatement, jlt.ReturnStatement,
         jlt.ThrowStatement, jlt.SynchronizedStatement, jlt.TryStatement,
         jlt.SwitchStatement, jlt.BlockStatement, jlt.StatementExpression, jlt.TryResource, jlt.CatchClause,
         jlt.CatchClauseParameter, jlt.SwitchStatementCase, jlt.ForControl, jlt.EnhancedForControl]

features = ['wmc', 'dit', 'noc', 'cbo', 'rfc', 'lcom', 'ca', 'ce', 'npm', 'lcom3', 'loc', 'dam', 'moa', 'mfa', 'cam',
            'ic', 'cbm', 'amc', 'max_cc', 'avg_cc']

def round_self(array):
    result = list()
    for a in array:
        if a > 0.5:
            result.append(1)
        else:
            result.append(0)
    return result


def generate_dir_recursive(path):
    import os
    path = path.strip()
    path = path.rstrip('/')
    phases = path.split('/')
    path = ''
    for i in range(len(phases)):
        path = path + phases[i] + '/'
        if not os.path.exists(path):
            os.mkdir(path)


def dump_data(path, obj):
    with open(path, 'wb') as file_obj:
        pickle.dump(obj, file_obj)


def load_data(path):
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as file_obj:
        return pickle.load(file_obj)


def append_suffix(df):
    for i in range(len(df['file_name'])):
        df.loc[i, 'file_name'] = df.loc[i, 'file_name'] + ".java"
    return df


def extract_handcraft_instances(path):
    """
    输入csv文件的相对路径，返回csv文件中所有文件实例的列表
    :param train_path:  手工标注的训练集csv文件路径
    :param test_path:   手工标注的测试集csv文件路径
    :return:        [手工标注训练集的文件名],[手工标注的测试集文件名]
    """
    handcraft_instances = pd.read_csv(path)
    handcraft_instances = append_suffix(handcraft_instances)
    handcraft_instances = np.array(handcraft_instances['file_name'])
    handcraft_instances = handcraft_instances.tolist()

    return handcraft_instances


def ast_parse(source_file_path):
    """
    解析源码生成ast树，再转换成token向量(长度随ast树有区别)
    :param source_file_path:   源码文件的路径
    :return:    [token]
    """
    with open(source_file_path, 'rb') as file_obj:
        content = file_obj.read()
        result = []
        tree = []
        try:
            tree = jl.parse.parse(content)
        except jl.parser.JavaSyntaxError:
            print(source_file_path)
        for path, node in tree:
            if isinstance(node, jlt.MethodInvocation) or isinstance(node, jlt.SuperMethodInvocation):
                result.append(str(node.member) + "()")
                continue
            if isinstance(node, jlt.ClassCreator):
                result.append(str(node.type.name))
                continue
            if type(node) in types:
                result.append(str(node))
        return result


def parse_source(project_root_path, handcraft_file_names, package_heads):
    """

    :param project_root_path: 项目文件的目录地址
    :param handcraft_file_names:     手工标注数据的文件名
    :param package_heads:               包名的开头单词
    :return: 返回一个字典，{file_name:encoding token vector}
    """
    result = {}
    count = 0
    for dir_path, dir_names, file_names in os.walk(project_root_path):
        if len(file_names) == 0:
            continue
        index = -1
        for _head in package_heads:
            index = int(dir_path.find(_head))
            if index >= 0:
                break
        if index < 0:
            continue
        package_name = dir_path[index:]
        package_name = package_name.replace(os.sep, '.')
        for file in file_names:
            if file.endswith('java'):
                if str(package_name + "." + str(file)) not in handcraft_file_names:
                    continue
                result[package_name + "." + str(file)] = ast_parse(str(os.path.join(dir_path, file)))
                count += 1

    print("data size : " + str(count))
    return result


def padding_vector(vector, size):
    """
    用0使向量维度和指定值一样
    :param vector: padding前的向量
    :param size:  需要填充到的长度
    :return:        len==size的向量
    """
    if len(vector) == size:
        return vector
    padding = np.zeros((1, size - len(vector)))
    padding = list(np.squeeze(padding))
    vector += map(int, padding)
    return vector


def padding_all(dict_token, size):
    """

    :param dict_token:   {文件名:[实值向量]}
    :param size:    向量长度
    :return: 文件名:[长度一样的实值向量]}
    """
    result = {}
    for key, vector in dict_token.items():
        pv = padding_vector(vector, size)
        result[key] = pv
    return result


def max_length(d):
    max_len = 0
    for value in d.values():
        if max_len < len(value):
            max_len = len(value)
    return max_len


def transform_token_to_number(list_dict_token):
    """
    :param list_dict_token: [字典 {文件名:[ast节点的名称列表]}]
    :return:           [字典 {文件名:[用数字表示的ast向量(长度一致)]}],数字向量的长度，字典的长度
    """
    frequence = {}
    for _dict_token in list_dict_token:
        for _token_vector in _dict_token.values():
            for _token in _token_vector:
                if frequence.__contains__(_token):
                    frequence[_token] = frequence[_token] + 1
                else:
                    frequence[_token] = 1

    vocabulary = {}  # 用来学习每个token的数字表示的映射表    token:number
    result = []
    count = 0
    max_len = 0
    for dict_token in list_dict_token:
        _dict_encode = {}
        for file_name, token_vector in dict_token.items():
            vector = []
            for v in token_vector:
                if frequence[v] < 3:
                    continue

                if vocabulary.__contains__(v):
                    vector.append(vocabulary.get(v))
                else:
                    count = count + 1
                    vector.append(count)
                    vocabulary[v] = count
            if len(vector) > max_len:
                max_len = len(vector)
            _dict_encode[file_name] = vector
        result.append(_dict_encode)

    for i in range(len(result)):
        result[i] = padding_all(result[i], max_len)
    return result, max_len, len(vocabulary)


def extract_data(path_handcraft_file, dict_encoding_vector):
    """
    获取训练和测试用的数据
    :param path_handcraft_file:   手工标注的文件路径
    :param dict_encoding_vector:    向量长度对齐后的字典
    :return:        ast向量，手工特征向量，标签(bug or clean)
    """

    def extract_label(df, file_name):
        """
        提取这个文件是否有bug的标记
        :param df:  DataFrame
        :param file_name:
        :return:
        """

        row = df[df.file_name == file_name]['bug']
        row = np.array(row).tolist()
        if row[0] > 1:
            row[0] = 1
        return row

    def extract_feature(df, file_name):
        """
        提取手工标注的特征
        :param df:
        :param file_name:   csv的file_name字段对应
        :return:        [features]
        """
        row = df[df.file_name == file_name][features]
        row = np.array(row).tolist()
        row = np.squeeze(row)
        row = list(row)
        return row

    ast_x_data = []  # 用于模型输入的ast数值向量   [[....],[.....]...]
    hand_x_data = []  # 用于模型输入的手工标注向量 [[....],[.....]...]
    label_data = []  # 标签向量                    [[1],[0],....]
    raw_handcraft = pd.read_csv(path_handcraft_file)
    raw_handcraft = append_suffix(raw_handcraft)
    for key, value in dict_encoding_vector.items():
        ast_x_data.append(value)
        hand_x_data.append(extract_feature(raw_handcraft, key))
        label_data.append(extract_label(raw_handcraft, key))
    ast_x_data = np.array(ast_x_data)
    hand_x_data = np.array(hand_x_data)
    label_data = np.array(label_data)

    return ast_x_data, hand_x_data, label_data


def f1(y_true, y_predict):
    """
    用于计算fscore
    :param y_true:
    :param y_predict:
    :return:
    """

    def recall(yt, yp):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(yt * yp, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(yt, 0, 1)))
        _recall = true_positives / (possible_positives + K.epsilon())
        return _recall

    def precision(yt, yp):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(yt * yp, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(yp, 0, 1)))
        _precision = true_positives / (predicted_positives + K.epsilon())
        return _precision

    precision = precision(y_true, y_predict)
    recall = recall(y_true, y_predict)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def generate_weight(label):
    a = 0
    b = 0
    label = np.array(label)
    label = label.reshape(-1)
    for i in label:
        if i == 0:
            a = a + 1
        else:
            b = b + 1
    s = a + b
    a = a * 1.0 / s
    b = b * 1.0 / s
    result = []
    for i in label:
        if i == 0:
            result.append(b)
        else:
            result.append(a)
    return np.array(result)


def generate_weight_for_class(label):
    a = 0
    b = 0
    label = np.array(label)
    label = label.reshape(-1)
    for i in label:
        if i == 0:
            a = a + 1
        else:
            b = b + 1
    s = a + b
    a = a * 1.0 / s
    b = b * 1.0 / s
    result = [b, a]
    return np.array(result)


def imbalance_process(x_ast, x_handcraft, label, processor):
    """
    需要重写的方法，用于处理不平衡数据集
    :param x_ast:           np.array([[...],[...],[....]...])
    :param x_handcraft:    np.array([[...],[...],[....]...])
    :param label:           np.array([[0],[1],[1]...])
    :param processor:   不平衡处理的处理器
    :return:
    """
    if processor is None:
        return x_ast, x_handcraft, label

    _len_ast = len(x_ast[0])
    _len_label = len(label)

    _x = np.hstack((x_ast, x_handcraft))
    _y = label.reshape(_len_label)
    _x_resampled, _y_resampled = processor.fit_sample(_x, _y)
    _ast, _handcraft = np.split(_x_resampled, [_len_ast], axis=1)
    _label = _y_resampled.reshape((len(_y_resampled), 1))

    print("after imbalance process:" + str(len(_y_resampled)))
    _count = 0
    for i in _y_resampled:
        if i == 0:
            _count += 1
    print("0 count is:" + str(_count))

    return _ast, _handcraft, _label


def count_binary(label):
    _count = 0
    for i in range(len(label)):
        if label[i][0] == 0:
            _count += 1
    print("0:" + str(_count))
    print("1:" + str(len(label) - _count))


def split_batch(data_size, batch_size):
    start_indexes = []
    end_indexes = []

    if data_size < batch_size:
        start_indexes.append(0)
        end_indexes.append(data_size)
    else:
        end_index = 0
        for i, j in zip(range(0, data_size, batch_size), range(batch_size, data_size, batch_size)):
            start_indexes.append(i)
            end_indexes.append(j)
            end_index = j
        if end_index < data_size:
            start_indexes.append(end_index)
            end_indexes.append(data_size)
    return start_indexes, end_indexes


def classifier_logistic_regression(train_x, train_y, learning_rate=0.001, epoch_size=500, batch_size=1000):
    """
    :param train_x:
    :param train_y: [[1][0]....]
    :param learning_rate:
    :param epoch_size
    :param batch_size
    :return:
    """
    data_size = len(train_x)
    train_y = np.concatenate((train_y, 1 - train_y), axis=1)

    n_feature = train_x.shape[1]
    n_class = train_y.shape[1]

    x = tf.placeholder(dtype=tf.float32, shape=[None, n_feature])
    y = tf.placeholder(dtype=tf.float32, shape=[None, n_class])
    _w = tf.Variable(tf.truncated_normal(shape=[n_feature, n_class]))

    tf.summary.histogram("w", values=_w)
    m = tf.summary.merge_all()

    # _w = tf.Variable(tf.zeros(shape=[n_feature, n_class]))
    _b = tf.Variable(tf.zeros([n_class]))
    _predict = tf.sigmoid(tf.matmul(x, _w) + _b)
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=_predict, labels=y))
    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=_predict, labels=y))
    optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(cost)
    w = []
    b = []
    # threshold = tf.Variable(tf.fill(dims=[None, 1], value=0.5))
    # threshold = tf.constant(shape=y.shape, value=0.5, dtype=tf.float32)
    correct_prediction = tf.equal(tf.argmax(_predict, axis=1), tf.argmax(y, axis=1))
    accuracy_rate = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))
    start_indexes, end_indexes = split_batch(data_size, batch_size)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter('data/log', sess.graph)
        count = 0
        for epoch in range(epoch_size):
            for start, end in zip(start_indexes, end_indexes):
                _, c, summary = sess.run([optimizer, cost, m],
                                         feed_dict={x: train_x[start:end], y: train_y[start:end]})
                writer.add_summary(summary, count)
                count = count + 1
            print("epoch:" + str(epoch) + "===============>cost:" + str(c))

    return w, b