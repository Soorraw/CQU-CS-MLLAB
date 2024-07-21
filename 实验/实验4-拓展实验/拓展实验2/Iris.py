import csv,os
import numpy as np
from easydict import EasyDict as edict
from matplotlib import pyplot as plt

import mindspore
from mindspore import nn,context,dataset,Tensor
from mindspore.train import Model
from mindspore.train.callback import TimeMonitor, LossMonitor
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")  # 设定运行模式为静态图模式,并且运行设备为昇腾芯片

#变量定义
cfg = edict({
    'data_size': 150,
    'train_size': 120,      #训练集大小
    'test_size': 30 ,       #测试集大小
    'feature_number': 4,       #输入特征数
    'num_class': 3,     #分类类别
    'batch_size': 30,   #批次大小
    'data_dir':    'iris.data',     # 数据集路径           
    'save_checkpoint_steps': 5,                 #多少步保存一次模型
    'keep_checkpoint_max': 1,                      #最多保存多少个模型
    'out_dir_no_opt':   './model_iris/no_opt',          #保存模型路径，无优化器模型
    'out_dir_sgd':   './model_iris/sgd',          #保存模型路径,SGD优化器模型
    'out_dir_momentum':   './model_iris/momentum',          #保存模型路径，momentum模型
    'out_dir_adam':   './model_iris/adam',          #保存模型路径，adam优化器模型
    'output_prefix': "checkpoint_fashion_forward"     #保存模型文件名
})

#鸢尾花数据集
with open(cfg.data_dir) as csv_file:data = list(csv.reader(csv_file, delimiter=','))

label_map = {'setosa': 0,'versicolor': 1,'virginica':2 }
#特征值和标签值
X = np.array([[float(x) for x in s[:-1]] for s in data[:cfg.data_size]], np.float32)
Y = np.array([label_map[s[-1]] for s in data[:cfg.data_size]], np.int32)

#将数据集分割为训练集和测试集
train_idx = np.random.choice(cfg.data_size, cfg.train_size, replace=False)
test_idx = np.array(list(set(range(cfg.data_size)) - set(train_idx)))
X_train, Y_train = X[train_idx], Y[train_idx]
X_test, Y_test = X[test_idx], Y[test_idx]

def gen_data(X_train, Y_train):#预处理
    XY_train = list(zip(X_train, Y_train))
    ds_train = dataset.GeneratorDataset(XY_train, ['x', 'y'])
    #随机化数据集并设置批规模
    ds_train = ds_train.shuffle(buffer_size=cfg.train_size).batch(cfg.batch_size, drop_remainder=True)
    XY_test = list(zip(X_test, Y_test))
    ds_test = dataset.GeneratorDataset(XY_test, ['x', 'y'])
    ds_test = ds_test.shuffle(buffer_size=cfg.test_size).batch(cfg.test_size, drop_remainder=True)
    return ds_train, ds_test

def train(network, net_opt, ds_train, prefix, directory, print_times):
    #定义网络损失函数
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")  # sparse为True时对Label数据做one_hot处理,reduction支持mean和sum
    #定义模型
    model = Model(network, loss_fn=net_loss, optimizer=net_opt, metrics={"acc"}) # 定义网络结构,损失函数,优化器,评估方式
    #定义损失值指标
    loss_cb = LossMonitor(per_print_times=print_times)   # 每隔 print_times 步监测一下损失值

    #设置checkpoint
    config_ck = CheckpointConfig(save_checkpoint_steps=cfg.save_checkpoint_steps,    # 每5步保存一下模型
                                 keep_checkpoint_max=cfg.keep_checkpoint_max)        # 最多保存1个模型
    ckpoint_cb = ModelCheckpoint(prefix=prefix, directory=directory, config=config_ck)  # 设置文件名,文件路径,以及checkpoint参数
    print("============== Starting Training ==============")
    #训练模型
    model.train(100, ds_train, callbacks=[ckpoint_cb, loss_cb]) # 设置训练次数,训练数据,回调函数（checkpoint和lossmonitor）,Ascend是否采用下沉模式
    return model

class_names=['setosa', 'versicolor', 'virginica']
# 评估预测函数
def eval_predict(model, ds_test):
    # 使用测试集评估模型,打印总体准确率
    metric = model.eval(ds_test)
    print(metric)
    # 预测
    test_ = ds_test.create_dict_iterator().__next__()  # 生成测试集
    test = Tensor(test_['x'], mindspore.float32)  # 将测试集的特征转换成mindspore数据类型
    predictions = model.predict(test)  # 用predict进行预测
    predictions = predictions.asnumpy()  # 将预测值转换成numpy数组类型, predictions.shape为(30, 3)
    true_label = test_['y'].asnumpy()  # 将真实值转换成numpy数组类型
    for i in range(30):
        p_np = predictions[i, :]  # 取第i个数据的预测值
        pre_label = np.argmax(p_np)  # 取最大值的索引作为输出标签
        print('第' + str(i) + '个sample预测结果:', class_names[pre_label], '   真实结果:', class_names[true_label[i]])  # 输出预测值和真实值的对比结果

# --------------------------------------------------无优化器-----------------------------------
print('------------------无优化器--------------------------')
# 数据
ds_train, ds_test = gen_data(X_train, Y_train)  # 生成训练集和测试集
# 定义网络并训练
network = nn.Dense(cfg.feature_number, cfg.num_class)  # 定义一个全连接网络层,输入特征为4,输出类别为3
model = train(network, None, ds_train, "checkpoint_no_opt", cfg.out_dir_no_opt, print_times=40)  # 用训练集训练网络,设置网络结构,模型名称,保存路径, print_times
# 评估预测
eval_predict(model, ds_test)  # 用测试集进行预测

import os
os.listdir('.')
os.listdir('./model_iris/no_opt')  # 查看保存的模型

# ---------------------------------------------------SGD-------------------------------------
lr = 0.01
print('-------------------SGD优化器-----------------------')
# 数据
ds_train, ds_test = gen_data(X_train, Y_train) # 生成训练集和测试集
# 定义网络并训练、测试、预测
network = nn.Dense(cfg.feature_number, cfg.num_class)  # 定义一个全连接网络层,输入特征为4,输出类别为3
net_opt = nn.SGD(network.trainable_params(), lr)  # 用SGD优化器进行优化 
model = train(network, net_opt, ds_train, "checkpoint_sgd", cfg.out_dir_sgd, 40)  # 用训练集训练网络,设置网络结构,优化器,模型名称,保存路径, print_times
# 评估预测
eval_predict(model, ds_test) # 用测试集进行预测     

os.listdir('./model_iris/sgd') # 查看保存的模型

# ----------------------------------------------------Momentum-------------------------------
lr = 0.01  # 学习率为0.01
print('-------------------Momentum优化器-----------------------')
# 数据
ds_train, ds_test = gen_data(X_train, Y_train)  # 生成训练集和测试集
# 定义网络并训练
network = nn.Dense(cfg.feature_number, cfg.num_class) # 定义一个全连接网络层,输入特征为4,输出类别为3
net_opt = nn.Momentum(network.trainable_params(), lr, 0.9)   # 用 momentum 优化器进行优化,学习率为0.01,动量大小为0.9
model = train(network, net_opt, ds_train, "checkpoint_momentum", cfg.out_dir_momentum, 40)  # 用训练集训练网络,设置网络结构,优化器,模型名称,保存路径, print_times
# 评估预测
eval_predict(model, ds_test)  # 用测试集进行预测  

os.listdir('./model_iris/momentum') # 查看保存的模型


# ----------------------------------------------------AdaGrad-----------------------------
lr = 0.01  # 学习率为0.01
print('-------------------AdaGrad优化器-----------------------')
# 数据
ds_train, ds_test = gen_data(X_train, Y_train)  # 生成训练集和测试集
# 定义网络并训练
network = nn.Dense(cfg.feature_number, cfg.num_class) # 定义一个全连接网络层,输入特征为4,输出类别为3
net_opt = nn.Adagrad(network.trainable_params(), 0.1, lr)   # 用 adagrad 优化器进行优化,累计增量初值为0.1
model = train(network, net_opt, ds_train, "checkpoint_momentum", cfg.out_dir_momentum, 40)  # 用训练集训练网络,设置网络结构,优化器,模型名称,保存路径, print_times
# 评估预测
eval_predict(model, ds_test)  # 用测试集进行预测  

os.listdir('./model_iris/momentum') # 查看保存的模型


# ----------------------------------------------------RMSProp-----------------------------
lr = 0.01  # 学习率为0.01
print('-------------------RMSProp优化器-----------------------')
# 数据
ds_train, ds_test = gen_data(X_train, Y_train)  # 生成训练集和测试集
# 定义网络并训练
network = nn.Dense(cfg.feature_number, cfg.num_class) # 定义一个全连接网络层,输入特征为4,输出类别为3
net_opt = nn.RMSProp(network.trainable_params(), lr)   # 用 RMSProp 优化器进行优化,累计增量初值为0.1
model = train(network, net_opt, ds_train, "checkpoint_momentum", cfg.out_dir_momentum, 40)  # 用训练集训练网络,设置网络结构,优化器,模型名称,保存路径, print_times
# 评估预测
eval_predict(model, ds_test)  # 用测试集进行预测  

os.listdir('./model_iris/momentum') # 查看保存的模型



# ----------------------------------------------------Adam-----------------------------------
lr = 0.1  # 学习率为0.1, 动态学习率
print('------------------Adam优化器--------------------------')
# 数据
ds_train, ds_test = gen_data(X_train, Y_train)  # 生成训练集和测试集
# 定义网络并训练
network = nn.Dense(cfg.feature_number, cfg.num_class)  # 定义一个全连接网络层,输入特征为4,输出类别为3
net_opt = nn.Adam(network.trainable_params(), learning_rate=lr)  # 用 Adam 优化器进行优化,学习率为0.1
model = train(network, net_opt, ds_train, "checkpoint_adam", cfg.out_dir_adam, 40)  # 用训练集训练网络,设置网络结构,优化器,模型名称,保存路径, print_times
# 评估预测
eval_predict(model, ds_test)

os.listdir('./model_iris/adam') # 查看保存的模型

