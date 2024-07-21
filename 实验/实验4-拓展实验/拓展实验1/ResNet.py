#!unzip data.zip

#通用模块
import numpy as np
import matplotlib.pyplot as plt
from easydict import EasyDict as edict#以类结构体方式访问字典

#MindSpore相关
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.dataset as ds
from mindspore import context,Tensor
from mindspore.train import Model
from mindspore.nn.optim.momentum import Momentum
from mindspore.dataset.vision import c_transforms as vision
from mindspore.train.callback import ModelCheckpoint,CheckpointConfig,LossMonitor 
from mindspore.train.serialization import export
from mindspore.train.loss_scale_manager import FixedLossScaleManager

#MindSpore的基本配置(运行模式和目标设备)
context.set_context(mode=context.GRAPH_MODE,device_target="Ascend")
#模型参数
cfg=edict({'data_path':'./data/flower_photos',#训练集
            'test_path':'./data/test',#验证集
            'data_size':3616,#数据规模
            'HEIGHT':224,#图像高度
            'WIDTH':224,#图像宽度
            '_R_MEAN':123.68,#RGB均值
            '_G_MEAN':116.78,
            '_B_MEAN':103.94,
            '_R_STD':1,#RGB标准差
            '_G_STD':1,
            '_B_STD':1,
            'num_class':5,#类别数
            'epoch_size':50,#迭代次数
            
            '_RESIZE_SIDE_MIN':256,
            '_RESIZE_SIDE_MAX':512,
            'batch_size':32,
            'loss_scale_num':1024,
            'prefix':'resnet-ai',
            'directory':'./model_resnet',
            'save_checkpoint_steps':10})

def read_data(path,cfg,usage='train'):#读取数据并进行预处理
    #读取指定路径下的数据集
    dataset=ds.ImageFolderDataset(path)

    #将图片转化为RGB色彩模式
    decode_op=vision.Decode()
    #随机翻转图像
    horizontal_flip_op=vision.RandomHorizontalFlip()
    #HWC2CHW算子
    channelswap_op=vision.HWC2CHW()
    #归一化
    normalize_op=vision.Normalize(mean=[cfg._R_MEAN,cfg._G_MEAN,cfg._B_MEAN],std=[cfg._R_STD,cfg._G_STD,cfg._B_STD])
    #重设大小
    resize_op=vision.Resize(cfg._RESIZE_SIDE_MIN)
    #中心化
    center_crop_op=vision.CenterCrop((cfg.HEIGHT,cfg.WIDTH))
    #裁剪和翻转
    random_crop_decode_resize_op=vision.RandomCropDecodeResize((cfg.HEIGHT,cfg.WIDTH),(0.5,1.0),(1.0,1.0),max_attempts=100)

    #更新设置
    if usage == 'train':#如果用于训练集
        dataset=dataset.map(input_columns="image",operations=random_crop_decode_resize_op)
        dataset=dataset.map(input_columns="image",operations=horizontal_flip_op)
    else:
        dataset=dataset.map(input_columns="image",operations=decode_op)
        dataset=dataset.map(input_columns="image",operations=resize_op)
        dataset=dataset.map(input_columns="image",operations=center_crop_op)

    dataset=dataset.map(input_columns="image",operations=normalize_op)
    dataset=dataset.map(input_columns="image",operations=channelswap_op)

    #随机打乱给定数据集并按指定大小分为若干批
    if usage == 'train':#如果用于训练集
        dataset=dataset.shuffle(buffer_size=10000)
        dataset=dataset.batch(cfg.batch_size,drop_remainder=True)
    else:
        dataset=dataset.batch(1,drop_remainder=True)
    dataset=dataset.repeat(1)
    dataset.map_model=4

    return dataset

#读取训练集和验证集
de_train=read_data(cfg.data_path,cfg,usage="train")
de_test=read_data(cfg.test_path,cfg,usage="test")
print('训练数据集数量:',de_train.get_dataset_size()*cfg.batch_size)
print('测试数据集数量:',de_test.get_dataset_size())

#转换为字典类型
de_dataset=de_train
data_next=de_dataset.create_dict_iterator(output_numpy=True).__next__()
print('通道数/图像长/宽:',data_next['image'][0,...].shape)
print('一张图像的标签样式:',data_next['label'][0])  # 一共5类,用0-4的数字表达类别。

#绘图
plt.figure()
plt.imshow(data_next['image'][0,0,...])
plt.colorbar()
plt.grid(False)
plt.show()

#随机化权重值
def _weight_variable(shape,factor=0.01):
    init_value=np.random.randn(*shape).astype(np.float32)*factor
    return Tensor(init_value)

#更新张量
def _fc(in_channel,out_channel):
    weight_shape=(out_channel,in_channel)
    weight=_weight_variable(weight_shape)
    return nn.Dense(in_channel,out_channel,has_bias=True,weight_init=weight,bias_init=0)

#批归一化
def _bn(channel):
    return nn.BatchNorm2d(channel,eps=1e-4,momentum=0.9,
                          gamma_init=1,beta_init=0,moving_mean_init=0,moving_var_init=1)
def _bn_last(channel):#结果仅包含β
    return nn.BatchNorm2d(channel,eps=1e-4,momentum=0.9,
                          gamma_init=0,beta_init=0,moving_mean_init=0,moving_var_init=1)

#不同卷积核的二维卷积
def _conv1x1(in_channel,out_channel,stride=1):
    weight_shape=(out_channel,in_channel,1,1)
    weight=_weight_variable(weight_shape)
    return nn.Conv2d(in_channel,out_channel,
                     kernel_size=1,stride=stride,padding=0,pad_mode='same',weight_init=weight)
def _conv3x3(in_channel,out_channel,stride=1):
    weight_shape=(out_channel,in_channel,3,3)
    weight=_weight_variable(weight_shape)
    return nn.Conv2d(in_channel,out_channel,
                     kernel_size=3,stride=stride,padding=0,pad_mode='same',weight_init=weight)
def _conv7x7(in_channel,out_channel,stride=1):
    weight_shape=(out_channel,in_channel,7,7)
    weight=_weight_variable(weight_shape)
    return nn.Conv2d(in_channel,out_channel,
                     kernel_size=7,stride=stride,padding=0,pad_mode='same',weight_init=weight)

class ResidualBlock(nn.Cell):
    """
    ResNet V1 residual block definition.

    Args:
        in_channel (int): Input channel.
        out_channel (int): Output channel.
        stride (int): Stride size for the first convolutional layer. Default: 1.

    Returns:
        Tensor,output tensor.

    Examples:
        >>> ResidualBlock(3,256,stride=2)
    """
    expansion=4

    def __init__(self,
                 in_channel,
                 out_channel,
                 stride=1):
        super(ResidualBlock,self).__init__()

        channel=out_channel // self.expansion
        self.conv1=_conv1x1(in_channel,channel,stride=1)
        self.bn1=_bn(channel)

        self.conv2=_conv3x3(channel,channel,stride=stride)
        self.bn2=_bn(channel)

        self.conv3=_conv1x1(channel,out_channel,stride=1)
        self.bn3=_bn_last(out_channel)

        self.relu=nn.ReLU()

        self.down_sample=False

        if stride != 1 or in_channel != out_channel:
            self.down_sample=True
        self.down_sample_layer=None

        if self.down_sample:
            self.down_sample_layer=nn.SequentialCell([_conv1x1(in_channel,out_channel,stride),
                                                        _bn(out_channel)])
        self.add=ops.Add()
    def construct(self,x): # pylint: disable=missing-docstring
        identity=x

        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)

        out=self.conv2(out)
        out=self.bn2(out)
        out=self.relu(out)

        out=self.conv3(out)
        out=self.bn3(out)

        if self.down_sample:
            identity=self.down_sample_layer(identity)

        out=self.add(out,identity)
        out=self.relu(out)

        return out

class ResNet(nn.Cell):
    """
    ResNet architecture.

    Args:
        block (Cell): Block for network.
        layer_nums (list): Numbers of block in different layers.
        in_channels (list): Input channel in each layer.
        out_channels (list): Output channel in each layer.
        strides (list):  Stride size in each layer.
        num_classes (int): The number of classes that the training images are belonging to.
    Returns:
        Tensor,output tensor.

    Examples:
        >>> ResNet(ResidualBlock,
        >>>        [3,4,6,3],
        >>>        [64,256,512,1024],
        >>>        [256,512,1024,2048],
        >>>        [1,2,2,2],
        >>>        10)
    """

    def __init__(self,
                 block,
                 layer_nums,
                 in_channels,
                 out_channels,
                 strides,
                 num_classes):
        super(ResNet,self).__init__()

        if not len(layer_nums) == len(in_channels) == len(out_channels) == 4:
            raise ValueError("the length of layer_num,in_channels,out_channels list must be 4!")

        self.conv1=_conv7x7(3,64,stride=2)
        self.bn1=_bn(64)
        self.relu=ops.ReLU()
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,pad_mode="same")

        self.layer1=self._make_layer(block,
                                       layer_nums[0],
                                       in_channel=in_channels[0],
                                       out_channel=out_channels[0],
                                       stride=strides[0])
        self.layer2=self._make_layer(block,
                                       layer_nums[1],
                                       in_channel=in_channels[1],
                                       out_channel=out_channels[1],
                                       stride=strides[1])
        self.layer3=self._make_layer(block,
                                       layer_nums[2],
                                       in_channel=in_channels[2],
                                       out_channel=out_channels[2],
                                       stride=strides[2])
        self.layer4=self._make_layer(block,
                                       layer_nums[3],
                                       in_channel=in_channels[3],
                                       out_channel=out_channels[3],
                                       stride=strides[3])

        self.mean=ops.ReduceMean(keep_dims=True)
        self.flatten=nn.Flatten()
        self.end_point=_fc(out_channels[3],num_classes)

    def _make_layer(self,block,layer_num,in_channel,out_channel,stride):
        """
        Make stage network of ResNet.

        Args:
            block (Cell): Resnet block.
            layer_num (int): Layer number.
            in_channel (int): Input channel.
            out_channel (int): Output channel.
            stride (int): Stride size for the first convolutional layer.

        Returns:
            SequentialCell,the output layer.

        Examples:
            >>> _make_layer(ResidualBlock,3,128,256,2)
        """
        layers=[]

        resnet_block=block(in_channel,out_channel,stride=stride)
        layers.append(resnet_block)

        for _ in range(1,layer_num):
            resnet_block=block(out_channel,out_channel,stride=1)
            layers.append(resnet_block)

        return nn.SequentialCell(layers)
    def construct(self,x): # pylint: disable=missing-docstring
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        c1=self.maxpool(x)

        c2=self.layer1(c1)
        c3=self.layer2(c2)
        c4=self.layer3(c3)
        c5=self.layer4(c4)

        out=self.mean(c5,(2,3))
        out=self.flatten(out)
        out=self.end_point(out)

        return out

def resnet50(class_num=10):
    """
    Get ResNet50 neural network.

    Args:
        class_num (int): Class number.

    Returns:
        Cell,cell instance of ResNet50 neural network.

    Examples:
        >>> net=resnet50(10)
    """
    return ResNet(ResidualBlock,
                  [3,4,6,3],
                  [64,256,512,1024],
                  [256,512,1024,2048],
                  [1,2,2,2],
                  class_num)

def get_lr(global_step,
           total_epochs,
           steps_per_epoch,
           lr_init=0.01,
           lr_max=0.1,
           warmup_epochs=5):
    """
    Generate learning rate array.

    Args:
        global_step (int): Initial step of training.
        total_epochs (int): Total epoch of training.
        steps_per_epoch (float): Steps of one epoch.
        lr_init (float): Initial learning rate. Default: 0.01.
        lr_max (float): Maximum learning rate. Default: 0.1.
        warmup_epochs (int): The number of warming up epochs. Default: 5.

    Returns:
        np.array,learning rate array.
    """
    lr_each_step=[]
    total_steps=steps_per_epoch*total_epochs
    warmup_steps=steps_per_epoch*warmup_epochs
    if warmup_steps != 0:
        inc_each_step=(float(lr_max)-float(lr_init))/float(warmup_steps)
    else:
        inc_each_step=0
    for i in range(int(total_steps)):
        if i < warmup_steps:
            lr=float(lr_init)+inc_each_step*float(i)
        else:
            base=(1.0-(float(i)-float(warmup_steps))/(float(total_steps)-float(warmup_steps)))
            lr=float(lr_max)*base*base
            if lr < 0.0:
                lr=0.0
        lr_each_step.append(lr)

    current_step=global_step
    lr_each_step=np.array(lr_each_step).astype(np.float32)
    learning_rate=lr_each_step[current_step:]

    return learning_rate

#初始化残差网络
net=resnet50(class_num=cfg.num_class)
#计算损失函数。
loss=nn.SoftmaxCrossEntropyWithLogits(sparse=True,reduction="mean")
#配置Adam优化器
train_step_size=de_train.get_dataset_size()
#初始化学习率张量作为MindSpore网络的基本计算单元
lr=Tensor(get_lr(global_step=0,total_epochs=cfg.epoch_size,steps_per_epoch=train_step_size))
#预设各项参数:学习率,动量,权值衰减和梯度放缩
opt=Momentum(net.trainable_params(),lr,momentum=0.9,weight_decay=1e-4,loss_scale=cfg.loss_scale_num)
#梯度放大系数,在溢出时仍然执行优化器
loss_scale=FixedLossScaleManager(cfg.loss_scale_num,False)
#评估训练结果
model=Model(net,loss_fn=loss,optimizer=opt,loss_scale_manager=loss_scale,metrics={'acc'})
#保留检查点的配置并保留模型的各项参数
loss_cb=LossMonitor(per_print_times=train_step_size)
ckpt_config=CheckpointConfig(save_checkpoint_steps=cfg.save_checkpoint_steps,keep_checkpoint_max=1)
ckpoint_cb=ModelCheckpoint(prefix=cfg.prefix,directory=cfg.directory,config=ckpt_config)



print("============== Starting Training ==============")
#开始训练,参数依次为迭代次数,数据集,返回值和操作的处理器下沉
model.train(cfg.epoch_size,de_train,callbacks=[loss_cb,ckpoint_cb],dataset_sink_mode=True)

#用测试集评估训练结果
metric=model.eval(de_test)
print(metric)

#根据训练结果进行预测
class_names={0:'daisy',1:'dandelion',2:'roses',3:'sunflowers',4:'tulips'}
for i in range(10):
    test_=de_test.create_dict_iterator().__next__()
    test=Tensor(test_['image'],mindspore.float32)
    predictions=model.predict(test)
    predictions=predictions.asnumpy()
    true_label=test_['label'].asnumpy()
    p_np=predictions[0,:]
    pre_label=np.argmax(p_np)
    print('第'+str(i)+'个sample预测结果:',class_names[pre_label],'   真实结果:',class_names[true_label[0]])

print("============== Mission  Completed ==============")