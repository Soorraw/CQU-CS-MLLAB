import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.colors as clr  # Matplotlib的色阶条
# ------------------定义目标函数beale、目标函数的偏导函数dbeale_dx，并画出目标函数---------------------
#定义beale公式
def beale(x1,x2):
    return (1.5-x1+x1*x2)**2+(2.25-x1+x1*x2**2)**2+(2.625-x1+x1*x2**3)**2
#定义beale公式的偏导函数
def dbeale_dx(x1, x2):
    dfdx1 = 2*(1.5-x1+x1*x2)*(x2-1)+2*(2.25-x1+x1*x2**2)*(x2**2-1)+2*(2.625-x1+x1*x2**3)*(x2**3-1) # 求beale公式关于x1的偏导数
    dfdx2 = 2*(1.5-x1+x1*x2)*x1+2*(2.25-x1+x1*x2**2)*(2*x1*x2)+2*(2.625-x1+x1*x2**3)*(3*x1*x2**2) # 求beale公式关于x2的偏导数
    return dfdx1, dfdx2


# 定义画图函数
def gd_plot(x_traj, ty):
    plt.rcParams['figure.figsize'] = [6, 6] # 窗口大小
    plt.contour(X1, X2, Y, levels=np.logspace(0, 6, 30),
                norm=clr.LogNorm(), cmap=plt.cm.jet)  # 画等高线图
    plt.title('2D Contour Plot of Beale function(%s)' % ty) # 添加标题
    plt.xlabel('$x_1$') # x轴标签
    plt.ylabel('$x_2$') # y轴标签
    plt.axis('equal') # 设置坐标轴为正方形
    plt.plot(3, 0.5, 'k*', markersize=10) # 画出最低点
    if x_traj is not None:
        x_traj = np.array(x_traj) # 将x_traj转为数组
        plt.plot(x_traj[:, 0], x_traj[:, 1], 'k-') 
# 以x_traj的第一列为x轴坐标，第二列为y轴坐标进行画图
    plt.show() # 显示图像

step_x1, step_x2 = 0.2, 0.2
X1, X2 = np.meshgrid(np.arange(-5, 5 + step_x1, step_x1),
                     np.arange(-5, 5 + step_x2, step_x2))  # 将图形从-5 到 5.2，步长为0.2 划分成网格点
Y = beale(X1, X2) # 将x1,x2坐标带入beale公式
print("目标结果 (x_1, x_2) = (3, 0.5)")
# gd_plot(None, "target") # 调用函数

#各优化器
def Without(init,epoch=500):
    return [init for i in range(epoch)]

def SGD(df,init,epoch=500,lr=0.01):
    x_traj=[init]
    for i in range(epoch):
        grad=np.array(df(x_traj[-1][0], x_traj[-1][1]))
        w=x_traj[-1]-lr*grad
        while(w[0]<-5 or w[0]>5 or w[1]<-5 or w[1]>5):
            grad/=2
            w+=lr*grad
        x_traj.append(w)
    return x_traj

def Momentum(df,init,epoch=500,lr=0.01,gamma=0.9):
    x_traj=[init]
    v=np.zeros_like(init)
    for i in range(epoch):
        grad=np.array(df(x_traj[-1][0], x_traj[-1][1]))
        v=gamma*v-lr*grad
        w=x_traj[-1]+v
        while(w[0]<-5 or w[0]>5 or w[1]<-5 or w[1]>5):
            v/=2
            w-=v
        x_traj.append(w)
    return x_traj

def AdaGrad(df,init,epoch=500,lr=0.01,eps=1e-8):
    x_traj=[init]
    G=np.zeros_like(init)
    for i in range(epoch):
        grad=np.array(df(x_traj[-1][0], x_traj[-1][1]))
        G+=np.square(grad)
        dw=-lr*grad/(np.sqrt(G)+eps)
        w=x_traj[-1]+dw
        while(w[0]<-5 or w[0]>5 or w[1]<-5 or w[1]>5):
            dw/=2
            w-=dw
        x_traj.append(w)
    return x_traj

def RMSProp(df,init,epoch=500,lr=0.01,gamma=0.9,eps=1e-8):
    x_traj=[init]
    v=np.zeros_like(init)
    for i in range(epoch):
        grad=np.array(df(x_traj[-1][0], x_traj[-1][1]))
        v=gamma*v+(1-gamma)*np.square(grad)
        dw=-lr*grad/(np.sqrt(v)+eps)
        w=x_traj[-1]+dw
        while(w[0]<-5 or w[0]>5 or w[1]<-5 or w[1]>5):
            dw/=2
            w-=dw
        x_traj.append(w)
    return x_traj

def Adam(df,init,epoch=500,lr=0.01,beta1=0.9,beta2=0.99,eps=1e-8):
    x_traj=[init]
    m=np.zeros_like(init)
    v=np.zeros_like(init)
    for i in range(epoch):
        grad=np.array(df(x_traj[-1][0], x_traj[-1][1]))
        m=beta1*m+(1-beta1)*grad
        v=beta2*v+(1-beta2)*np.square(grad)
        dw=-lr*m/((np.sqrt(v/(1-beta2))+eps)*(1-beta1))
        w=x_traj[-1]+dw
        while(w[0]<-5 or w[0]>5 or w[1]<-5 or w[1]>5):
            dw/=2
            w-=dw
        x_traj.append(w)
    return x_traj

init=np.array([np.random.uniform(-5.0,5.0),np.random.uniform(-5.0, 5.0)])

# x_traj=Without(init)
# print("无优化器结果 (x_1,x2_2) = ",x_traj[-1])
# gd_plot(x_traj,"Without")

# x_traj=SGD(dbeale_dx,init)
# print("SGD优化器结果 (x_1,x2_2) = ",x_traj[-1])
# gd_plot(x_traj,"SGD")

# x_traj=Momentum(dbeale_dx,init)
# print("Momentum优化器结果 (x_1,x2_2) = ",x_traj[-1])
# gd_plot(x_traj,"Momentum")

# x_traj=AdaGrad(dbeale_dx,init,lr=5)
# print("AdaGrad优化器结果 (x_1,x2_2) = ",x_traj[-1])
# gd_plot(x_traj,"AdaGrad")

# x_traj=RMSProp(dbeale_dx,init)
# print("RMSProp优化器结果 (x_1,x2_2) = ",x_traj[-1])
# gd_plot(x_traj,"RMSProp")

x_traj=Adam(dbeale_dx,init,5000)
print("Adam优化器结果 (x_1,x2_2) = ",x_traj[-1])
gd_plot(x_traj,"Adam")