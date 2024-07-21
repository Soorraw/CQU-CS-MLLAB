import numpy as np

class Logistic_Regression:
    def __init__(self,X,y,c1,c2,times,delta):
        self.X=X
        self.X=self.initX()#插入一列
        self.y=y
        self.row=self.X.shape[0]
        self.col=self.X.shape[1]
        self.beta=self.initBeta()
        self.c1=c1
        self.c2=c2
        self.times=times
        self.delta=delta
    def exp(self,x):
        x[x>=30]=1.0e8
        x[x<30]=np.exp(x)
        return x
    def initX(self,):
        return np.insert(self.X,self.X.shape[1],1,axis=1)
    def initBeta(self):
        beta=np.random.rand(self.col)
        beta.reshape(self.col,1)
        return beta
    def Cost(self,beta):
        power=-self.y*np.dot(self.X,beta)
        x=power[power<0]
        y=power[power>=0]
        cost=np.sum(np.log(1+self.exp(x)))+np.sum(np.log(1+self.exp(-y))+y)#ln(1+e^x)=ln(1+e^(-x))+x
        return cost
    def Grad(self,beta):
        k=-self.y/(1+self.exp(self.y*np.dot(self.X,beta)))
        grad=np.dot(self.X.T,k)
        return grad
    def Hessian(self,beta):
        exp=self.exp(self.y*np.dot(self.X,beta))
        k=exp/np.square(1+exp)
        hessian=np.zeros((self.row,self.col))
        for i in range(self.col):
            hessian[:,i]=k*self.X[:,i]
        hessian=np.dot(hessian.T,(self.X))
        return hessian
    def Direction(self,grad,hessian):
        H_inv=np.linalg.inv(hessian)
        direction=-np.dot(H_inv,grad)
        return direction
    def isArmijo(self,alpha):
        g=self.Grad(self.beta)
        h=self.Hessian(self.beta)
        d=self.Direction(g,h)
        return self.Cost(self.beta+alpha*d)<=self.Cost(self.beta)+self.c1*alpha*np.dot(d.T,g)
    def isWolfe(self,alpha):
        g=self.Grad(self.beta)
        h=self.Hessian(self.beta)
        d=self.Direction(g,h)
        return np.dot(d.T,self.Grad(self.beta+alpha*d))>=self.c2*np.dot(d.T,g)
    def Line_Search(self,l,r):
        alpha=(l+r)/2
        g=self.Grad(self.beta)
        h=self.Hessian(self.beta)
        d=self.Direction(g,h)
        while(True):
            isArmijo=self.isArmijo(alpha)
            isWolfe=self.isWolfe(alpha)
            if isArmijo and isWolfe:
                return alpha
            elif not isArmijo and isWolfe:
                r=alpha
            elif isArmijo and not isWolfe:
                l=alpha
            elif np.dot(d.T,self.Grad(self.beta+alpha*d))<0:
                l=alpha
            else: r=alpha
            alpha=(l+r)/2
    def Binary_Classification(self,l,r):
        for i in range(self.times):
            g=self.Grad(self.beta)
            h=self.Hessian(self.beta)
            if np.linalg.norm(g)<=self.delta:
                break
            alpha=self.Line_Search(l,r)
            self.beta=self.beta+alpha*self.Direction(g,h)
        return self.beta
