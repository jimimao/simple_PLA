import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt

##加载数据
def loadmat(name):
    path = 'data/' + name
    data = scio.loadmat(path)
    Train_data = data['Ldata']
    Train_target = data['Ltarget']
    Test_data = data['Tdata']
    Test_target = data['Ttarget']
    return Train_data, Train_target, Test_data, Test_target

#权重矩阵初始化
def defineweight(n):
    w = [0] * (n)
    w = np.mat(w)
    return w

# t 为阈值
def predict(weight,X,t):
    res = np.mat(weight) * np.transpose(np.mat(X)) ## * == np.dot(),矩阵乘法
    prediction = np.sign(res + t)
    return prediction

#更新权重矩阵
def renewweight(weight,prediction,target,lr,Train_data,m):
    prediction = np.transpose(prediction)
    for i in range(0,m):
        weight = np.mat(weight) + lr * (target[i] - prediction[i]) * np.mat(Train_data[i])
    return weight


#计算正确率
def acc(pre,data_target):
    l = data_target.shape[0]
    tt = np.array(pre) + np.transpose(np.array(data_target))
    wrong = np.sum(tt == 0)
    return 1- wrong/l

#开始训练
if __name__ == "__main__":
    #超参数
    lr = 0.01
    epoch = 100
    t = 0.1  # threshold

    #正确率
    acclist = []

    Train_data, Train_target, Test_data, Test_target = loadmat('diabetes.mat')
    m = Train_data.shape[0] #数据 训练集长度
    n = Train_data.shape[1] #数据 维度
    #
    #训练
    weight = defineweight(n) # 初始化权重矩阵
    # print(weight)
    prediction = [0]*m
    for i in range(0,epoch):
        # 预测结果
        prediction = predict(weight,Train_data,t)
        # 对于一条数据的 n 个维度，都要更新权重, t作为threshold 也要更新
        weight = renewweight(weight, prediction, Train_target, lr, Train_data,m)
        # print(weight)
        #利用目前的weight与threshold，对现有的训练数据集进行预测，并记录
        pre_train = predict(weight,Train_data,t)
        acc_train = acc(pre_train,Train_target)
        # print(pre_train)
        print("TrainACC%d:%.5f"%(i,acc_train))
        acclist.append(acc_train * 100)

    ##测试
    print("Test ACC")
    mt = Test_data.shape[0] #数据 训练集长度
    nt = Test_data.shape[1] #数据 维度
    pre = predict(weight,Test_data,t) # pre 为1 * mt，Test_target 为mt *1
    acc_test = acc(pre,Test_target)
    print("%.5f"%acc_test)

    ##ACC的折线图
    x1 = range(0,epoch)
    y1 = acclist
    plt.subplot(1,1,1) #第一个1 表示一共几幅图
    plt.plot(x1,y1,'o-')
    plt.title("ACC vs. epoch")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.show()