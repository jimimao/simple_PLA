import scipy.io as scio
import numpy as np
##加载数据
def loadmat(name):
    path = 'data/' + name
    data = scio.loadmat(path)
    Train_data = data['Ldata']
    Train_target = data['Ltarget']
    Test_data = data['Tdata']
    Test_target = data['Ttarget']
    return Train_data, Train_target, Test_data, Test_target

def defineweight(n):
    w = [0] * (n)
    w = np.mat(w)
    return w

# t 为阈值
def predict(weight,X,t):
    res = np.mat(weight) * np.transpose(np.mat(X)) ## * == np.dot(),矩阵乘法
    prediction = np.sign(res + t)
    return prediction

#更新权重矩阵,每一条x更新一次
def renewweight(weight,prediction,target,lr,X):
    renewweight = np.mat(weight) + lr * (target - prediction) * np.mat(X)
    return renewweight

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
    epoch = 500
    t = 0.1

    Train_data, Train_target, Test_data, Test_target = loadmat('diabetes.mat')
    # Train_data = np.mat(Train_data)
    # Train_target = np.mat(Train_target)
    # Test_data = np.mat(Test_data)
    # Test_target = np.mat(Test_target)
    m = Train_data.shape[0] #数据 训练集长度
    n = Train_data.shape[1] #数据 维度
    #
    #训练
    weight = defineweight(n) # 初始化权重矩阵
    # print(weight)
    prediction = [0]*m
    for i in range(0,epoch):
        # 预测结果
        for j in range(0,m):
            prediction[j] = predict(weight,Train_data[j],t)
        for j in range(0, m):
            # 对于一条数据的 n 个维度，都要更新权重
                weight = renewweight(weight, prediction[j], Train_target[j], lr, Train_data[j])
        # print(weight)
        pre_train = np.sign(weight * np.transpose(np.mat(Train_data)))
        acc_train = acc(pre_train,Train_target)
        # print(pre_train)
        print("TrainACC%d:%.5f"%(i,acc_train))

    ##测试
    print("Test ACC")
    mt = Test_data.shape[0] #数据 训练集长度
    nt = Test_data.shape[1] #数据 维度
    pre = np.sign(weight * np.transpose(np.mat(Test_data))) # pre 为1 * mt，Test_target 为mt *1
    acc_test = acc(pre,Test_target)
    print("%.5f"%acc_test)