from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from typing import List

def kfold_dataloader(k, dataset_construct,X,args_train:List,args_test:List,batch_train,batch_test):
    """返回k个trainloader,testloader

    Args:
        k (int): k
        dataset_construct (): 用于构造dataset的方法
        X (): 用于划分的数据,同时也是dataset_construt的第一个参数
        args_train:训练集的剩余构造参数
        args_test:测试集的剩余构造参数
        batch_train:训练集batch
        batch_test:测试集batch
    """

    kf = KFold(n_splits=k)
    for train_index, test_index in kf.split(X):
        trainset = dataset_construct(X[train_index],*args_train)
        testset = dataset_construct(X[test_index],*args_test)

        trainloader = DataLoader(trainset, batch_size=batch_train, shuffle=True)
        testloader = DataLoader(testset, batch_size=batch_test, shuffle=False)
        yield trainloader, testloader