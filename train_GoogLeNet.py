import time
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt

from GoogLeNet_model import GoogLeNet
from mydataloader import get_train_dataloader, get_test_dataloader


def train_epoch(model,train_loader,criterion,optimizer,device):
    """训练一个epoch"""
    model.train()
    running_loss=0.0
    correct=0
    total=0

    for batch_idx,(inputs,targets) in enumerate(train_loader):
        inputs=inputs.to(device)
        targets=targets.to(device)


        outputs=model(inputs)
        loss=criterion(outputs,targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()
        predicted=outputs.argmax(1)
        total+=targets.size(0)
        correct+=predicted.eq(targets).sum().item()

    epoch_loss=running_loss/len(train_loader)
    epoch_acc=100.*correct/total

    return epoch_loss,epoch_acc


def test_epoch(model,test_loader,criterion,device):
    """验证一个epoch"""
    model.eval()
    test_loss=0.0
    correct=0
    total=0

    with torch.no_grad():
        for inputs,targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss=criterion(outputs,targets)

            test_loss+=loss.item()
            predicted=outputs.argmax(1)
            total+=targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_loss=test_loss/len(test_loader)
    test_acc=100.*correct/total

    return test_loss,test_acc


def whole_train_test_process():
    """训练与验证的完整过程"""
    batch_size=32
    learning_rate=0.001
    epochs=20
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据加载器
    train_loader=get_train_dataloader(batch_size=batch_size)
    test_loader=get_test_dataloader(batch_size=batch_size)

    # 模型
    model=GoogLeNet(num_classes=5)
    model=model.to(device)

    # 损失函数
    criterion=nn.CrossEntropyLoss()

    # GoogLeNet优化器需要用Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # # 优化器(VGG参数比较特殊，优化器需要在SGD、Adam之间调整、浅层神经网络用Adam、深层神经网络用SGD)
    # optimizer = torch.optim.SGD(
    #     model.parameters(),
    #     lr=learning_rate,  # 增大学习率
    #     momentum=0.9,  # 添加动量
    #     weight_decay=5e-4  # 权重衰减
    # )

    # 网络比较深，学习率下降的话效果将会不明显
    # 学习率调度器(每7轮,lr*=0.1,使得在后面的训练中更加能够接近最佳结果)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=7,gamma=0.1)

    # 记录训练过程
    train_losses=[]
    train_accuracies=[]
    test_losses = []
    test_accuracies = []
    best_test_acc=100.*0.0
    sum_time_use=0

    print('开始训练GoogLeNet...')

    for epoch in range(epochs):
        since = time.time()
        print('-'*20)
        print(f'当前轮次:{epoch+1}/{epochs}')

        # 训练轮次
        train_loss,train_acc=train_epoch(model,train_loader,criterion,optimizer,device)
        # 验证轮次
        test_loss,test_acc=test_epoch(model,test_loader,criterion,device)

        # 更新学习率
        scheduler.step()

        # 记录训练、测试结果
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        print(f'训练损失:{train_loss:.4f},训练准确率:{train_acc:.2f}%')
        print(f'测试损失: {test_loss:.4f}, 测试准确率: {test_acc:.2f}%')

        # 保存最佳模型参数
        if test_acc>best_test_acc:
            best_test_acc=test_acc
            torch.save(model.state_dict(),'./best_model.pth')

        # 计算训练和验证的耗时
        time_use = time.time() - since
        print("训练和验证耗费的时间{:.0f}m{:.0f}s".format(time_use // 60, time_use % 60))
        sum_time_use += time_use

    print("训练和验证耗费的总时间{:.0f}m{:.0f}s".format(sum_time_use // 60, sum_time_use % 60))

    # 训练过程记录下来
    train_process = pd.DataFrame(
        data={
            'epoch': range(1, epochs + 1),  # 训练轮次
            'train_losses': train_losses,  # 训练集损失值列表
            'test_losses': test_losses,  # 验证集损失值列表
            'train_accuracies': train_accuracies,  # 训练集准确度列表
            'test_accuracies': test_accuracies,  # 验证集准确度列表
        }
    )

    return train_process

def matplot_acc_loss(train_process):
    plt.figure(figsize=(12,4))

    plt.subplot(1,2,1)
    plt.plot(train_process['epoch'],train_process['train_losses'],'ro-',label='train loss')
    plt.plot(train_process['epoch'],train_process['test_losses'],'bs-',label='test loss')
    plt.legend()    # 打开图例
    plt.xlabel('epoch')
    plt.ylabel('loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_process['epoch'], train_process['train_accuracies'], 'ro-', label='train loss')
    plt.plot(train_process['epoch'], train_process['test_accuracies'], 'bs-', label='test loss')
    plt.legend()  # 打开图例
    plt.xlabel('epoch')
    plt.ylabel('acc')

    plt.show()

if __name__=='__main__':
    train_process=whole_train_test_process()
    matplot_acc_loss(train_process)

