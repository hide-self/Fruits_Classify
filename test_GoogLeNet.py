import torch

from GoogLeNet_model import GoogLeNet
from mydataloader import getAllClasses, get_test_dataloader


def evaluate_final_model(model,test_loader,device):
    """最终模型评估"""
    model.eval()
    classes=getAllClasses()

    class_correct=list(0. for _ in range(len(classes)))
    class_total=list(0. for _ in range(len(classes)))

    with torch.no_grad():
        for inputs,targets in test_loader:
            inputs=inputs.to(device)
            targets=targets.to(device)
            outputs=model(inputs)
            predicted=torch.argmax(outputs,1)   # 也可以写成outputs.argmax(1)

            c=(predicted==targets).squeeze()    # squeeze用来移除张量的单维度：(4, 1) → (4,)

            for i in range(targets.size(0)):
                label=targets[i]
                class_correct[label]+=c[i].item()
                class_total[label]+=1


    # 打印出每个类别的准确率
    print('\n每个类别的准确率:')
    for i in range(len(classes)):
        accuaracy=100*class_correct[i]/class_total[i] if class_total[i]>0 else 0
        print(f'{classes[i]:10s}:{accuaracy:.2f}%')

    # 总体准确率
    total_accuracy=100.*sum(class_correct)/sum(class_total)
    print(f'\n总体准确率:{total_accuracy:.2f}%')

    return total_accuracy

if __name__=='__main__':
    # 加载模型
    model=GoogLeNet(num_classes=5)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load('./best_model.pth',map_location=device))
    model=model.to(device)

    # 数据加载器（验证集的）
    test_loader=get_test_dataloader(batch_size=128)

    # 测试评估
    final_acc=evaluate_final_model(model=model,test_loader=test_loader,device=device)


