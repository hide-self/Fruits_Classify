import torch
from torch import nn
from torchsummary import summary

class Inception(nn.Module):
    def __init__(self,in_channels,c1,c2,c3,c4):
        """
        :param in_channels: 表示输入通道数
        :param c1: 表示路径1的输出通道
        :param c2: 表示路径2的输出通道，是一个有两个值的元组,第一个是1*1卷积的输出通道，第二个是3*3卷积的输出通道
        :param c3: 表示路径3的输出通道，是一个有两个值的元组,第一个是1*1卷积的输出通道，第二个是5*5卷积的输出通道
        :param c4: 表示路径4的输出通道
        """
        super(Inception,self).__init__()

        # 路线1:单个1*1卷积层
        self.p1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, kernel_size=1, out_channels=c1),
            nn.BatchNorm2d(c1), # BatchNorm归一化，使得训练效果更好
            nn.ReLU(inplace=True)  # 每个卷积后都需要激活函数
        )

        # 路线2:1*1卷积层、3*3卷积层
        self.p2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, kernel_size=1, out_channels=c2[0]),
            nn.BatchNorm2d(c2[0]),
            nn.ReLU(inplace=True),  # 第一个卷积后的激活函数
            nn.Conv2d(in_channels=c2[0], kernel_size=3, out_channels=c2[1], padding=1),
            nn.BatchNorm2d(c2[1]),
            nn.ReLU(inplace=True)  # 第二个卷积后的激活函数
        )

        # 路线3:1*1卷积层、5*5卷积层
        self.p3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, kernel_size=1, out_channels=c3[0]),
            nn.BatchNorm2d(c3[0]),
            nn.ReLU(inplace=True),  # 第一个卷积后的激活函数
            nn.Conv2d(in_channels=c3[0], kernel_size=5, out_channels=c3[1], padding=2),
            nn.BatchNorm2d(c3[1]),
            nn.ReLU(inplace=True)  # 第二个卷积后的激活函数
        )

        # 路线4:3*3最大池化层、1*1卷积层
        self.p4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
            nn.Conv2d(in_channels=in_channels, kernel_size=1, out_channels=c4),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True)  # 卷积后的激活函数
        )

    def forward(self, x):
        path1 = self.p1(x)
        path2 = self.p2(x)
        path3 = self.p3(x)
        path4 = self.p4(x)
        return torch.cat((path1, path2, path3, path4), dim=1)   # 在通道这个维度上融合



class GoogLeNet(nn.Module):
    def __init__(self,num_classes:int):
        super(GoogLeNet,self).__init__()
        self.b1=nn.Sequential(
            nn.Conv2d(in_channels=3,kernel_size=7,stride=2,padding=3,out_channels=64),
            nn.BatchNorm2d(64),  # BN在ReLU之前
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels=64, kernel_size=1, out_channels=64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, kernel_size=3,padding=1, out_channels=192),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.b3=nn.Sequential(
            Inception(in_channels=192,c1=64,c2=(96,128),c3=(16,32),c4=32),
            Inception(in_channels=256,c1=128,c2=(128,192),c3=(32,96),c4=64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.b4 = nn.Sequential(
            Inception(in_channels=480, c1=192, c2=(96, 208), c3=(16, 48), c4=64),
            Inception(in_channels=512, c1=160, c2=(112, 224), c3=(24, 64), c4=64),
            Inception(in_channels=512, c1=128, c2=(128, 256), c3=(24, 64), c4=64),
            Inception(in_channels=512, c1=112, c2=(144, 288), c3=(32, 64), c4=64),
            Inception(in_channels=528, c1=256, c2=(160, 320), c3=(32, 128), c4=128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.b5 = nn.Sequential(
            Inception(in_channels=832, c1=256, c2=(160, 320), c3=(32, 128), c4=128),
            Inception(in_channels=832, c1=384, c2=(192, 384), c3=(48, 128), c4=128),
            nn.AdaptiveAvgPool2d((1,1)),  # 全局平均池化
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(in_features=1024,out_features=num_classes)  # 10分类任务，out_features=10
        )


        # 权重初始化
        for m in self.modules():
            if isinstance(m,nn.Conv2d): # 卷积层初始化
                nn.init.kaiming_normal_(m.weight,mode="fan_out",nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m, nn.BatchNorm2d): # BatchNorm层初始化
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m,nn.Linear):   # 全连接层初始化
                nn.init.normal_(m.weight,mean=0,std=0.01)

                if m.bias is not None:
                    nn.init.constant_(m.bias,0)


    def forward(self,x):
        x=self.b1(x)
        x=self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        y = self.b5(x)
        return y

if __name__=='__main__':
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model=GoogLeNet(num_classes=10).to(device)

    print(summary(model,(3,224,224)))





