import torch
from PIL import Image
from torchvision import transforms
from GoogLeNet_model import GoogLeNet
from mydataloader import getAllClasses


def predict_single_image(model,image_path,transform,device):
    """预测单张图片"""

    # 加载图像
    # 加载图像 - 使用 with 语句确保文件正确关闭
    with Image.open(image_path) as image:
        # 可选：将图像转换为RGB模式（避免RGBA等格式问题）
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # 直接复制图像数据到内存，避免文件依赖
        image = image.copy()

    # 预处理
    image_tensor=transform(image).unsqueeze(0).to(device)

    # 预测
    model.eval()
    with torch.no_grad():
        output=model(image_tensor)
        probabilities=torch.nn.functional.softmax(output[0],dim=0)
        predicted_class=torch.argmax(probabilities).item()
        confidence=probabilities[predicted_class].item()

    return predicted_class,confidence



if __name__=='__main__':
    # 示例使用（需要实际图像文件）
    # 加载模型
    model = GoogLeNet(num_classes=5)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load('./best_model.pth', map_location=device))
    # 图像预处理
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.687, 0.589, 0.430],
            std=[0.298, 0.293, 0.340]
        )
    ])

    predicted_class, confidence = predict_single_image(model, './image_for_test/test_grape.png', test_transform, device)

    classes=getAllClasses()
    print(f'预测类别: {classes[predicted_class]}, 置信度: {confidence:.4f}')


