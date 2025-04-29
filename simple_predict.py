import torch
import argparse
from torchvision import transforms
from PIL import Image
from model import resnext50_32x4d
nlabels = 2

#使用命令：
#python ./simple_predict.py ./test.jpg  ./checkpoints/model.pytorch 
#其中的./test.jpg是选择预测的图片，./checkpoints/model.pytorch是选择要使用的训练好的模型

class Predictor:
    def __init__(self, model_path):
        self.model = resnext50_32x4d(nlabels)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        self.classes = ['cat', 'dog']
    
    def predict(self, image_path):
        img = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(img).unsqueeze(0)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            prob = torch.nn.functional.softmax(output[0], dim=0)
        
        pred_idx = torch.argmax(prob).item()
        return self.classes[pred_idx], prob[pred_idx].item()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SimplePredict ResNeXt on CatsVsDogs', 
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Positional arguments
    parser.add_argument('image', type=str, help='Root for the your image.')
    parser.add_argument('model', type=str, help='Root for the your best model.')
    args = parser.parse_args()

    predictor = Predictor(args.model)
    image_path = args.image
    pred, conf = predictor.predict(image_path)
    print(f"预测结果: {pred} | 置信度: {conf*100:.1f}%")