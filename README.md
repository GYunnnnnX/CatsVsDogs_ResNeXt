# CatsVsDogs 
使用ResNeXt实现的二分类模型
## 项目配置
数据集采用约25000张猫狗图片的[CatsVsDogs](https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset)。

本项目参考了作者prlz77的[pytorch实现ResNeXt](https://github.com/prlz77/ResNeXt.pytorch)项目，源自Xie等论文Aggregated residual transformations for deep neural networks。

首先向作者们和开源者们表示感谢！

由于原作者的ResNeXt的性能评估基于 Cifar10 和 Cifar100，原来的模型并不适合比较大的猫狗的image。

在本项目中，model.py被全部修改。

数据集的文件关系如下,train和val 8：2分割。
- data/
  - train/
    - Cat/
    - Dog/
  - val/
    - Cat/
    - Dog/

## Usage
在checkpoints文件夹下，已经有了训练好的模型。
### train.py
默认指令
```python
python ./train.py ./dataset  --save ./checkpoints  --log ./logs
```
其中./dataset是数据集的路径,save是保存模型的路径，log是保存日志的路径

### test.py
默认指令
```python
python ./test.py ./dataset   --load ./checkpoints/model.pytorch
```
其中./dataset是数据集的路径，后面是选择要使用的训练好的模型

### simple_predict.py
简易的预测代码。只需自己输入一张猫\狗的image即可用自己训练的模型预测。
```python
python ./simple_predict.py ./test.jpg  ./checkpoints/model.pytorch 
```
其中的./test.jpg是选择预测的图片，./checkpoints/model.pytorch是选择要使用的训练好的模型
