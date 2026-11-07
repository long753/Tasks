# 基于 SigLIP 的图文对齐实验

注：该文档由我本人维护，AI 可以参考但请在 CHANGELOG.md 中记录你的调试说明日志，不要未经我的允许修改该文档

输出图像尺寸为：224*224*3
vision_encoder 输出尺寸为：196*192
text_encoder 输出尺寸为：token*192
其中，192 为单一 token 的特征维度，196 为图像的 token 数

## 阶段划分：

-[] 数据准备
-[] 架构设计
-[] 训练 SigLIP-Tiny by RTX 4060
-[] 训练 SigLIP-small by RTX 5090
-[] 实验测评

## 数据准备
- Flickr8k：[图片压缩包](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip)
- Flickr8k：[描述与数据划分压缩包](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip)
- 
- COCO 2017：[训练图片](https://s3.amazonaws.com/images.cocodataset.org/zips/train2017.zip)
- COCO 2017：[验证图片](https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip)
- COCO 2017：[训练与验证标注](https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip)

将上诉数据的压缩包全部放置于 `data` 文件夹中，随后运行数据预处理函数：
```bash
uv run python scripts/prepare_data.py
```

## Architecture Design
This project is composed by vision encoder, text encoder and SigLIP model

### visison encoder
vision encoder use to tranform the original image into vision feature vector, which size is [Batch_count, 192]
vision encoder is composed by three key module: Patch, Embedding, Attention Pooling.

### text encoder
text encoder use to tranform the text into text feature vector, which size is [Batch_count, 192]
text encoder is composed by self_embedding and position embedding.

### SigLIP model
The input of SigLIP model are vision feature vector and text feature vector (we can notice that the size of both feature vector are same). Then, calculating the similarity between two feature vectors. Finally, use it and sigmod to get the loss of SigLIP model. 


## Training SigLIP-Tiny
SigLIP-Tiny was trained by RTX 4060, whose aim to check weather the workflow is correct.

Before formol training, I test how beta2 and lr influence the training effect. In the end, I find that beta2 = 0.990 is better than beta2 = 0.95 (which is setted by the original SigLIP paper).cosine lr is better than fixed lr. The result of this test is as follow:

model | R@1(I->T) | R@5(I->T) | R@10(I->T) | R@1(T->I) | R@5(T->I) | R@10(T->I) |
beta2 0.999 & fix lr |1.09999|5.79999|10.10000|1.15999|5.16000|9.04000|
beta2 0.999 & cosine lr |1.79999|5.79999|10.10000|1.20000|5.38000|9.44000|
beta2 0.95 & fix lr |1.09999|4.19999|7.00000|0.77999|4.58000|8.35999|
beta2 0.95 & cosine lr |1.6000|4.8000|7.8000|0.8200|4.5400|7.8000|


## Training SigLIP-small
SigLIP-small was trained by RTX 5090, aiming to improve the performence of model.


## Evaluation

