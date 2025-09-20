import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class ImageEncoder(nn.Module):
    """图像编码器 - 使用预训练的ResNet"""
    def __init__(self, embed_size):
        super(ImageEncoder, self).__init__()
        # 使用预训练的ResNet-50作为特征提取器
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # 移除最后的全连接层
        modules = list(resnet.children())[:-1]# 去掉最后的fc层，只保留特征提取部分
        self.resnet = nn.Sequential(*modules)
        
        # 添加线性层将ResNet特征映射到嵌入维度
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
    def forward(self, images):
        """前向传播"""
        with torch.no_grad():
            features = self.resnet(images)
        
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features

class TextDecoder(nn.Module):
    """文本解码器 - 使用LSTM"""
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, max_seq_length=20):
        super(TextDecoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        
        # 词嵌入层
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        # LSTM层
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        
        # 输出层
        self.linear = nn.Linear(hidden_size, vocab_size)
        
        # Dropout层
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, features, captions, lengths):
        """训练时的前向传播"""
        # 词嵌入
        embeddings = self.embed(captions)
        
        # 将图像特征作为第一个输入
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        
        # 打包序列以处理不同长度
        packed = nn.utils.rnn.pack_padded_sequence(
            embeddings, lengths, batch_first=True, enforce_sorted=False
        )
        
        # LSTM前向传播
        hiddens, _ = self.lstm(packed)
        
        # 输出层
        outputs = self.linear(hiddens[0])
        
        return outputs
    
    def sample(self, features, states=None):
        """推理时的采样生成"""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        
        for i in range(self.max_seq_length):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            _, predicted = outputs.max(1)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)
            
        sampled_ids = torch.stack(sampled_ids, 1)
        return sampled_ids

class ImageCaptioningModel(nn.Module):
    """完整的图像描述生成模型"""
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = ImageEncoder(embed_size)
        self.decoder = TextDecoder(embed_size, hidden_size, vocab_size, num_layers)
        
    def forward(self, images, captions, lengths):
        """训练时的前向传播"""
        features = self.encoder(images)
        outputs = self.decoder(features, captions, lengths)
        return outputs
    
    def sample(self, images):
        """推理时生成描述"""
        features = self.encoder(images)
        sampled_ids = self.decoder.sample(features)
        return sampled_ids

if __name__ == '__main__':
    # 测试模型
    embed_size = 256
    hidden_size = 512
    vocab_size = 1000
    
    model = ImageCaptioningModel(embed_size, hidden_size, vocab_size)
    
    # 测试输入
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224)
    captions = torch.randint(0, vocab_size, (batch_size, 10))
    lengths = [10, 8, 6, 5]
    
    # 前向传播测试
    outputs = model(images, captions, lengths)
    print(f"输出形状: {outputs.shape}")
    
    # 采样测试
    sampled = model.sample(images)
    print(f"采样输出形状: {sampled.shape}") 