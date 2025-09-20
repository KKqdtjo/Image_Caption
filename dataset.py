import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from collections import Counter # 计数器，用于统计词频
import pickle  # 序列化工具，保存/加载词汇表
import matplotlib.pyplot as plt
import numpy as np

# 解决OpenMP冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 设置matplotlib中文字体
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

class Vocabulary:
    """词汇表类，用于处理文本的编码和解码"""
    def __init__(self):
        self.word2idx = {}   #词汇→索引的映射字典
        self.idx2word = {}  #索引→词汇的映射字典
        self.idx = 0        #当前索引计数器
        
        # 添加特殊标记
        self.add_word('<pad>')  # 填充标记，用于统一序列长度
        self.add_word('<start>')  # 句子开始标记，告诉模型句子开始
        self.add_word('<end>')  # 句子结束标记， 告诉模型句子结束
        self.add_word('<unk>')  # 未知词标记
        
    def add_word(self, word):
        """添加单词到词汇表"""
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
            
    def __call__(self, word):
        """获取单词的索引"""
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]
    
    def __len__(self):
        return len(self.word2idx)

def build_vocab(json_path, threshold=2):   
    """构建词汇表"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 统计所有词汇的出现频率
    counter = Counter()
    for image in data['images']:
        for sentence in image['sentences']:
            tokens = sentence['tokens']
            counter.update(tokens)
    
    # 只保留出现次数大于等于threshold的词
    words = [word for word, cnt in counter.items() if cnt >= threshold]
    
    vocab = Vocabulary()
    for word in words:
        vocab.add_word(word)
    
    return vocab

class Flickr8kDataset(Dataset):
    """Flickr8K数据集类"""
    def __init__(self, json_path, img_dir, vocab, split='train', transform=None):#不对图像变到同一大小吗？
        """
        Args:
            json_path: 数据集JSON文件路径
            img_dir: 图像文件夹路径
            vocab: 词汇表对象
            split: 数据集分割 ('train', 'val', 'test')
            transform: 图像预处理变换
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.img_dir = img_dir
        self.vocab = vocab
        self.transform = transform
        
        # 筛选指定split的数据，同时验证图像文件存在性（我发现源数据集里面有8091张，而json记录的仅有8000张）
        self.data = []
        missing_files = []
        
        for image in data['images']:
            if image['split'] == split:
                img_path = os.path.join(img_dir, image['filename'])
                
                # 检查图像文件是否存在
                if os.path.exists(img_path):
                    for sentence in image['sentences']:
                        self.data.append({
                            'filename': image['filename'],
                            'caption': sentence['tokens']
                        })
                else:
                    missing_files.append(image['filename'])
        
        # 报告缺失文件情况
        if missing_files:
            print(f"注意：{split}集中有 {len(missing_files)} 个图像文件缺失，已自动跳过")
            print(f"实际可用的 {split} 数据：{len(self.data)} 个图文对")
        else:
            print(f"{split}集数据加载完成：{len(self.data)} 个图文对")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 加载图像
        img_path = os.path.join(self.img_dir, item['filename'])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # 处理标题文本
        caption = item['caption']
        caption = ['<start>'] + caption + ['<end>']
        
        # 转换为索引
        caption_indices = [self.vocab(word) for word in caption]
        caption_tensor = torch.tensor(caption_indices, dtype=torch.long)
        
        return image, caption_tensor

def collate_fn(batch):#PyTorch要求批次中张量形状一致 - 需要将不同长度序列填充到相同长度
    """自定义批处理函数，处理不同长度的标题"""
    # 按标题长度排序
    batch.sort(key=lambda x: len(x[1]), reverse=True)#LSTM的pack_padded_sequence要求按长度降序 - 提高RNN处理效率，减少无效计算
    
    images, captions = zip(*batch)
    
    # 堆叠图像
    images = torch.stack(images, 0)# 为啥图像形状已统一，直接堆叠？？？
    
    # 填充标题到相同长度
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long() # 创建填充矩阵
    
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    
    return images, targets, lengths

def get_data_loader(json_path, img_dir, vocab, split='train', batch_size=32, shuffle=True, num_workers=0):
    """获取数据加载器"""
    # 图像预处理
    if split == 'train':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(0.5),# 50%概率水平翻转 (数据增强)
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    dataset = Flickr8kDataset(json_path, img_dir, vocab, split, transform)
    
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,# 是否打乱 (训练时True，测试时False)
        num_workers=num_workers, # 多进程加载 (Windows建议设为0)
        collate_fn=collate_fn
    )
    
    return data_loader

if __name__ == '__main__':
    # 测试代码
    json_path = 'flickr8k_aim3/dataset_flickr8k.json'
    img_dir = 'flickr8k_aim3/images'
    
    # 构建词汇表
    vocab = build_vocab(json_path)
    print(f"词汇表大小: {len(vocab)}")
    
    # 保存词汇表
    with open('vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    
    # 创建数据加载器
    train_loader = get_data_loader(json_path, img_dir, vocab, 'train', batch_size=4)
    
    # 获取数据集对象以便后续获取原始数据
    train_dataset = train_loader.dataset
    
    # 测试一个批次
    print("\n=== 批次数据展示 ===")
    for batch_idx, (images, captions, lengths) in enumerate(train_loader):
        print(f"📊 批次 {batch_idx + 1} 基本信息:")
        print(f"   图像张量形状: {images.shape}")
        print(f"   标题张量形状: {captions.shape}")
        print(f"   序列长度列表: {lengths}")
        
        print(f"\n📋 批次内容详情:")
        batch_size = images.shape[0]
        
        # 从训练集中随机选择4张不同的图像
        import random
        random.seed(42)  # 固定随机种子
        
        # 获取所有唯一的图像文件名
        unique_images = {}
        for idx, sample in enumerate(train_dataset.data):
            filename = sample['filename']
            if filename not in unique_images:
                unique_images[filename] = idx  # 保存第一次出现的索引
        
        # 随机选择4张不同图像
        selected_files = random.sample(list(unique_images.keys()), min(4, len(unique_images)))
        
        print(f"🎲 随机选择的图像文件:")
        for i, filename in enumerate(selected_files):
            print(f"   {i+1}. {filename}")
        
        # 创建图像显示窗口
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes = axes.flatten()
        
        print(f"\n📋 样本详情:")
        
        for i, filename in enumerate(selected_files):
            print(f"\n  🔸 图像 {i+1}:")
            print(f"     📁 文件: {filename}")
            
            # 获取该图像的第一个描述
            sample_idx = unique_images[filename]
            image, caption_tensor = train_dataset[sample_idx]
            
            # 解码描述
            caption_indices = caption_tensor.tolist()
            caption_words = []
            for idx in caption_indices:
                word = vocab.idx2word[idx]
                if word not in ['<pad>']:
                    caption_words.append(word)
            
            # 去除特殊标记的干净描述
            clean_caption = ' '.join([w for w in caption_words if w not in ['<start>', '<end>']])
            print(f"     📝 描述: {clean_caption}")
            
            # 加载原始图像
            img_path = os.path.join(img_dir, filename)
            
            try:
                if os.path.exists(img_path):
                    original_image = Image.open(img_path).convert('RGB')
                    
                    # 显示图像（不显示标题，避免重复）
                    axes[i].imshow(original_image)
                    axes[i].axis('off')  # 移除坐标轴
                    
                    # 在图像下方添加描述文本
                    if len(clean_caption) > 85:
                        display_caption = clean_caption[:85] + "..."
                    else:
                        display_caption = clean_caption
                    
                    axes[i].text(0.5, -0.08, display_caption,
                               transform=axes[i].transAxes,
                               ha='center', va='top',
                               fontsize=11, wrap=True,
                               bbox=dict(boxstyle="round,pad=0.6", 
                                       facecolor="lightsteelblue", 
                                       alpha=0.9,
                                       edgecolor="steelblue",
                                       linewidth=1))
                    
                    print(f"     ✅ 加载成功")
                else:
                    axes[i].text(0.5, 0.5, f"图像不存在",
                               transform=axes[i].transAxes, ha='center', va='center',
                               fontsize=14, color='red')
                    axes[i].axis('off')
                    print(f"     ❌ 文件不存在")
                    
            except Exception as e:
                print(f"     ⚠️  错误: {e}")
                axes[i].text(0.5, 0.5, f"加载失败",
                           transform=axes[i].transAxes, ha='center', va='center',
                           fontsize=14, color='red')
                axes[i].axis('off')
            
            print(f"     {'-'*60}")
        
        # 优化布局
        plt.tight_layout()
        plt.subplots_adjust(top=0.92, bottom=0.15, hspace=0.3, wspace=0.1)
        
        # 保存图像
        plt.savefig('dataset_samples.png', dpi=200, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"\n💾 数据集样本展示已保存为 'dataset_samples.png'")
        
        # 显示图像
        plt.show()
        
        print(f"\n✅ 多样化图像展示完成！")
        
        break 