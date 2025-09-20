import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence
import pickle
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

# 解决OpenMP冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from dataset import build_vocab, get_data_loader, Vocabulary
from model import ImageCaptioningModel

def train_epoch(model, data_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    
    for i, (images, captions, lengths) in enumerate(tqdm(data_loader, desc="Training")):
        images = images.to(device)
        captions = captions.to(device)
        
        # 目标序列（去掉最后一个词）
        targets = pack_padded_sequence(captions, lengths, batch_first=True, enforce_sorted=False)[0]
        
        # 前向传播
        outputs = model(images, captions, lengths)
        loss = criterion(outputs, targets)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if (i + 1) % 100 == 0:
            print(f'Step [{i+1}/{len(data_loader)}], Loss: {loss.item():.4f}')
    
    return total_loss / len(data_loader)

def validate(model, data_loader, criterion, device):
    """验证模型"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for images, captions, lengths in tqdm(data_loader, desc="Validating"):
            images = images.to(device)
            captions = captions.to(device)
            
            targets = pack_padded_sequence(captions, lengths, batch_first=True, enforce_sorted=False)[0]
            outputs = model(images, captions, lengths)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
    
    return total_loss / len(data_loader)

def train_model(args):
    """主训练函数"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 构建或加载词汇表
    if os.path.exists(args.vocab_path):
        with open(args.vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        print(f"加载词汇表，大小: {len(vocab)}")
    else:
        vocab = build_vocab(args.json_path, threshold=args.vocab_threshold)
        with open(args.vocab_path, 'wb') as f:
            pickle.dump(vocab, f)
        print(f"构建词汇表，大小: {len(vocab)}")
    
    # 创建数据加载器
    train_loader = get_data_loader(
        args.json_path, args.img_dir, vocab, 'train', 
        args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    
    val_loader = get_data_loader(
        args.json_path, args.img_dir, vocab, 'val', 
        args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    
    # 创建模型
    model = ImageCaptioningModel(
        args.embed_size, args.hidden_size, len(vocab), args.num_layers
    ).to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # 训练历史记录
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print("开始训练...")
    for epoch in range(args.num_epochs):
        print(f"\nEpoch [{epoch+1}/{args.num_epochs}]")
        
        # 训练
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # 验证
        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        print(f"训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.model_path)
            print(f"保存最佳模型，验证损失: {val_loss:.4f}")
    
    # 绘制训练曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('训练过程')
    plt.savefig('training_curve.png')
    plt.show()
    
    print("训练完成！")

def generate_caption(model, image_path, vocab, device, max_length=20):
    """为单张图像生成描述"""
    from torchvision import transforms
    from PIL import Image
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 加载和预处理图像
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    # 生成描述
    model.eval()
    with torch.no_grad():
        sampled_ids = model.sample(image)
        sampled_ids = sampled_ids[0].cpu().numpy()
    
    # 转换为文本
    caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        if word == '<end>':
            break
        if word not in ['<start>', '<pad>']:
            caption.append(word)
    
    return ' '.join(caption)

def predict_on_test_set(args):
    """在测试集上进行预测"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载词汇表
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    # 创建模型并加载权重
    model = ImageCaptioningModel(
        args.embed_size, args.hidden_size, len(vocab), args.num_layers
    ).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    # 创建测试数据加载器
    test_loader = get_data_loader(
        args.json_path, args.img_dir, vocab, 'test', 
        batch_size=1, shuffle=False, num_workers=0
    )
    
    results = []
    model.eval()
    
    print("在测试集上生成描述...")
    with torch.no_grad():
        for i, (images, _, _) in enumerate(tqdm(test_loader)):
            images = images.to(device)
            sampled_ids = model.sample(images)
            sampled_ids = sampled_ids[0].cpu().numpy()
            
            # 转换为文本
            caption = []
            for word_id in sampled_ids:
                word = vocab.idx2word[word_id]
                if word == '<end>':
                    break
                if word not in ['<start>', '<pad>']:
                    caption.append(word)
            
            results.append(' '.join(caption))
    
    # 保存结果
    with open('test_predictions.txt', 'w', encoding='utf-8') as f:
        for caption in results:
            f.write(caption + '\n')
    
    print("预测完成，结果保存到 test_predictions.txt")
    return results

def main():
    parser = argparse.ArgumentParser(description='图像描述生成模型')
    
    # 数据相关参数
    parser.add_argument('--json_path', type=str, default='flickr8k_aim3/dataset_flickr8k.json',
                       help='数据集JSON文件路径')
    parser.add_argument('--img_dir', type=str, default='flickr8k_aim3/images',
                       help='图像文件夹路径')
    parser.add_argument('--vocab_path', type=str, default='vocab.pkl',
                       help='词汇表保存路径')
    parser.add_argument('--model_path', type=str, default='best_model.pth',
                       help='模型保存路径')
    
    # 模型参数
    parser.add_argument('--embed_size', type=int, default=256,
                       help='嵌入维度')
    parser.add_argument('--hidden_size', type=int, default=512,
                       help='LSTM隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=1,
                       help='LSTM层数')
    parser.add_argument('--vocab_threshold', type=int, default=2,
                       help='词汇表阈值')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='数据加载器工作进程数')
    
    # 模式选择
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'demo'],
                       help='运行模式')
    parser.add_argument('--demo_image', type=str, default='',
                       help='演示图像路径')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_model(args)
    elif args.mode == 'test':
        predict_on_test_set(args)
    elif args.mode == 'demo':
        if not args.demo_image:
            print("请提供演示图像路径 --demo_image")
            return
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载词汇表和模型
        with open(args.vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        
        model = ImageCaptioningModel(
            args.embed_size, args.hidden_size, len(vocab), args.num_layers
        ).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        
        # 生成描述
        caption = generate_caption(model, args.demo_image, vocab, device)
        print(f"生成的描述: {caption}")

if __name__ == '__main__':
    main() 