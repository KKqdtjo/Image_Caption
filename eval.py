import torch
import pickle
import json
import os
from tqdm import tqdm
import argparse
from collections import defaultdict
import nltk
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt
from PIL import Image
import random

# 解决OpenMP冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 设置matplotlib中文字体
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

from dataset import get_data_loader, Vocabulary
from model import ImageCaptioningModel

# 下载必要的NLTK数据
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

def generate_captions_for_eval(model, data_loader, vocab, device):
    """为评估生成所有图像的描述"""
    model.eval()
    generated_captions = []
    reference_captions = []
    
    with torch.no_grad():
        for images, captions, lengths in tqdm(data_loader, desc="生成描述"):
            images = images.to(device)
            
            # 生成描述
            sampled_ids = model.sample(images)
            
            for i in range(images.size(0)):
                # 转换生成的描述
                generated = []
                for word_id in sampled_ids[i].cpu().numpy():
                    word = vocab.idx2word[word_id]
                    if word == '<end>':
                        break
                    if word not in ['<start>', '<pad>']:
                        generated.append(word)
                
                # 转换真实描述
                reference = []
                caption_length = lengths[i] if isinstance(lengths, list) else lengths[i].item()
                for word_id in captions[i][:caption_length].cpu().numpy():
                    word = vocab.idx2word[word_id]
                    if word == '<end>':
                        break
                    if word not in ['<start>', '<pad>']:
                        reference.append(word)
                
                generated_captions.append(generated)
                reference_captions.append([reference])  # BLEU需要参考句子列表
    
    return generated_captions, reference_captions

def calculate_bleu_scores(generated_captions, reference_captions):
    """计算BLEU分数"""
    # 计算corpus-level BLEU
    bleu_1 = corpus_bleu(reference_captions, generated_captions, weights=(1, 0, 0, 0))
    bleu_2 = corpus_bleu(reference_captions, generated_captions, weights=(0.5, 0.5, 0, 0))
    bleu_3 = corpus_bleu(reference_captions, generated_captions, weights=(0.33, 0.33, 0.33, 0))
    bleu_4 = corpus_bleu(reference_captions, generated_captions, weights=(0.25, 0.25, 0.25, 0.25))
    
    return {
        'BLEU-1': bleu_1,
        'BLEU-2': bleu_2,
        'BLEU-3': bleu_3,
        'BLEU-4': bleu_4
    }

def calculate_meteor_scores(generated_captions, reference_captions):
    """计算METEOR分数"""
    meteor_scores = []
    
    for gen, refs in zip(generated_captions, reference_captions):
        # METEOR需要字符串格式
        gen_str = ' '.join(gen)
        ref_str = ' '.join(refs[0])
        
        try:
            score = meteor_score([ref_str], gen_str)
            meteor_scores.append(score)
        except:
            meteor_scores.append(0.0)
    
    return sum(meteor_scores) / len(meteor_scores)

def calculate_rouge_scores(generated_captions, reference_captions):
    """计算ROUGE分数"""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rouge_scores = defaultdict(list)
    
    for gen, refs in zip(generated_captions, reference_captions):
        gen_str = ' '.join(gen)
        ref_str = ' '.join(refs[0])
        
        scores = scorer.score(ref_str, gen_str)
        
        rouge_scores['ROUGE-1'].append(scores['rouge1'].fmeasure)
        rouge_scores['ROUGE-2'].append(scores['rouge2'].fmeasure)
        rouge_scores['ROUGE-L'].append(scores['rougeL'].fmeasure)
    
    return {
        'ROUGE-1': sum(rouge_scores['ROUGE-1']) / len(rouge_scores['ROUGE-1']),
        'ROUGE-2': sum(rouge_scores['ROUGE-2']) / len(rouge_scores['ROUGE-2']),
        'ROUGE-L': sum(rouge_scores['ROUGE-L']) / len(rouge_scores['ROUGE-L'])
    }

def show_visual_results_single_image(model, data_loader, vocab, device, img_dir, split_name):
    """展示单张图片的多个标注对比"""
    print(f"\n📸 单图多标注展示模式...")
    
    model.eval()
    
    # 收集一张图片的多个标注
    with torch.no_grad():
        for images, captions, lengths in data_loader:
            images = images.to(device)
            sampled_ids = model.sample(images)
            
            # 只取第一张图片
            image = images[0]
            dataset = data_loader.dataset
            filename = dataset.data[0]['filename']
            
            # 获取AI生成的描述
            generated = []
            for word_id in sampled_ids[0].cpu().numpy():
                word = vocab.idx2word[word_id]
                if word == '<end>':
                    break
                if word not in ['<start>', '<pad>']:
                    generated.append(word)
            ai_caption = ' '.join(generated)
            
            # 获取这张图片的所有人工标注
            human_captions = []
            for sentence in dataset.data[0]['sentences']:
                human_captions.append(' '.join(sentence['tokens']))
            
            # 创建展示
            fig = plt.figure(figsize=(16, 12))
            
            # 左侧显示图像
            ax_img = plt.subplot2grid((3, 2), (0, 0), rowspan=3)
            
            img_path = os.path.join(img_dir, filename)
            if os.path.exists(img_path):
                original_image = Image.open(img_path).convert('RGB')
                ax_img.imshow(original_image)
                ax_img.axis('off')
                ax_img.set_title(f'图像: {filename}', fontsize=14, fontweight='bold', pad=20)
            
            # 右侧显示AI生成和人工标注
            ax_text = plt.subplot2grid((3, 2), (0, 1), rowspan=3)
            ax_text.axis('off')
            
            # 构建文本内容
            text_content = f" 模型生成描述:\n{ai_caption}\n\n"
            text_content += f" 人工标注 (共{len(human_captions)}条):\n"
            for idx, caption in enumerate(human_captions, 1):
                text_content += f"{idx}. {caption}\n"
            
            ax_text.text(0.05, 0.95, text_content, 
                        transform=ax_text.transAxes,
                        fontsize=12, va='top', ha='left',
                        bbox=dict(boxstyle="round,pad=1", 
                                facecolor="lightyellow", 
                                alpha=0.9,
                                edgecolor="orange",
                                linewidth=2))
            
            plt.suptitle('单图多标注对比展示', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            plt.savefig(f'eval_single_image_{split_name}.png', dpi=200, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"💾 单图展示已保存为 'eval_single_image_{split_name}.png'")
            
            plt.show()
            break
    
    print("✅ 单图多标注展示完成！")

def show_visual_results_diverse_images(model, data_loader, vocab, device, img_dir, split_name, num_samples=15):
    """展示多张不同图片，每张一个标注"""
    print(f"\n📸 多图单标注展示模式（{num_samples}张不同图片）...")
    
    model.eval()
    
    # 收集不同图片的样本 - 使用随机采样确保图片多样性
    import random
    random.seed(42)  # 固定随机种子保证可重现
    
    dataset = data_loader.dataset
    unique_files = list(set([data['filename'] for data in dataset.data]))
    selected_files = random.sample(unique_files, min(num_samples, len(unique_files)))
    
    samples_collected = []
    
    with torch.no_grad():
        # 为每个选中的文件生成描述
        for filename in selected_files:
            # 找到对应的数据索引
            data_idx = next(i for i, data in enumerate(dataset.data) if data['filename'] == filename)
            
            # 获取图像和标注
            image, caption = dataset[data_idx]
            image = image.unsqueeze(0).to(device)  # 添加batch维度
            
            # 生成描述
            sampled_ids = model.sample(image)
            generated = []
            for word_id in sampled_ids[0].cpu().numpy():
                word = vocab.idx2word[word_id]
                if word == '<end>':
                    break
                if word not in ['<start>', '<pad>']:
                    generated.append(word)
            
            # 获取真实描述（从dataset.data中直接获取caption）
            reference = ' '.join(dataset.data[data_idx]['caption'])
            
            samples_collected.append({
                'filename': filename,
                'generated': ' '.join(generated),
                'reference': reference
            })
    
    print(f"\n📋 多图展示详情:")
    
    for i, sample in enumerate(samples_collected):
        print(f"\n  📸 图片 {i+1}:")
        print(f"     📁 文件: {sample['filename']}")
        print(f"     🤖 AI生成: {sample['generated']}")
        print(f"     👨 人工标注: {sample['reference']}")
        
        # 为每张图像创建单独的图表
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # 加载原始图像
        img_path = os.path.join(img_dir, sample['filename'])
        
        try:
            if os.path.exists(img_path):
                original_image = Image.open(img_path).convert('RGB')
                
                # 显示图像
                ax.imshow(original_image)
                ax.axis('off')
                
                # 添加生成描述和真实描述 - 支持换行显示
                generated_text = sample['generated']
                reference_text = sample['reference']
                
                # 创建对比文本 - 让matplotlib自动换行
                comparison_text = f"模型生成描述:\n{generated_text}\n\n人工标注:\n{reference_text}"
                
                # 添加文本框 - 在图像下方显示，增大字体和框大小
                ax.text(0.5, -0.18, comparison_text,
                       transform=ax.transAxes,
                       ha='center', va='top',
                       fontsize=14,
                       bbox=dict(boxstyle="round,pad=1.2", 
                               facecolor="lightblue", 
                               alpha=0.9,
                               edgecolor="darkblue",
                               linewidth=1.5),
                       wrap=True,
                       horizontalalignment='center',
                       verticalalignment='top')
                
                print(f"     ✅ 图像加载成功")
                
            else:
                ax.text(0.5, 0.5, f"图像不存在\n{sample['filename']}",
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=14, color='red')
                ax.axis('off')

                print(f"     ❌ 图像文件不存在")
                
        except Exception as e:
            ax.text(0.5, 0.5, f"加载失败\n{str(e)}",
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=12, color='red')
            ax.axis('off')
            
            print(f"     ⚠️ 错误: {e}")
        
                 # 调整布局 - 为更大的文本框留出更多空间
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.3)
        
        # 保存单独的图像
        plt.savefig(f'eval_image_{i+1}_{split_name}.png', dpi=200, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"     💾 图像 {i+1} 已保存为 'eval_image_{i+1}_{split_name}.png'")
        
        # 显示图像
        plt.show()
        
        # 关闭当前图表以释放内存
        plt.close(fig)
    
    print(f"\n✅ 多图展示完成！")

def show_visual_results(model, data_loader, vocab, device, img_dir, split_name, mode='diverse', num_samples=15):
    """可视化展示模型生成效果
    
    Args:
        mode: 'single' - 单图多标注, 'diverse' - 多图单标注
        num_samples: 多图模式下展示的图片数量
    """
    
    if mode == 'single':
        show_visual_results_single_image(model, data_loader, vocab, device, img_dir, split_name)
    else:
        show_visual_results_diverse_images(model, data_loader, vocab, device, img_dir, split_name, num_samples)

def evaluate_model(args):
    """评估模型性能"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载词汇表
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    print(f"词汇表大小: {len(vocab)}")
    
    # 创建模型并加载权重
    model = ImageCaptioningModel(
        args.embed_size, args.hidden_size, len(vocab), args.num_layers
    ).to(device)
    
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print("加载训练好的模型")
    else:
        print("警告: 未找到训练好的模型，使用随机初始化的模型")
    
    # 创建测试数据加载器
    test_loader = get_data_loader(
        args.json_path, args.img_dir, vocab, args.split, 
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    
    print(f"在{args.split}集上评估模型...")
    
    # 生成描述
    generated_captions, reference_captions = generate_captions_for_eval(
        model, test_loader, vocab, device
    )
    
    print(f"生成了 {len(generated_captions)} 个描述")
    
    # 计算评估指标
    print("计算评估指标...")
    
    # BLEU分数
    bleu_scores = calculate_bleu_scores(generated_captions, reference_captions)
    
    # METEOR分数
    meteor = calculate_meteor_scores(generated_captions, reference_captions)
    
    # ROUGE分数
    rouge_scores = calculate_rouge_scores(generated_captions, reference_captions)
    
    # 打印结果
    print("\n=== 评估结果 ===")
    for metric, score in bleu_scores.items():
        print(f"{metric}: {score:.4f}")
    
    print(f"METEOR: {meteor:.4f}")
    
    for metric, score in rouge_scores.items():
        print(f"{metric}: {score:.4f}")
    
    # 显示一些示例
    print("\n=== 生成示例 ===")
    for i in range(min(5, len(generated_captions))):
        print(f"生成: {' '.join(generated_captions[i])}")
        print(f"真实: {' '.join(reference_captions[i][0])}")
        print("-" * 50)
    
    # 可视化展示生成效果
    show_visual_results(model, test_loader, vocab, device, args.img_dir, args.split, args.visual_mode, num_samples=15)
    
    # 保存评估结果
    results = {
        **bleu_scores,
        'METEOR': meteor,
        **rouge_scores
    }
    
    with open(f'eval_results_{args.split}.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"评估结果保存到 eval_results_{args.split}.json")

def main():
    parser = argparse.ArgumentParser(description='模型评估')
    
    # 数据相关参数
    parser.add_argument('--json_path', type=str, default='flickr8k_aim3/dataset_flickr8k.json',
                       help='数据集JSON文件路径')
    parser.add_argument('--img_dir', type=str, default='flickr8k_aim3/images',
                       help='图像文件夹路径')
    parser.add_argument('--vocab_path', type=str, default='vocab.pkl',
                       help='词汇表路径')
    parser.add_argument('--model_path', type=str, default='best_model.pth',
                       help='模型路径')
    
    # 模型参数
    parser.add_argument('--embed_size', type=int, default=256,
                       help='嵌入维度')
    parser.add_argument('--hidden_size', type=int, default=512,
                       help='LSTM隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=1,
                       help='LSTM层数')
    
    # 评估参数
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                       help='评估的数据集分割')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='批次大小')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='数据加载器工作进程数')
    parser.add_argument('--visual_mode', type=str, default='diverse', choices=['single', 'diverse'],
                       help='可视化模式: single-单图多标注, diverse-多图单标注')
    
    args = parser.parse_args()
    evaluate_model(args)

if __name__ == '__main__':
    main() 