import json
from collections import Counter
import os

# --- 配置区 ---
# 定义我们关心的颜色和衣物词汇
# 注意：数据集中的tokens已经是小写形式
TARGET_COLORS = ['red', 'blue', 'green', 'black', 'white', 'yellow', 'brown', 'pink', 'purple', 'orange', 'gray']
TARGET_CLOTHING = ['shirt', 'jacket', 't-shirt', 'dress', 'hat', 'jeans', 'shorts', 'pants', 'skirt', 'coat', 'suit']
TOP_N = 15  # 显示最高频的前N个结果

# --- 函数定义 ---
def analyze_captions(json_path):
    """
    加载Flickr8K数据集的JSON文件，并对训练集的描述进行深度分析。
    """
    print(f"🚀 开始分析数据集: {json_path}")

    if not os.path.exists(json_path):
        print(f"❌ 错误: JSON文件未找到 at '{json_path}'")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print("✅ JSON文件加载成功！")

    # 1. 提取所有训练集的描述 (tokens)
    train_captions_tokens = []
    for image in data['images']:
        if image['split'] == 'train':
            for sentence in image['sentences']:
                train_captions_tokens.append(sentence['tokens'])

    print(f"📊 找到 {len(train_captions_tokens)} 条训练集描述。")

    # 2. 初始化计数器
    all_words_counter = Counter()
    color_counter = Counter()
    clothing_counter = Counter()
    color_clothing_bigram_counter = Counter()

    # 3. 遍历所有描述进行统计
    for tokens in train_captions_tokens:
        # 更新总词频
        all_words_counter.update(tokens)

        # 生成bigrams (词组)
        bigrams = list(zip(tokens[:-1], tokens[1:]))

        # 遍历tokens统计颜色和衣物
        for token in tokens:
            if token in TARGET_COLORS:
                color_counter[token] += 1
            if token in TARGET_CLOTHING:
                clothing_counter[token] += 1
        
        # 遍历bigrams统计 "颜色 + 衣物" 组合
        for bigram in bigrams:
            word1, word2 = bigram
            if word1 in TARGET_COLORS and word2 in TARGET_CLOTHING:
                color_clothing_bigram_counter[f"{word1} {word2}"] += 1

    # 4. 打印分析报告
    print("\n" + "="*50)
    print("📜 训 练 集 描 述 分 析 报 告 📜")
    print("="*50)

    print(f"\n🎨 颜色词汇 Top {TOP_N} (基于预定义列表):")
    for color, count in color_counter.most_common(TOP_N):
        print(f"  - {color:<10}: {count} 次")

    print(f"\n👕 衣物词汇 Top {TOP_N} (基于预定义列表):")
    for clothing, count in clothing_counter.most_common(TOP_N):
        print(f"  - {clothing:<10}: {count} 次")
        
    print(f"\n🎨👕 '颜色 + 衣物' 组合 Top {TOP_N}:")
    for combo, count in color_clothing_bigram_counter.most_common(TOP_N):
        print(f"  - {combo:<15}: {count} 次")

    print(f"\n⭐ 最高频的 '颜色 + 衣物' 组合是: '{color_clothing_bigram_counter.most_common(1)[0][0]}'")

    print("\n" + "="*50)
    print("✅ 分析完成！")


# --- 主程序入口 ---
if __name__ == '__main__':
    # 数据集JSON文件的路径
    dataset_json_path = 'flickr8k_aim3/dataset_flickr8k.json'
    analyze_captions(dataset_json_path) 