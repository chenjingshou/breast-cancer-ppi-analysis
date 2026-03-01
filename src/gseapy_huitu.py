import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置绘图风格
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
import seaborn as sns

def manual_plot(module_name):
    csv_file = f"enrichment_{module_name}.csv"
    if not os.path.exists(csv_file):
        print(f"找不到文件: {csv_file}")
        return

    print(f"正在为 {module_name} 绘制气泡图...")
    df = pd.read_csv(csv_file)
    
    # 筛选最显著的 Top 10 GO 和 Top 10 KEGG
    # 区分颜色
    df['type'] = df['Gene_set'].apply(lambda x: 'KEGG' if 'KEGG' in x else 'GO')
    
    # 取 P-value 最小的前 15 个（混合 GO 和 KEGG）
    df = df.sort_values('P-value').head(15).copy()
    
    # 计算 -log10(P-value) 用于展示显著性
    df['log_p'] = -np.log10(df['P-value'])
    
    # 提取 Term 的简短名称 (去掉 GO:xxxxx 等)
    df['Short_Term'] = df['Term'].apply(lambda x: x.split(' (GO')[0] if '(' in x else x)

    # 开始绘图
    plt.figure(figsize=(10, 8))
    
    # 使用 Seaborn 绘制散点图
    # x轴: 显著性, y轴: 通路名称, 大小: 基因重叠数(Overlap), 颜色: 类型
    
    # 处理 Overlap 列 (格式如 "73/97")，提取分子作为大小
    df['Count'] = df['Overlap'].apply(lambda x: int(str(x).split('/')[0]))
    
    sns.scatterplot(data=df, x='log_p', y='Short_Term', 
                    hue='type', size='Count', 
                    sizes=(100, 600), palette='viridis', alpha=0.8)
    
    plt.title(f"{module_name} 功能富集分析 (Top Terms)", fontsize=16)
    plt.xlabel("-Log10(P-value)", fontsize=12)
    plt.ylabel("")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
    plt.tight_layout()
    output_img = f"fixed_plot_{module_name}.png"
    plt.savefig(output_img, dpi=300)
    print(f"图片已修复并保存为: {output_img}")
    plt.show()

# 对三个模块分别绘图
manual_plot("Module_1")
manual_plot("Module_2")
manual_plot("Module_3")