import pandas as pd
import gseapy as gp
import os
import sys

# 设置风格
print("="*60)

# ==========================================
# 1. 读取基因列表
# ==========================================
filename = 'string_interactions_short.tsv'
print(f"正在从网络文件 {filename} 中提取基因列表...")

try:
    df = pd.read_csv(filename, sep='\t', header=None)
    # 提取所有唯一的基因名 (node1 和 node2)
    genes = pd.unique(df[[0, 1]].values.ravel('K'))
    # 过滤掉可能的表头
    genes = [g for g in genes if g not in ['node1', '#node1', 'node2']]
    
    print(f"-> 提取成功！网络中共有 {len(genes)} 个唯一基因。")
    # 举例打印几个看看
    print(f"-> 待搜索基因示例: {genes[:5]} ...")

except Exception as e:
    print(f"读取失败: {e}")
    sys.exit(1)

# ==========================================
# 2. 定义要搜索的数据库
# ==========================================
# 通常搜这两个最经典的库
gene_sets = [
    'GO_Biological_Process_2021', # GO 生物学过程
    'KEGG_2021_Human'             # KEGG 信号通路
]

print(f"\n准备在以下数据库中搜索注释信息: {gene_sets}")

# ==========================================
# 3. 执行富集分析 (Enrichment Analysis)
# ==========================================
print("正在连接 Enrichr API 进行批量搜索 (请保持联网)...")

try:
    # 调用 gseapy 的 enrichr 方法
    enr = gp.enrichr(
        gene_list=genes,           # 1009 个基因
        gene_sets=gene_sets,       # 搜 GO 和 KEGG
        organism='Human',          # 物种：人类
        outdir=None                # 不直接保存，自己处理
    )
    
    results = enr.results
    
    # 筛选显著的结果 (P-value < 0.05)
    sig_results = results[results['Adjusted P-value'] < 0.05].copy()
    
    print(f"-> 搜索完成！共找到 {len(sig_results)} 条显著的富集通路。")
    
    # ==========================================
    # 4. 保存为标准格式供后续画图使用
    # ==========================================
    output_file = "enrichment_All_Genes.csv"
    sig_results.to_csv(output_file, index=False)
    print(f"数据已保存至: {output_file}")
    
    # 打印前几条看看
    print("\nTop 5 富集结果:")
    print(sig_results[['Term', 'Adjusted P-value', 'Overlap']].head(5))

except Exception as e:
    print(f"搜索失败 (可能是网络原因): {e}")

print("="*60)