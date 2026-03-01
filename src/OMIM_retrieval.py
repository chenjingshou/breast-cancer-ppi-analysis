import mygene
import pandas as pd
import sys

# 设置显示选项，防止打印时列对不齐
pd.set_option('display.unicode.east_asian_width', True)

print("="*60)
print("种子基因信息验证程序 (MyGene Info)")
print("="*60)

# 1. 定义种子基因列表
seed_genes = [
    'TP53', 'BRCA1', 'BRCA2', 'PTEN', 'PIK3CA', 
    'ATM', 'CCND1', 'ERBB2', 'MYC', 'EGFR'
]
print(f"正在检索以下 {len(seed_genes)} 个种子基因的详细信息...")

try:
    # 2. 初始化查询对象
    mg = mygene.MyGeneInfo()

    # 3. 批量查询
    results = mg.querymany(seed_genes, scopes='symbol', fields='name,summary,type_of_gene', species='human')

    # 4. 整理结果为 DataFrame
    data = []
    for res in results:
        if 'notfound' not in res:
            data.append({
                'Symbol': res['query'],
                'Gene Name': res.get('name', 'N/A'),
                'Description': res.get('summary', 'No summary available')[:100] + '...' 
            })

    if not data:
        print("错误：未获取到任何数据，请检查网络连接。")
        sys.exit(1)

    #直接从 list of dicts 创建 DataFrame
    df_seeds = pd.DataFrame(data)

    # 5. 输出显示
    print("\n检索成功！种子基因元数据如下：")
    print("-" * 80)
    print(df_seeds[['Symbol', 'Gene Name']].to_string(index=False))
    print("-" * 80)

    # 保存
    df_seeds.to_csv('seed_genes_info.csv', index=False)
    print("基因信息已保存至: seed_genes_info.csv")

except Exception as e:
    print(f"\n发生未知错误: {e}")