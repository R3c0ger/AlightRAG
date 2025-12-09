import ast

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.metrics import context_recall, answer_correctness, answer_similarity, faithfulness

from datasets import Dataset

# 1. 读取 JSONL 文件并处理数据
data_samples = {
    'id': [],
    'question': [],
    'contexts': [],  # 注意：Ragas 要求这里是字符串列表 list[str]
    'answer': [],  # 对应 lightrag_answer
    'ground_truth': []  # 对应 ground_truth
}

file_path = './eval_iter2.jsonl'

with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            item = ast.literal_eval(line)  # <-- 使用这行
        except (ValueError, SyntaxError):
            print(f"跳过无法解析的行: {line[:50]}...")
            continue
        # 填充数据
        data_samples['id'].append(item.get('id', ''))
        data_samples['question'].append(item.get('question', ''))
        data_samples['ground_truth'].append(item.get('ground_truth', ''))
        # data_samples['answer'].append(item.get('lightrag_answer', ''))
        ans = item.get('alightrag_answer', '')

        ctx = item.get('context', '')
        if isinstance(ctx, str):
            data_samples['contexts'].append([ctx])  # 如果是字符串，转为列表
        elif isinstance(ctx, list):
            data_samples['contexts'].append(ctx)
        else:
            data_samples['contexts'].append([""])

# 2. 转换为 HuggingFace Dataset 对象
dataset = Dataset.from_dict(data_samples)

# 3. 配置评估指标
# context_recall: 衡量 ground_truth 是否在 context 中 (你要求的比较)
# context_precision: 衡量 context 是否与 question 相关 (通常一起使用)
metrics = [context_recall, faithfulness, answer_correctness, answer_similarity]

my_llm = ChatOpenAI(
    model="deepseek-chat",  # 或是 "moonshot-v1-8k" 等
    openai_api_key="sk-xx",
    openai_api_base="https://api.deepseek.com/v1", # 替换为对应厂商的 Base URL
    temperature=0
)

silicon_embeddings = OpenAIEmbeddings(
    # 替换为你想要使用的 SiliconFlow 上的 Embedding 模型名称
    # 例如: "BAAI/bge-m3", "BAAI/bge-large-zh-v1.5", "pro/BAAI/bge-m3" 等
    model="BAAI/bge-m3",

    # SiliconFlow 的 Base URL
    openai_api_base="https://api.siliconflow.cn/v1",

    # 你的 SiliconFlow API Key
    openai_api_key="sk-xx",
    check_embedding_ctx_length=False
)

# 4. 运行评估
print("开始评估...")
results = evaluate(
    dataset=dataset,
    metrics=metrics,
    llm=my_llm,
    embeddings=silicon_embeddings
)

# 5. 输出结果
print("\n评估结果:")
print(results)

# 导出为 Pandas DataFrame 查看详情
df_results = results.to_pandas()
print("\n详细数据预览:")
print(df_results[['question', 'context_recall', 'context_precision']].head())

# 保存结果
df_results.to_csv('./ragas_evaluation_results_reflect.csv', index=False)
