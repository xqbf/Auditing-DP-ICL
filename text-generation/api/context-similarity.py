import os
import csv
from sentence_transformers import SentenceTransformer, util
import torch

# 确保文件路径正确
csv_file_path = "/local/scratch/yxia327/data/ag_news_csv/test.csv"

try:
    with open(csv_file_path, 'r', encoding='utf-8') as f:
        # 假设需要比较的文本在CSV的第一列，我们直接在读取时提取
        all_data = [(r[0],r[1],r[2]) for r in csv.reader(f)][1:] # 直接提取每行的第一个元素作为文本
except FileNotFoundError:
    print(f"Error: `{csv_file_path}` not found.")
    exit()
except IndexError:
    print(f"Error: CSV file `{csv_file_path}` seems to be empty or has an incorrect format.")
    exit()

# --- 主要修改部分 (来自您的代码) ---
# 在加载任何Hugging Face模型之前，设置缓存目录的环境变量
# 将路径替换为你指定的路径
model_path = "/local/scratch/yxia327/bge-large-en-v1.5"
os.environ['HF_HOME'] = model_path
os.environ['HUGGINGFACE_HUB_CACHE'] = model_path
# --------------------

# 1. 选择并加载一个针对英文的预训练模型
# sentence-transformers库会自动读取上面的环境变量，并将模型下载到指定目录
model_name = 'BAAI/bge-large-en-v1.5'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 增加模型加载的try-except，确保模型路径有效
try:
    model = SentenceTransformer(model_name, device=device)
    print(f"Model '{model_name}' has been loaded to '{device}'.")
    # 打印确认缓存路径
    print(f"Model files are expected to be in: {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    print(f"Please ensure the model '{model_name}' exists at the path '{model_path}'.")
    exit()


# 2. 准备你的英文文本数据
# base_context = "At Microsoft Education, we believe every student deserves the opportunity to thrive. Make math accessible for learners with powerful, inclusive tools designed to unlock potential and build confidence to make math fun."
base_context = 'The old clock chimed a forgotten, dusty tune.'

# base_context = "Real Madrid has signed a new player."

# 从已加载的数据中选择一部分进行比较
other_contexts = ["FOXBOROUGH -- Looking at his ridiculously developed upper body, with huge biceps and hardly an ounce of fat."] # 选择前30条文本

# 3. 将所有文本转换为Embedding向量
print("\nEncoding texts...")
base_embedding = model.encode(base_context, convert_to_tensor=True)
other_embeddings = model.encode(other_contexts, convert_to_tensor=True)
print("Encoding complete.")

# 4. 计算余弦相似度
cosine_scores = util.cos_sim(base_embedding, other_embeddings)

# 5. 输出结果
print(f"\nSemantic similarity with '{base_context}':")
for i, context in enumerate(other_contexts):
    score = cosine_scores[0][i].item()
    # 为了终端输出整洁，可以只打印部分文本
    print(f"- Similarity with '{context[:50]}...': {score:.4f}")

# --- 以下是为计算平均相似度新增的代码 ---

# 6. 计算所有相似度分数的平均值
# cosine_scores 是一个 [1, N] 的张量，直接调用 mean() 即可
average_score = torch.mean(cosine_scores).item()

# 7. 打印平均相似度
print("\n" + "="*30)
print(f"Average Cosine Similarity: {average_score:.4f}")
print("="*30)