import os
import torch
import pandas as pd
import numpy as np
import argparse
import random
import warnings
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import evaluate # 使用 evaluate 库来计算 ROUGE
from sklearn.metrics.pairwise import cosine_similarity

# 忽略不必要的警告
warnings.filterwarnings("ignore")

# --- 全局常量和配置 ---

# 从用户输入中获取路径
LLM_PATH = "/local/scratch/yxia327/Meta-Llama-3-8B-Instruct-HF"
DATA_PATH = "/local/scratch/yxia327/data/Samsum/train.csv"

# 用于嵌入的模型 (一个轻量级、高性能的模型)
EMBEDDING_MODEL_NAME = '/local/scratch/yxia327/bge-large-en-v1.5'

# --- 1. 模型和数据加载功能 ---

def load_llm_and_tokenizer(model_path):
    """
    加载 LLM 和分词器，使用 4-bit 量化以节省显存。
    """
    print(f"开始加载 LLM 和分词器从: {model_path}...")
    
    # 配置 4-bit 量化
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto", # 自动分配到 GPU
            torch_dtype=torch.bfloat16
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # LLaMA-3 需要一个 pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id

        model.eval() # 设置为评估模式
        print("LLM 和分词器加载成功。")
        return model, tokenizer
    except Exception as e:
        print(f"加载 LLM 失败: {e}")
        print("请确保你的 LLM 路径正确，并且你有足够的 VRAM。")
        exit(1)

def load_data(data_path, num_exemplars, num_test_samples):
    """
    加载 SAMSum 数据集并将其分为 'exemplar_pool' 和 'test_data'。
    """
    print(f"加载数据从: {data_path}...")
    try:
        df = pd.read_csv(data_path, encoding='latin-1')
        # 确保列名正确
        if 'dialogue' not in df.columns or 'summary' not in df.columns:
            print(f"错误: CSV 文件必须包含 'dialogue' 和 'summary' 列。")
            exit(1)
            
        # 清理数据
        df = df.dropna(subset=['dialogue', 'summary'])
        df = df[df['dialogue'].str.strip() != '']
        records = df.to_dict('records')
        
        # 随机打乱
        random.shuffle(records)
        
        # 分割数据
        if len(records) < num_exemplars + num_test_samples:
            print(f"警告: 数据集太小。总共 {len(records)} 条记录。")
            print(f"将使用 {int(len(records)*0.8)} 作为 exemplars, {int(len(records)*0.2)} 作为测试。")
            exemplar_pool = records[:int(len(records)*0.8)]
            test_data = records[int(len(records)*0.8):]
        else:
            exemplar_pool = records[:num_exemplars]
            test_data = records[num_exemplars : num_exemplars + num_test_samples]
            
        print(f"数据加载成功: {len(exemplar_pool)} 条 exemplars, {len(test_data)} 条测试样本。")
        return exemplar_pool, test_data
        
    except FileNotFoundError:
        print(f"错误: 数据文件未找到: {data_path}")
        exit(1)
    except Exception as e:
        print(f"加载数据时出错: {e}")
        exit(1)

def load_embedding_model(model_name):
    """
    加载句子嵌入模型。
    """
    print(f"加载嵌入模型: {model_name}...")
    try:
        model = SentenceTransformer(model_name)
        print("嵌入模型加载成功。")
        return model
    except Exception as e:
        print(f"加载嵌入模型失败: {e}")
        print("请确保你已安装 'sentence-transformers' 并且可以访问 Hugging Face Hub。")
        exit(1)

# --- 2. 提示和文本生成功能 ---

def format_prompt(query_dialogue, exemplars):
    """
    构建 In-Context Learning (ICL) 提示。
    """
    prompt_lines = ["Summarize the following dialogue:\n"]
    for ex in exemplars:
        prompt_lines.append(f"---\nDialogue:\n{ex['dialogue']}\n")
        prompt_lines.append(f"Summary:\n{ex['summary']}\n")
    prompt_lines.append(f"---\nDialogue:\n{query_dialogue}\n")
    prompt_lines.append("Summary:\n")
    return "".join(prompt_lines)

def format_zeroshot_prompt(query_dialogue):
    """
    构建 Zero-Shot 提示 (用于生成候选)。
    """
    prompt_lines = [
        "Summarize the following dialogue:\n",
        f"---\nDialogue:\n{query_dialogue}\n",
        "Summary:\n"
    ]
    return "".join(prompt_lines)

def generate_text(model, tokenizer, prompt, max_new_tokens, diverse=False):
    """
    使用 LLM 生成文本。
    'diverse=True' 用于生成候选，使用采样。
    'diverse=False' 用于教师模型，使用贪婪解码。
    """
    device = model.device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(device)
    
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id
    }
    
    if diverse:
        gen_kwargs.update({
            "do_sample": True,
            "temperature": 1.0
        })
    else:
        gen_kwargs["do_sample"] = False
        
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    
    # 解码并提取新生成的文本
    input_length = inputs.input_ids.shape[1]
    new_tokens = outputs[0, input_length:]
    decoded_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    return decoded_text.strip()

# --- 3. ESA 核心逻辑 ---

def get_embedding(embed_model, text):
    """
    生成文本的 L2 归一化嵌入。
    """
    emb = embed_model.encode(text)
    norm = np.linalg.norm(emb)
    if norm > 0:
        return emb / norm
    return emb

def calculate_sigma(epsilon, delta, N):
    if epsilon == float('inf'):
        return 0.0 # 无噪声
    
    sensitivity = 2.0 / N
    sigma = (sensitivity * np.sqrt(2 * np.log(1.25 / delta))) / epsilon
    return sigma

def run_esa_aggregation(query_dialogue, exemplar_pool, N, k_exemplars, llm, tokenizer, embed_model, epsilon, delta):
    """
    执行 ESA 算法的第 1-8 步 (Alg 2): 生成并聚合带噪声的嵌入。
    """
    
    # 1. 分区 (Partition)
    # 随机选择 N * k 个不重复的 exemplars
    if len(exemplar_pool) < N * k_exemplars:
        print(f"警告: Exemplar 池 ({len(exemplar_pool)}) 不足以创建 {N} 个 {k_exemplars}-shot 子集。")
        print("将使用有放回抽样。")
        all_exemplars = [random.choice(exemplar_pool) for _ in range(N * k_exemplars)]
    else:
        all_exemplars = random.sample(exemplar_pool, N * k_exemplars)
        
    subsets = [all_exemplars[i*k_exemplars:(i+1)*k_exemplars] for i in range(N)]
    
    embeddings = []
    
    # 2-6. 循环、提示、获取输出、嵌入
    for subset in subsets:
        prompt = format_prompt(query_dialogue, subset)
        # diverse=False: 使用贪婪解码获取教师模型的"共识"
        summary_i = generate_text(llm, tokenizer, prompt, max_new_tokens=150, diverse=True)
        emb_i = get_embedding(embed_model, summary_i)
        embeddings.append(emb_i)
        
    # 7. 取平均 (Take the mean)
    mean_emb = np.mean(embeddings, axis=0)
    
    # 8. 添加噪声 (Adding noise)
    sigma = calculate_sigma(epsilon, delta, N)
    if sigma > 0:
        noise = np.random.normal(loc=0.0, scale=sigma, size=mean_emb.shape)
        noisy_mean_emb = mean_emb + noise
    else:
        noisy_mean_emb = mean_emb
        
    return noisy_mean_emb

def get_candidates_and_select(query_dialogue, noisy_mean_emb, num_candidates, llm, tokenizer, embed_model):
    """
    执行 ESA 算法的第 9 步: 生成候选并选择最相似的。
    """
    
    candidates = []
    candidate_embs = []
    
    # 9a. 生成公共候选 (Generate public candidates)
    for _ in range(num_candidates):
        prompt = format_zeroshot_prompt(query_dialogue)
        # diverse=True: 使用采样生成多样化的候选
        candidate_summary = generate_text(llm, tokenizer, prompt, max_new_tokens=150, diverse=True)
        cand_emb = get_embedding(embed_model, candidate_summary)
        
        candidates.append(candidate_summary)
        candidate_embs.append(cand_emb)
        
    # 9b. 选择最相似的 (Select best match)
    if not candidates:
        return "" # 以防万一
        
    # 计算余弦相似度
    noisy_emb_reshaped = noisy_mean_emb.reshape(1, -1)
    embs_array = np.array(candidate_embs)
    
    similarities = cosine_similarity(noisy_emb_reshaped, embs_array)[0]
    
    best_index = np.argmax(similarities)
    return candidates[best_index]

# --- 4. 主实验循环 ---

def main():
    parser = argparse.ArgumentParser(description="复现 DP-ICL ESA 实验")
    parser.add_argument('--num_subsets', type=str, default="2,4,6,8,10,12,14", help="要测试的子集数量 N (逗号分隔)")
    parser.add_argument('--epsilons', type=str, default="1,2,4,8", help="要测试的 Epsilon 值 (逗号分隔)")
    parser.add_argument('--num_exemplars_total', type=int, default=1000, help="用于抽取的 Exemplar 池的总大小")
    parser.add_argument('--num_test_samples', type=int, default=50, help="用于评估的测试样本数量")
    parser.add_argument('--k_exemplars', type=int, default=1, help="每个子集中的 exemplar 数量 (k-shot)")
    parser.add_argument('--num_candidates', type=int, default=10, help="为最终选择生成的 zero-shot 候选数量")
    parser.add_argument('--delta', type=float, default=1e-5, help="差分隐私的 delta 值 (与 SAMSum 论文一致)")

    args = parser.parse_args()

    # 解析列表参数
    list_num_subsets = [int(n) for n in args.num_subsets.split(',')]
    list_epsilons = [float(e) if e != 'inf' else float('inf') for e in args.epsilons.split(',')]

    print("--- 实验配置 ---")
    print(f"LLM 路径: {LLM_PATH}")
    print(f"数据路径: {DATA_PATH}")
    print(f"测试的 N (子集数): {list_num_subsets}")
    print(f"测试的 Epsilon: {list_epsilons}")
    print(f"Delta: {args.delta}")
    print(f"K-shot (每个子集): {args.k_exemplars}")
    print(f"Exemplar 池大小: {args.num_exemplars_total}")
    print(f"测试样本数: {args.num_test_samples}")
    print(f"候选生成数: {args.num_candidates}")
    print("--------------------")

    # 1. 加载所有模型和数据
    llm, tokenizer = load_llm_and_tokenizer(LLM_PATH)
    embed_model = load_embedding_model(EMBEDDING_MODEL_NAME)
    exemplar_pool, test_data = load_data(DATA_PATH, args.num_exemplars_total, args.num_test_samples)
    rouge = evaluate.load('rouge')

    results = []

    # 2. 主循环
    for N in list_num_subsets:
        for epsilon in list_epsilons:
            
            print(f"\n--- 正在运行: N = {N}, Epsilon = {epsilon} ---")
            
            predictions = []
            references = []
            
            pbar = tqdm(test_data, desc=f"N={N}, Eps={epsilon}")
            for sample in pbar:
                query_dialogue = sample['dialogue']
                reference_summary = sample['summary']
                
                # 步骤 1-8: 聚合
                noisy_mean_emb = run_esa_aggregation(
                    query_dialogue, exemplar_pool, N, args.k_exemplars,
                    llm, tokenizer, embed_model, epsilon, args.delta
                )
                
                # 步骤 9: 选择
                final_summary = get_candidates_and_select(
                    query_dialogue, noisy_mean_emb, args.num_candidates,
                    llm, tokenizer, embed_model
                )
                
                predictions.append(final_summary)
                references.append(reference_summary)

            # 3. 计算 Utility (ROUGE)
            try:
                rouge_scores = rouge.compute(predictions=predictions, references=references)
                rouge_l_f1 = rouge_scores['rougeL'] # ROUGE-L F1-score
            except Exception as e:
                print(f"计算 ROUGE 时出错: {e}")
                rouge_l_f1 = 0.0

            print(f"--- 结果: N = {N}, Epsilon = {epsilon} ---")
            print(f"ROUGE-L (F1-score): {rouge_l_f1:.4f}")
            
            results.append({
                "N (Subsets)": N,
                "Epsilon": epsilon,
                "ROUGE-L": rouge_l_f1
            })

    # 4. 打印最终报告
    print("\n\n--- 实验总结 ---")
    results_df = pd.DataFrame(results)
    
    # 将结果数据透视，以便更清晰地查看
    try:
        pivot_df = results_df.pivot(index="N (Subsets)", columns="Epsilon", values="ROUGE-L")
        pivot_df.columns.name = "Epsilon"
        print(pivot_df.to_string(float_format="%.4f"))
    except Exception as e:
        print(f"无法创建数据透视表 ({e}), 打印原始结果:")
        print(results_df.to_string(index=False))

if __name__ == "__main__":
    main()
