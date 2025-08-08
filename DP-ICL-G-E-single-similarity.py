import dspy
import numpy as np
import math
import pdb
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
import warnings
import re
from collections import Counter
from scipy.stats import beta

# 忽略来自huggingface_hub的特定版本警告
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")
output_file = open('DP-ICL-G-E-single_MIA_Sim.txt', 'w') # 使用新的输出文件名
# ==============================================================================
# 1. DSPy和模型设置
# ==============================================================================
print("正在初始化模型...",file=output_file)
# 用于生成摘要和进行MIA判断的LLM
generator_lm = dspy.LM("ollama_chat/llama3.1:8b",
                       base_url="http://localhost:11434",
                       temperature=0.5)

# 用于将文本转换为嵌入向量的模型
embedding_model = SentenceTransformer('/mnt/8t_data/yuyang/bge-large-en-v1.5', trust_remote_code=True)

dspy.configure(lm=generator_lm)


# ==============================================================================
# 2. DP-ICL 框架实现 (这部分未作修改)
# ==============================================================================
class DP_ICL:
    def __init__(self, llm, embed_model):
        self.llm = llm
        self.embed_model = embed_model
        self.qa_signature = dspy.Signature("context, question -> answer")

    def _get_llm_responses(self, question, exemplar_sets):
        responses = []
        for i, exemplars in enumerate(exemplar_sets):
            context_str = [val for ex in exemplars for val in ex.get('ocr_tokens','')]
            context_str = " ".join(context_str) if isinstance(context_str, list) else context_str
            context_str = context_str.strip()
            context_str = "context: " + context_str if context_str else "context: No relevant context provided."
            with dspy.context(lm=self.llm):
                result = dspy.Predict(self.qa_signature)(context=context_str, question=question)
                responses.append(result.answer)
        return responses

    def _embedding_space_aggregation(self, responses, epsilon, delta, num_candidates=10, original_question=""):
        response_embeddings = self.embed_model.encode(responses, normalize_embeddings=True)
        mean_embedding = np.mean(response_embeddings, axis=0)
        sigma = (math.sqrt(2 * math.log(1.25 / delta))) / epsilon
        noise = np.random.normal(0, sigma, mean_embedding.shape)
        private_mean_embedding = mean_embedding + noise
        
        candidate_signature = dspy.Signature("question -> answer")
        candidates = []
        # print("Candidates generated:", file=output_file) # 可以取消注释来观察
        with dspy.context(lm=self.llm):
            for i in range(num_candidates):
                diversified_question = f"{original_question}\n\n---\n(Variation {i+1})"
                result = dspy.Predict(candidate_signature)(question=diversified_question)
                candidates.append(result.answer)
                # print(f"  - {result.answer}...", file=output_file)
            
        candidate_embeddings = self.embed_model.encode(candidates, normalize_embeddings=True)
        similarities = [1 - cosine(private_mean_embedding, emb) for emb in candidate_embeddings]
        best_candidate_index = np.argmax(similarities)
        final_answer = candidates[best_candidate_index]
        return final_answer

    def _keyword_space_aggregation_ptr(self, responses, question, top_k=10, epsilon=3.0, delta=1e-5):
        all_words = []
        for res in responses:
            words = re.findall(r'\b\w+\b', res.lower())
            all_words.extend(list(set(words)))
        word_counts = Counter(all_words)
        if not word_counts or len(word_counts) <= top_k:
            with dspy.context(lm=self.llm):
                 return dspy.Predict(dspy.Signature("question -> answer"))(question=question).answer

        sorted_counts = word_counts.most_common()
        H_k = sorted_counts[top_k-1][1]
        H_k_plus_1 = sorted_counts[top_k][1]
        gap = H_k - H_k_plus_1
        
        sensitivity = 2
        scale = sensitivity / epsilon
        noise = np.random.laplace(0, scale)
        noisy_gap = gap + noise
        
        if noisy_gap > 2:
            top_keywords = [item[0] for item in sorted_counts[:top_k]]
        else:
            with dspy.context(lm=self.llm):
                return dspy.Predict(dspy.Signature("question -> answer"))(question=question).answer

        reconstruction_prompt = f"Please answer the question based on potential context. Use the following suggested words to guide your answer: {', '.join(top_keywords)}."
        reconstruction_signature = dspy.Signature("guidance, question -> answer")
        with dspy.context(lm=self.llm):
            final_answer = dspy.Predict(reconstruction_signature)(guidance=reconstruction_prompt, question=question).answer
        return final_answer

    def predict(self, sensitive_data, question, method='esa', num_partitions=10, subsample_rate=0.3, epsilon=3.0, delta=1e-5):
        subsampled_data = [item for item in sensitive_data if np.random.rand() < subsample_rate]
        if len(subsampled_data) < num_partitions:
             with dspy.context(lm=self.llm):
                 return dspy.Predict(dspy.Signature("question -> answer"))(question=question).answer
        
        exemplar_sets = np.array_split(subsampled_data, num_partitions)
        responses = self._get_llm_responses(question, exemplar_sets)
        
        if method == 'esa':
            return self._embedding_space_aggregation(responses, epsilon, delta, original_question=question)
        elif method == 'ksa':
            return self._keyword_space_aggregation_ptr(responses, question, epsilon=epsilon, delta=delta)
        else:
            raise ValueError("方法必须是 'esa' 或 'ksa'")


# ==============================================================================
# 3. 成员推断攻击（MIA）审计实现【修改后的版本】
# ==============================================================================

# <--- MODIFIED SECTION START --->

def calculate_clopper_pearson_interval(k, n, alpha=0.05):
    """
    计算Clopper-Pearson置信区间。 (此函数未变，保留使用)
    :param k: 成功次数
    :param n: 总试验次数
    :param alpha: 显著性水平 (例如 0.05 对应 95% 置信度)
    :return: (下界, 上界)
    """
    if n == 0: return (0, 1)
    lower_bound = beta.ppf(alpha / 2, k, n - k + 1)
    upper_bound = beta.ppf(1 - alpha / 2, k + 1, n - k)
    if k == 0: lower_bound = 0
    if k == n: upper_bound = 1
    return (lower_bound, upper_bound)

def audit_esa_by_semantic_similarity(dp_framework, sensitive_db, target_record, question, 
                                     num_trials=50, epsilon=3.0, delta=1e-5, similarity_threshold=0.75):
    """
    使用输出与目标答案的语义相似度来对ESA进行MIA审计。
    """
    print("\n" + "="*60, file=output_file)
    print("开始使用语义相似度法对ESA进行MIA审计", file=output_file)
    print(f"参数: num_trials={num_trials}, epsilon={epsilon}, delta={delta}, similarity_threshold={similarity_threshold}", file=output_file)
    print("="*60, file=output_file)

    # --- 1. 准备数据集和攻击者的“地标” ---
    exemplars_with_target = sensitive_db + [target_record]
    exemplars_without_target = sensitive_db
    
    # 攻击者知道目标答案，并将其编码为“地标”向量
    ground_truth_answer = target_record['answers']
    ground_truth_embedding = dp_framework.embed_model.encode(ground_truth_answer, normalize_embeddings=True)
    print(f"攻击者使用此答案作为判断基准: '{ground_truth_answer[:100]}...'", file=output_file)

    # --- 2. 审计运行阶段 ---
    print("\n开始运行审计试验...", file=output_file)
    true_positives = 0  # 目标存在时，攻击者猜对的次数 (TP)
    false_positives = 0 # 目标不存在时，攻击者猜对的次数 (FP)

    for i in range(num_trials):
        # 场景1: 目标记录在数据集中 (D_in)
        private_answer_with = dp_framework.predict(
            sensitive_data=exemplars_with_target, question=question, method='esa',
            num_partitions=2, subsample_rate=1, epsilon=epsilon, delta=delta
        )
        embedding_with = dp_framework.embed_model.encode(private_answer_with, normalize_embeddings=True)
        # 计算与基准答案的相似度
        similarity_with = 1 - cosine(embedding_with, ground_truth_embedding)
        
        # 应用攻击策略：如果相似度超过阈值，攻击者就猜测“目标在数据集中”
        if similarity_with >= similarity_threshold:
            true_positives += 1

        # 场景2: 目标记录不在数据集中 (D_out)
        private_answer_without = dp_framework.predict(
            sensitive_data=exemplars_without_target, question=question, method='esa',
            num_partitions=2, subsample_rate=1, epsilon=epsilon, delta=delta
        )
        embedding_without = dp_framework.embed_model.encode(private_answer_without, normalize_embeddings=True)
        # 计算与基准答案的相似度
        similarity_without = 1 - cosine(embedding_without, ground_truth_embedding)
        
        # 应用攻击策略
        if similarity_without >= similarity_threshold:
            false_positives += 1
            
        print(f"审计试验 {i+1}/{num_trials} 完成。相似度 (With/Without): {similarity_with:.4f} / {similarity_without:.4f}", file=output_file)
        # print(f"  - With Target: {private_answer_with[:80]}...", file=output_file)
        # print(f"  - Without Target: {private_answer_without[:80]}...", file=output_file)

    # --- 3. 结果分析 ---
    print("\n" + "="*60, file=output_file)
    print("审计结果分析", file=output_file)
    print("="*60, file=output_file)
    
    # 计算攻击准确率
    tpr = true_positives / num_trials  # True Positive Rate
    fpr = false_positives / num_trials # False Positive Rate
    
    # 计算置信区间
    tpr_interval = calculate_clopper_pearson_interval(true_positives, num_trials)
    fpr_interval = calculate_clopper_pearson_interval(false_positives, num_trials)

    print(f"攻击阈值 (Similarity Threshold): {similarity_threshold}", file=output_file)
    print(f"攻击准确率 (TPR - 目标存在时): {tpr:.2%} ({true_positives}/{num_trials})", file=output_file)
    print(f"  - 95% 置信区间 (Clopper-Pearson): [{tpr_interval[0]:.2%}, {tpr_interval[1]:.2%}]", file=output_file)
    print(f"攻击误报率 (FPR - 目标不存在时): {fpr:.2%} ({false_positives}/{num_trials})", file=output_file)
    print(f"  - 95% 置信区间 (Clopper-Pearson): [{fpr_interval[0]:.2%}, {fpr_interval[1]:.2%}]", file=output_file)
    print("-" * 60, file=output_file)

    # 计算经验性Epsilon
    # 为避免除以零，对fpr_interval[1]进行处理
    if tpr_interval[0] > 0 and fpr_interval[1] > 0:
        empirical_epsilon = math.log(tpr_interval[0] / fpr_interval[1])
        print(f"计算出的经验性 ε (log(lower_bound_tpr / upper_bound_fpr)): {empirical_epsilon:.4f}", file=output_file)
    elif tpr_interval[0] > 0 and fpr_interval[1] == 0:
        # 如果FPR上限为0，意味着区分能力非常强，经验epsilon趋于无穷
        print("经验性 ε: inf (FPR极低，攻击者可以完美区分)", file=output_file)
    else:
        print("经验性 ε: N/A (TPR为0或FPR过高，攻击无效或无法计算)", file=output_file)

    # 结论解读
    advantage = tpr - fpr
    print(f"攻击者优势 (Advantage = TPR - FPR): {advantage:.2%}", file=output_file)
    if advantage > 0.1: # 优势大于10%可以认为存在明显泄露
        print("[结论] 存在明显的隐私泄露风险。", file=output_file)
    else:
        print("[结论] 隐私保护似乎有效，攻击者优势不明显。", file=output_file)

# <--- MODIFIED SECTION END --->


if __name__ == '__main__':
    # --- 1. 数据准备 ---
    print("准备审计所需数据...", file=output_file)
    filepath = "/mnt/8t_data/yuyang/PFL-DocVQA_imdbs/imdb_train_client_0.npy"
    all_data = np.load(filepath, allow_pickle=True)
    
    # 定义目标记录、敏感数据库和查询
    target_record = all_data[1]
    target_record['question'] = "What happened in Paris during WW 2?"
    target_record['answers'] = "Paris was a major hub of resistance against Nazi occupation during World War II."
    target_record['ocr_tokens'] = ['Paris was a major hub of resistance against Nazi occupation during World War II.', "The city's intellectuals, artists, and politicians played a significant role in organizing and participating in the French Resistance."]
    sensitive_database = list(all_data[2:6]) # 基础数据库
    user_question = target_record['question']

    print(f"目标记录 Question: {target_record['question'][:80]}...", file=output_file)
    print(f"敏感数据库大小: {len(sensitive_database)}", file=output_file)
    print(f"审计查询 Question: {user_question[:80]}...", file=output_file)

    # --- 2. 初始化DP-ICL框架 ---
    dp_icl_framework = DP_ICL(llm=generator_lm, embed_model=embedding_model)

    # --- 3. 运行MIA审计 ---
    # <--- MODIFIED SECTION START --->
    # 调用新的、基于语义相似度的审计函数
    audit_esa_by_semantic_similarity(
        dp_framework=dp_icl_framework,
        sensitive_db=sensitive_database,
        target_record=target_record,
        question=user_question,
        num_trials=2000,  # 建议至少20次以获得初步统计意义，100次更佳
        epsilon=3.0,    # 理论epsilon
        delta=1e-5,
        similarity_threshold=0.8 # 您设定的攻击阈值
    )
    # <--- MODIFIED SECTION END --->