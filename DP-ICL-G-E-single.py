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
output_file = open('DP-ICL-G-E-single.txt', 'w')
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
# 2. DP-ICL 框架实现 (与您提供的代码相同，未作修改)
# ==============================================================================
class DP_ICL:
    def __init__(self, llm, embed_model):
        self.llm = llm
        self.embed_model = embed_model
        self.qa_signature = dspy.Signature("context, question -> answer")

    def _get_llm_responses(self, question, exemplar_sets):
        responses = []
        # print(f"正在为 {len(exemplar_sets)} 个典范集生成回答...")
        for i, exemplars in enumerate(exemplar_sets):
            
            # context_str = "\n\n".join([f"Q: {ex.get('question', '')}\nA: {ex.get('answers', '')[0] if isinstance(ex.get('answers'), list) and ex.get('answers') else ex.get('answers', '')}" for ex in exemplars_with_target])

            context_str = [val for ex in exemplars for val in ex.get('ocr_tokens','')]
            context_str = " ".join(context_str) if isinstance(context_str, list) else context_str
            context_str = context_str.strip()
            context_str = "context: " + context_str if context_str else "context: No relevant context provided."
            # pdb.set_trace()  # 调试断点
            with dspy.context(lm=self.llm):
                result = dspy.Predict(self.qa_signature)(context=context_str, question=question)
                responses.append(result.answer)
            # print(f"  已生成回答 {i+1}/{len(exemplar_sets)}")
        return responses

    def _embedding_space_aggregation(self, responses, epsilon, delta, num_candidates=10, original_question=""):
        # print("\n--- 开始执行嵌入空间聚合 (ESA) ---")
        response_embeddings = self.embed_model.encode(responses, normalize_embeddings=True)
        mean_embedding = np.mean(response_embeddings, axis=0)
        sigma = (math.sqrt(2 * math.log(1.25 / delta))) / epsilon
        noise = np.random.normal(0, sigma, mean_embedding.shape)
        private_mean_embedding = mean_embedding + noise
        
        candidate_signature = dspy.Signature("question -> answer")
        candidates = []
        print("Candidates generated:", file=output_file)
        with dspy.context(lm=self.llm):
            for _ in range(num_candidates):
                result = dspy.Predict(candidate_signature)(question=original_question)
                candidates.append(result.answer)
                print(f"  - {result.answer}...", file=output_file)
            
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
# 3. 成员推断攻击（MIA）审计实现【新增部分】
# ==============================================================================

def create_reference_vector(llm, embed_model, question, exemplars_with_target, exemplars_without_target, L=20):
    """
    (攻击者准备阶段) 创建参考向量，用于区分包含目标和不包含目标的输出。
    这个过程模拟攻击者的准备阶段，不使用DP机制，直接调用基础LLM。
    """
    print("正在创建参考向量 (攻击者设置阶段)...")
    
    predictor = dspy.Predict(dspy.Signature("context, question -> answer"))
    
    embeddings_with = []
    embeddings_without = []

    # 格式化上下文
    # context_with_str = "\n\n".join([f"Q: {ex.get('question', '')}\nA: {ex.get('answers', '')[0] if isinstance(ex.get('answers'), list) and ex.get('answers') else ex.get('answers', '')}" for ex in exemplars_with_target])
    # context_without_str = "\n\n".join([f"Q: {ex.get('question', '')}\nA: {ex.get('answers', '')[0] if isinstance(ex.get('answers'), list) and ex.get('answers') else ex.get('answers', '')}" for ex in exemplars_without_target])
    context_with_str = [val for ex in exemplars_with_target for val in ex.get('ocr_tokens','')]
    context_with_str = " ".join(context_with_str) if isinstance(context_with_str, list) else context_with_str
    context_with_str = context_with_str.strip()
    context_with_str = "context: " + context_with_str if context_with_str else "context: No relevant context provided."

    context_without_str = [val for ex in exemplars_without_target for val in ex.get('ocr_tokens','')]
    context_without_str = " ".join(context_without_str) if isinstance(context_without_str, list) else context_without_str
    context_without_str = context_without_str.strip()
    context_without_str = "context: " + context_without_str if context_without_str else "context: No relevant context provided."

    for i in range(L):
        # 场景1: 目标在数据集中
        with dspy.context(lm=llm):
            response_with = predictor(context=context_with_str, question=question)
            embeddings_with.append(embed_model.encode(response_with.answer, normalize_embeddings=True))

        # 场景2: 目标不在数据集中
        with dspy.context(lm=llm):
            response_without = predictor(context=context_without_str, question=question)
            embeddings_without.append(embed_model.encode(response_without.answer, normalize_embeddings=True))
        print(f"  正在生成参考样本 {i+1}/{L}",file=output_file)

        # pdb.set_trace()  # 调试断点


    mean_vec_with = np.mean(embeddings_with, axis=0)
    mean_vec_without = np.mean(embeddings_without, axis=0)
    
    v_ref = mean_vec_with - mean_vec_without
    v_ref = v_ref / np.linalg.norm(v_ref)
    
    print("参考向量创建完毕。",file=output_file)
    return v_ref

def calculate_clopper_pearson_interval(k, n, alpha=0.05):
    """
    计算Clopper-Pearson置信区间。
    :param k: 成功次数
    :param n: 总试验次数
    :param alpha: 显著性水平 (例如 0.05 对应 95% 置信度)
    :return: (下界, 上界)
    """
    if n == 0: return (0, 1)
    lower_bound = beta.ppf(alpha / 2, k, n - k + 1)
    upper_bound = beta.ppf(1 - alpha / 2, k + 1, n - k)
    # 处理 k=0 或 k=n 的边界情况
    if k == 0: lower_bound = 0
    if k == n: upper_bound = 1
    return (lower_bound, upper_bound)

def audit_esa_with_reference_vector(dp_framework, sensitive_db, target_record, question, num_trials=50, epsilon=3.0, delta=1e-5):
    """
    使用参考向量法对ESA进行完整的MIA审计。
    """
    print("\n" + "="*60,file=output_file)
    print("开始使用参考向量法对ESA进行MIA审计",file=output_file)
    print(f"参数: num_trials={num_trials}, epsilon={epsilon}, delta={delta}",file=output_file)
    print("="*60,file=output_file)

    # --- 1. 准备数据集 ---
    exemplars_with_target = sensitive_db + [target_record]
    exemplars_without_target = sensitive_db
    
    # --- 2. 攻击者准备阶段 ---
    v_ref = create_reference_vector(
        dp_framework.llm, dp_framework.embed_model, question,
        exemplars_with_target, exemplars_without_target
    )

    # --- 3. 审计运行阶段 ---
    print("\n开始运行审计试验...",file=output_file)
    scores_with_target = []
    scores_without_target = []

    # pdb.set_trace()  # 调试断点

    for i in range(num_trials):
        # 场景1: 目标记录在数据集中 (D_in)
        private_answer_with = dp_framework.predict(
            sensitive_data=exemplars_with_target, question=question, method='esa',
            num_partitions=3, subsample_rate=1, epsilon=epsilon, delta=delta
        )
        embedding_with = dp_framework.embed_model.encode(private_answer_with, normalize_embeddings=True)
        scores_with_target.append(np.dot(embedding_with, v_ref))

        # 场景2: 目标记录不在数据集中 (D_out)
        private_answer_without = dp_framework.predict(
            sensitive_data=exemplars_without_target, question=question, method='esa',
            num_partitions=3, subsample_rate=1, epsilon=epsilon, delta=delta
        )
        embedding_without = dp_framework.embed_model.encode(private_answer_without, normalize_embeddings=True)
        print(f"审计试验 {i+1}/{num_trials} 结果：",file=output_file)

        print("With target: ", private_answer_with,file=output_file)  # 打印前80个字符以检查答案
        print("Without target", private_answer_without,file=output_file)  # 打印前80个字符以检查答案

        scores_without_target.append(np.dot(embedding_without, v_ref))
        # pdb.set_trace()  # 调试断点
        print(f"  审计试验 {i+1}/{num_trials} 完成。",file=output_file)

    # --- 4. 结果分析 ---
    print("\n" + "="*60,file=output_file)
    print("审计结果分析",file=output_file)
    print("="*60,file=output_file)
    # pdb.set_trace()  # 调试断点
    threshold = (np.mean(scores_with_target) + np.mean(scores_without_target)) / 2
    
    # 计算攻击准确率
    # TPR (True Positive Rate): 目标存在时，攻击者猜对的概率
    tp = sum(1 for s in scores_with_target if s >= threshold)
    tpr = tp / num_trials
    
    # FPR (False Positive Rate): 目标不存在时，攻击者猜错的概率（误报）
    fp = sum(1 for s in scores_without_target if s >= threshold)
    fpr = fp / num_trials
    # pdb.set_trace()  # 调试断点
    # 计算置信区间
    tpr_interval = calculate_clopper_pearson_interval(tp, num_trials)
    fpr_interval = calculate_clopper_pearson_interval(fp, num_trials)

    print(f"区分阈值 (Threshold): {threshold:.4f}",file=output_file)
    print(f"攻击准确率 (TPR - 目标存在时): {tpr:.2%} ({tp}/{num_trials})",file=output_file)
    print(f"  - 95% 置信区间 (Clopper-Pearson): [{tpr_interval[0]:.2%}, {tpr_interval[1]:.2%}]",file=output_file)
    print(f"攻击准确率 (FPR - 目标不存在时): {fpr:.2%} ({fp}/{num_trials})",file=output_file)
    print(f"  - 95% 置信区间 (Clopper-Pearson): [{fpr_interval[0]:.2%}, {fpr_interval[1]:.2%}]",file=output_file)
    print("-" * 60,file=output_file)

    # 计算经验性Epsilon
    if tpr > 0 and fpr > 0:
        empirical_epsilon = math.log(tpr_interval[0] / fpr_interval[1])
        print(f"计算出的经验性 ε (log(TPR/FPR)): {empirical_epsilon:.4f}",file=output_file)
    elif tpr > 0 and fpr == 0:
        # 如果FPR为0，经验性epsilon理论上为无穷大，表示完美的区分能力
        print("经验性 ε: inf (FPR为0，攻击者可以完美区分)",file=output_file)
    else:
        # 如果TPR为0，无法计算有意义的epsilon
        print("经验性 ε: N/A (TPR为0，攻击无效)",file=output_file)

    # 结论解读
    advantage = tpr - fpr
    print(f"攻击者优势 (Advantage = TPR - FPR): {advantage:.2%}",file=output_file)
    if advantage > 0.1: # 优势大于10%可以认为存在明显泄露
        print("[结论] 存在明显的隐私泄露风险。",file=output_file)
    else:
        print("[结论] 隐私保护似乎有效。",file=output_file)


if __name__ == '__main__':
    # --- 1. 数据准备 ---
    print("准备审计所需数据...",file=output_file)
    filepath = "/mnt/8t_data/yuyang/PFL-DocVQA_imdbs/imdb_train_client_0.npy"
    all_data = np.load(filepath, allow_pickle=True)
    
    # 定义目标记录、敏感数据库和查询
    target_record = all_data[1]
    target_record['question'] = "What happened in Paris during WW 2?"
    target_record['answers'] = "blabla"
    target_record['ocr_tokens'] = ['blabla', 'blabla']
    # pdb.set_trace()  # 调试断点
    sensitive_database = list(all_data[2:6]) # 基础数据库
    user_question = target_record['question']

    print(f"目标记录 Question: {target_record['question'][:80]}...",file=output_file)
    print(f"敏感数据库大小: {len(sensitive_database)}",file=output_file)
    print(f"审计查询 Question: {user_question[:80]}...",file=output_file)

    # --- 2. 初始化DP-ICL框架 ---
    dp_icl_framework = DP_ICL(llm=generator_lm, embed_model=embedding_model)

    # --- 3. 运行MIA审计 ---
    # 你可以调整这里的参数来观察不同隐私预算下的泄露情况
    audit_esa_with_reference_vector(
        dp_framework=dp_icl_framework,
        sensitive_db=sensitive_database,
        target_record=target_record,
        question=user_question,
        num_trials=10,  # 建议至少20次以获得初步统计意义，100次更佳
        epsilon=3.0,    # 理论epsilon
        delta=1e-5
    )