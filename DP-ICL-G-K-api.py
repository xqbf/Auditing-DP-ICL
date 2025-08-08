import dspy
import numpy as np
import math
import pdb
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
import warnings
import re
from collections import Counter
from scipy.stats import beta, norm

# 忽略来自huggingface_hub的特定版本警告
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")
output_file = open('DP-ICL-G-K-api.txt', 'w')
# ==============================================================================
# 1. DSPy和模型设置
# ==============================================================================
print("正在初始化模型...",file = output_file)
# 用于生成摘要和进行MIA判断的LLM
generator_lm = dspy.LM("ollama_chat/llama3.1:8b",
                       base_url="http://localhost:11434",
                       temperature=0.1)

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
            # pdb.set_trace()  # 调试断点
            context_str = "\n".join([f"Q: {ex.get('question', '')}\nA: {ex.get('answers', '')[0] if isinstance(ex.get('answers'), list) and ex.get('answers') else ex.get('answers', '')}" for ex in exemplars])
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
        noise = np.random.normal(0, sigma**2, mean_embedding.shape)
        private_mean_embedding = mean_embedding + noise
        
        candidate_signature = dspy.Signature("question -> answer")
        candidates = []
        with dspy.context(lm=self.llm):
            for _ in range(num_candidates):
                result = dspy.Predict(candidate_signature)(question=original_question)
                candidates.append(result.answer)
        
        candidate_embeddings = self.embed_model.encode(candidates, normalize_embeddings=True)
        similarities = [1 - cosine(private_mean_embedding, emb) for emb in candidate_embeddings]
        best_candidate_index = np.argmax(similarities)
        final_answer = candidates[best_candidate_index]
        return final_answer

    # 在 DP_ICL 类中，请使用这个最终的、实用的函数版本
    def _keyword_space_aggregation_ptr(self, responses, question, top_k=10, epsilon=2.0, delta=1e-5):
        """
        【最终实用版】实现一个标准的高斯机制阈值测试。
        这个版本能够在隐私和可用性之间取得实际的平衡。
        """
        all_words = []
        def fallback():
            with dspy.context(lm=self.llm):
                answer = dspy.Predict(dspy.Signature("question -> answer"))(question=question).answer
                return answer, []
        for res in responses:
            words = re.findall(r'\b\w+\b', res.lower())
            all_words.extend(list(set(words)))
        word_counts = Counter(all_words)
        
        top_k = len(word_counts) - 1

        if not word_counts or len(word_counts) <= top_k:
            return fallback()
        
        sorted_counts = word_counts.most_common()
        H_k = sorted_counts[top_k-1][1]
        H_k_plus_1 = sorted_counts[top_k][1]
        gap = H_k - H_k_plus_1
        
        # --- 标准高斯机制开始 ---
        sensitivity = 2

        if epsilon <= 0:
            return fallback()
        
        # 1. 根据(epsilon, delta)为高斯机制校准sigma
        # 这是实现(ε, δ)-DP的标准做法
        sigma = (sensitivity / epsilon) * np.sqrt(2 * np.log(1.25 / delta))

        # 2. 对gap值添加高斯噪声
        noise = np.random.normal(0, sigma)
        noisy_gap = gap + noise
        
        # 3. 执行一个简单、标准且有效的决策
        # 这里的决策阈值是0。我们是在用隐私保护的方式问：“gap是否大于0？”
        # 这个决策是后处理步骤，不消耗额外隐私预算。
        # 当epsilon很大时，sigma很小，noisy_gap的符号很可能与真实gap一致，测试易通过。
        # 当epsilon很小时，sigma很大，noisy_gap的符号几乎是随机的，从而保护隐私。
        print(noisy_gap)
        if noisy_gap > 0:
            top_keywords = [item[0] for item in sorted_counts[:top_k]]
        else:
            return fallback()

        # --- 标准高斯机制结束 ---

        reconstruction_prompt = f"Please answer the question based on potential context. Use the following suggested words to guide your answer: {', '.join(top_keywords)}."
        reconstruction_signature = dspy.Signature("guidance, question -> answer")
        with dspy.context(lm=self.llm):
            final_answer = dspy.Predict(reconstruction_signature)(guidance=reconstruction_prompt, question=question).answer
        
        return final_answer, top_keywords

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
    print("正在创建参考向量 (攻击者设置阶段)...", file=output_file)
    
    predictor = dspy.Predict(dspy.Signature("context, question -> answer"))
    
    embeddings_with = []
    embeddings_without = []

    # 格式化上下文
    context_with_str = "\n".join([f"Q: {ex.get('question', '')}\nA: {ex.get('answers', '')[0] if isinstance(ex.get('answers'), list) and ex.get('answers') else ex.get('answers', '')}" for ex in exemplars_with_target])
    context_without_str = "\n".join([f"Q: {ex.get('question', '')}\nA: {ex.get('answers', '')[0] if isinstance(ex.get('answers'), list) and ex.get('answers') else ex.get('answers', '')}" for ex in exemplars_without_target])

    for i in range(L):
        # 场景1: 目标在数据集中
        with dspy.context(lm=llm):
            response_with = predictor(context=context_with_str, question=question)
            embeddings_with.append(embed_model.encode(response_with.answer, normalize_embeddings=True))

        # 场景2: 目标不在数据集中
        with dspy.context(lm=llm):
            response_without = predictor(context=context_without_str, question=question)
            embeddings_without.append(embed_model.encode(response_without.answer, normalize_embeddings=True))
        print(f"  正在生成参考样本 {i+1}/{L}", file=output_file)

        # pdb.set_trace()  # 调试断点


    mean_vec_with = np.mean(embeddings_with, axis=0)
    mean_vec_without = np.mean(embeddings_without, axis=0)
    
    v_ref = mean_vec_with - mean_vec_without
    v_ref = v_ref / np.linalg.norm(v_ref)
    
    print("参考向量创建完毕。", file=output_file)
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

def audit_ksa_standard_mia(dp_framework, sensitive_db, target_record, question, num_trials=50, epsilon=3.0, delta=1e-5):
    """
    对KSA进行标准的成员推断攻击审计。
    """
    print("\n" + "="*60, file=output_file)
    print("方法一：开始对KSA进行标准成员推断攻击审计", file=output_file)
    print(f"参数: num_trials={num_trials}, epsilon={epsilon}", file=output_file)
    print("="*60, file=output_file)

    # --- 1. 准备数据集和“秘密” ---
    exemplars_with_target = sensitive_db + [target_record]
    exemplars_without_target = sensitive_db
    
    # 定义攻击者要寻找的“秘密”关键词。我们从目标记录的答案中提取一些独特的词。
    # 这里假设答案在'answers'字段，且为列表
    answer = target_record.get('answers')
    if(isinstance(target_record['answers'], str)):
        target_answer_text = target_record['answers']
    else:
        target_answer_text = ' '.join(target_record.get('answers', []))
    secret_keywords = set(re.findall(r'\b\w+\b', target_answer_text.lower()))
    # pdb.set_trace()  # 调试断点
    if not secret_keywords:
        print("错误：目标记录中没有可供审计的秘密关键词。", file=output_file)
        return
    print(f"将寻找的秘密关键词 (来自目标记录): {list(secret_keywords)[:5]}...", file=output_file)

    # --- 2. 审计运行阶段 ---
    print("\n开始运行审计试验...", file=output_file)
    true_positives = 0
    false_positives = 0

    for i in range(num_trials):
        # 场景1: 目标记录在数据集中 (D_in)
        # 我们直接调用内部函数以获取关键词列表
        _, keywords_with = dp_framework._keyword_space_aggregation_ptr(
            dp_framework._get_llm_responses(question, np.array_split(exemplars_with_target, 30)),
            question, epsilon=epsilon, delta=delta
        )
        # 如果任何秘密关键词出现在选中的关键词中，攻击者就认为目标存在
        if secret_keywords.intersection(set(keywords_with)):
            true_positives += 1

        # 场景2: 目标记录不在数据集中 (D_out)
        _, keywords_without = dp_framework._keyword_space_aggregation_ptr(
            dp_framework._get_llm_responses(question, np.array_split(exemplars_without_target, 30)),
            question, epsilon=epsilon, delta=delta
        )
        # 如果秘密关键词错误地出现了，计为一个误报
        if secret_keywords.intersection(set(keywords_without)):
            false_positives += 1

        print(f"  审计试验 {i+1}/{num_trials} 完成。", file=output_file)

    # --- 3. 结果分析 ---
    print("\n" + "="*60, file=output_file)
    print("KSA标准MIA审计结果分析", file=output_file)
    print("="*60, file=output_file)
    
    tpr = true_positives / num_trials
    fpr = false_positives / num_trials

    tpr_interval = calculate_clopper_pearson_interval(true_positives, num_trials)
    fpr_interval = calculate_clopper_pearson_interval(false_positives, num_trials)

    print(f"攻击准确率 (TPR - 目标存在时): {tpr:.2%} ({true_positives}/{num_trials})", file=output_file)
    print(f"  - 95% 置信区间: [{tpr_interval[0]:.2%}, {tpr_interval[1]:.2%}]", file=output_file)
    print(f"攻击准确率 (FPR - 目标不存在时): {fpr:.2%} ({false_positives}/{num_trials})", file=output_file)
    print(f"  - 95% 置信区间: [{fpr_interval[0]:.2%}, {fpr_interval[1]:.2%}]", file=output_file)
    print("-" * 60, file=output_file)
    
    if tpr_interval[0]>0:
        empirical_eps = math.log(tpr_interval[0]/fpr_interval[1])
        print(f"经验隐私预算 (Empirical Epsilon): {empirical_eps:.2f}", file=output_file)
    else:
        print("经验隐私预算 (Empirical Epsilon): 无法计算，TPR下界为0", file=output_file)

    # advantage = tpr - fpr
    # print(f"攻击者优势 (Advantage = TPR - FPR): {advantage:.2%}", file=output_file)
    # if advantage > 0.1:
    #     print("[结论] KSA存在明显的隐私泄露风险。", file=output_file)
    # else:
    #     print("[结论] KSA的隐私保护似乎有效。", file=output_file)


def audit_ksa_with_poisoning(dp_framework, sensitive_db, num_trials=50, epsilon=3.0, delta=1e-5):
    """
    对KSA进行有毒金丝雀（Poisoning）攻击审计。
    """
    print("\n" + "="*80, file=output_file)
    print("方法二：开始对KSA进行有毒金丝雀攻击审计",    file=output_file)
    print(f"参数: num_trials={num_trials}, epsilon={epsilon}", file=output_file)
    print("="*80, file=output_file)

    # --- 1. 构造有毒数据和查询 ---
    # 定义一个非常独特、不可能自然出现的触发词
    poison_trigger = "my-secret-canary-code-91827"
    # 定义攻击者想要泄露的秘密信息
    poison_secret = "The secret meeting is at dawn"
    
    # 构造有毒记录
    poisoned_canary = {
        'question': f'what is the plan for {poison_trigger}?',
        'answers': [poison_secret],
        'ocr_tokens': f'Plan for {poison_trigger}: {poison_secret}' # 模拟ocr内容
    }
    
    # 审计用的查询就是这个有毒问题
    audit_question = poisoned_canary['question']
    
    # 定义秘密关键词
    secret_keywords = set(['meeting', 'dawn'])

    print(f"注入的有毒金丝雀: Q='{poisoned_canary['question']}', A='{poisoned_canary['answers'][0]}'", file=output_file)
    print(f"将寻找的秘密关键词: {secret_keywords}", file=output_file)

    # --- 2. 准备数据集 ---
    exemplars_with_poison = sensitive_db + [poisoned_canary]
    exemplars_without_poison = sensitive_db
    
    # --- 3. 审计运行阶段 ---
    print("\n开始运行审计试验...", file=output_file)
    true_positives = 0
    false_positives = 0
    
    for i in range(num_trials):
        # 场景1: 有毒记录在数据集中
        _, keywords_with = dp_framework._keyword_space_aggregation_ptr(
            dp_framework._get_llm_responses(audit_question, np.array_split(exemplars_with_poison, 3)),
            audit_question, epsilon=epsilon, delta=delta
        )
        if secret_keywords.issubset(set(keywords_with)): # 使用issubset更严格
            true_positives += 1

        # 场景2: 有毒记录不在数据集中
        _, keywords_without = dp_framework._keyword_space_aggregation_ptr(
            dp_framework._get_llm_responses(audit_question, np.array_split(exemplars_without_poison, 3)),
            audit_question, epsilon=epsilon, delta=delta
        )
        if secret_keywords.issubset(set(keywords_without)):
            false_positives += 1
            
        print(f"  审计试验 {i+1}/{num_trials} 完成。", file=output_file)

    # --- 4. 结果分析 (与方法一相同) ---
    print("\n" + "="*60,    file=output_file)
    print("KSA有毒金丝雀审计结果分析", file=output_file)
    print("="*60, file=output_file)
    
    tpr = true_positives / num_trials
    fpr = false_positives / num_trials

    tpr_interval = calculate_clopper_pearson_interval(true_positives, num_trials)
    fpr_interval = calculate_clopper_pearson_interval(false_positives, num_trials)

    print(f"攻击准确率 (TPR - 毒药存在时): {tpr:.2%} ({true_positives}/{num_trials})", file=output_file)
    print(f"  - 95% 置信区间: [{tpr_interval[0]:.2%}, {tpr_interval[1]:.2%}]", file=output_file)
    print(f"攻击准确率 (FPR - 毒药不存在时): {fpr:.2%} ({false_positives}/{num_trials})", file=output_file)
    print(f"  - 95% 置信区间: [{fpr_interval[0]:.2%}, {fpr_interval[1]:.2%}]", file=output_file)
    print("-" * 60, file=output_file)
    
    advantage = tpr - fpr
    print(f"攻击者优势 (Advantage = TPR - FPR): {advantage:.2%}",   file=output_file)
    if advantage > 0.1:
        print("[结论] KSA存在明显的隐私泄露风险。", file=output_file)
    else:
        print("[结论] KSA的隐私保护似乎有效。", file=output_file)

if __name__ == '__main__':
    # --- 1. 数据准备 ---
    print("准备审计所需数据...", file=output_file)
    filepath = "/mnt/8t_data/yuyang/PFL-DocVQA_imdbs/imdb_train_client_0.npy"
    all_data = np.load(filepath, allow_pickle=True)
    
    # 基础数据库
    sensitive_database = list(all_data[2:60]) 
    print(f"敏感数据库大小: {len(sensitive_database)}", file=output_file)

    # --- 2. 初始化DP-ICL框架 ---
    dp_icl_framework = DP_ICL(llm=generator_lm, embed_model=embedding_model)

    # ==========================================================
    # 运行审计方法一: 标准成员推断攻击
    # ==========================================================
    # 定义目标记录和查询
    target_record_mia = all_data[1]
    user_question_mia = target_record_mia['question']
    print(f"\n标准MIA的目标记录 Question: {target_record_mia['question'][:80]}...", file=output_file)
    
    audit_ksa_standard_mia(
        dp_framework=dp_icl_framework,
        sensitive_db=sensitive_database,
        target_record=target_record_mia,
        question=user_question_mia,
        num_trials=2000,
        epsilon=100
    )
    
    # ==========================================================
    # 运行审计方法二: 有毒金丝雀攻击
    # ==========================================================
    # audit_ksa_with_poisoning(
    #     dp_framework=dp_icl_framework,
    #     sensitive_db=sensitive_database, # 使用相同的基础数据库
    #     num_trials=20,
    #     epsilon=3.0
    # )

    # # (可选) 运行对ESA的审计
    # print("\n\n--- 重新运行对ESA的审计 ---")
    # audit_esa_with_reference_vector(
    #     dp_framework=dp_icl_framework,
    #     sensitive_db=sensitive_database,
    #     target_record=target_record_mia,
    #     question=user_question_mia,
    #     num_trials=20,
    #     epsilon=3.0,
    #     delta=1e-5
    # )