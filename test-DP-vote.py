import numpy as np
from collections import Counter
import math
import re
import dspy
import csv
import pdb
from scipy.stats import beta
from mia_2 import EnhancedClassifier, MIAModule

# ==============================================================================
# 1. 模拟模型和数据加载 - 与您原文件相同
# ==============================================================================
# 为了代码独立性，我们先定义一个 placeholder
# 实际运行时，dspy 和 mia_2 的依赖需要被满足
try:
    with open("/mnt/8t_data/yuyang/ag_news_csv/test.csv", 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        data = list(reader)[1:]

    target_lm = dspy.LM("ollama_chat/llama3.1:8b6k",
                             api_key="sk-LGORY9rX516364f6fd78T3BLbKFJ0cD7f508E7aF41C0a769", 
                             base_url="http://localhost:11434",
                             temperature=0.1)
    
    attack_lm = dspy.LM("ollama_chat/llama3.1:8b6k",
                            api_key="sk-LGORY9rX516364f6fd78T3BLbKFJ0cD7f508E7aF41C0a769", 
                            base_url="http://localhost:11434",
                            temperature=0.1)
    
    dspy.configure(lm=attack_lm)

except FileNotFoundError:
    print("警告：无法加载数据文件和dspy模型。将使用纯模拟函数。")
    data = [[i, f"Class {(i % 4) + 1}", f"Title {i}", f"Sample text for data point {i}"] for i in range(100)]
    def mock_llm_classifier(contexts: str, original_news_sample: str) -> str:
        # 这是一个简化的模拟，实际逻辑在您的mia_2中
        # 模拟：如果上下文中提到了样本的ID，则有高概率返回True
        sample_id_match = re.search(r'data point (\d+)', original_news_sample)
        if sample_id_match and sample_id_match.group(1) in contexts:
            return "True"
        return "False"
    llm_classifier_callable = mock_llm_classifier
else:
    def llm_classifier_callable(contexts: str, original_news_sample: str) -> str:
        mia_module = MIAModule(attack_lm=attack_lm)
        classifier = EnhancedClassifier(target_lm=target_lm, compiled_mia_module=mia_module)
        result = classifier.forward(contexts=contexts, original_news_sample_text=original_news_sample)
        classification = getattr(result, "is_member", False)
        return "True" if classification else "False"

# ==============================================================================
# 2. 统一的私有聚合器
# ==============================================================================
def private_voting_aggregator(votes: list[str], class_labels: list, epsilon: float, delta: float) -> str:
    """
    一个独立的私有聚合器，执行 Report-Noisy-Max with Gaussian Noise。
    两个实验都将调用这个函数，以证明最后一步是完全相同的。
    """
    num_classes = len(class_labels)
    label_to_id = {label: i for i, label in enumerate(class_labels)}
    id_to_label = {i: label for i, label in enumerate(class_labels)}

    # 步骤 1: 创建输出直方图
    histogram = np.zeros(num_classes)
    for vote in votes:
        if vote in label_to_id:
            histogram[label_to_id[vote]] += 1
            
    # 计算高斯噪声的标准差
    sensitivity = math.sqrt(2) # L2 敏感度
    sigma = (sensitivity * math.sqrt(2 * math.log(1.25 / delta))) / epsilon

    # 步骤 2: 添加高斯噪声
    noisy_histogram = histogram + np.random.normal(0, sigma, num_classes)
    
    # 步骤 3: 释放噪声最大的类别
    winner_index = np.argmax(noisy_histogram)
    return id_to_label[winner_index]


# ==============================================================================
# 3. DP-ICL (基于LLM) 与 传统Private Voting (基于ID检查) 的实现
# ==============================================================================

class DPICLForClassification:
    """您原有的DP-ICL实现，现在调用统一的聚合器。"""
    def __init__(self, llm_callable, class_labels: list, epsilon: float, delta: float):
        self.llm = llm_callable
        self.class_labels = class_labels
        self.epsilon = epsilon
        self.delta = delta

    def __call__(self, exemplars: list[str], query_str: str, num_subsets: int) -> str:
        if not exemplars or num_subsets <= 0:
            raise ValueError("Exemplars 列表不能为空且 num_subsets 必须为正数。")
        
        partitioned_exemplars = np.array_split(exemplars, num_subsets)

        votes = []
        for subset in partitioned_exemplars:
            contexts = "\n".join(subset)
            # 核心区别(1): 这里的投票是由一个复杂的、黑盒的LLM生成的
            prediction = self.llm(contexts=contexts, original_news_sample=query_str)
            votes.append(prediction)
        
        return private_voting_aggregator(votes, self.class_labels, self.epsilon, self.delta)


class TraditionalPrivateVoting:
    """
    传统的Private Voting实现，不依赖LLM。
    投票通过直接检查成员ID是否存在于子集中来产生。
    """
    def __init__(self, class_labels: list, epsilon: float, delta: float):
        self.class_labels = class_labels
        self.epsilon = epsilon
        self.delta = delta

    def __call__(self, exemplar_ids: list[int], query_id: int, num_subsets: int) -> str:
        if not exemplar_ids or num_subsets <= 0:
            raise ValueError("Exemplar IDs 列表不能为空且 num_subsets 必须为正数。")

        partitioned_exemplar_ids = np.array_split(exemplar_ids, num_subsets)
        
        votes = []
        for subset in partitioned_exemplar_ids:
            # 核心区别(2): 这里的投票是一个简单的、本地的、确定性的检查
            is_member = query_id in subset.tolist()
            vote = "True" if is_member else "False"
            votes.append(vote)
        
        # 最后的聚合与加噪步骤，与DP-ICL完全相同
        return private_voting_aggregator(votes, self.class_labels, self.epsilon, self.delta)

# ==============================================================================
# 4. 统计与审计工具
# ==============================================================================
def clopper_pearson(k, n, alpha=0.01):
    """计算单个比例的Clopper-Pearson置信区间。"""
    lower = beta.ppf(alpha / 2, k, n - k + 1)
    upper = beta.ppf(1 - alpha / 2, k + 1, n - k)
    return lower, upper

def calculate_empirical_epsilon(tpr_ci_upper, fpr_ci_lower, delta):
    """根据置信区间的边界计算经验epsilon。"""
    if fpr_ci_lower <= 0: # 避免除以零或负数
        return float('inf')
    # 从 TPR <= exp(eps) * FPR + delta 反推
    # 忽略delta的小影响，eps ≈ log(TPR/FPR)
    # 使用置信区间的边界来获得一个鲁棒的估计
    return math.log(tpr_ci_upper / fpr_ci_lower)

# ==============================================================================
# 5. 对比实验
# ==============================================================================
if __name__ == '__main__':
    # --- 实验参数 ---
    EPSILON = 1.0
    DELTA = 1e-5 # 根据论文建议，对于小数据集使用较小的delta
    NUM_SUBSETS = 20
    TOTAL_EXEMPLARS = 40
    T = 1000 # 实验次数
    CI_ALPHA = 0.01 # 99% 置信区间

    # --- 准备数据 ---
    # 为简单和可复现，我们使用数据索引作为ID
    all_data_text = [row[2] for row in data[:100]]

    # 成员情况 (D): 查询点 0 在数据集中
    exemplars_D_text = all_data_text[:TOTAL_EXEMPLARS]
    exemplars_D_ids = list(range(TOTAL_EXEMPLARS))
    
    # 非成员情况 (D'): 查询点 0 不在数据集中
    exemplars_D_prime_text = all_data_text[1:TOTAL_EXEMPLARS + 1]
    exemplars_D_prime_ids = list(range(1, TOTAL_EXEMPLARS + 1))
    
    query_text = all_data_text[0]
    query_id = 0

    print("实验参数:")
    print(f"  - 目标 Epsilon: {EPSILON}")
    print(f"  - Delta: {DELTA}")
    print(f"  - 分组数量 (N): {NUM_SUBSETS}")
    print(f"  - 每次实验迭代次数 (T): {T}\n")

    # --- 实验1: DP-ICL (您的方法) ---
    print("="*20 + " 实验1: 运行 DP-ICL (基于LLM) " + "="*20)
    dp_classifier = DPICLForClassification(
        llm_callable=llm_classifier_callable,
        class_labels=["True", "False"],
        epsilon=EPSILON,
        delta=DELTA
    )

    # 计算 TPR (True Positive Rate)
    tp_count = 0
    for _ in range(T):
        result = dp_classifier(exemplars_D_text, query_text, NUM_SUBSETS)
        if result == "True":
            tp_count += 1
    
    # 计算 FPR (False Positive Rate)
    fp_count = 0
    for _ in range(T):
        result = dp_classifier(exemplars_D_prime_text, query_text, NUM_SUBSETS)
        if result == "True":
            fp_count += 1
            
    tpr = tp_count / T
    fpr = fp_count / T
    tpr_lower, tpr_upper = clopper_pearson(tp_count, T, CI_ALPHA)
    fpr_lower, fpr_upper = clopper_pearson(fp_count, T, CI_ALPHA)
    empirical_eps_dpicl = calculate_empirical_epsilon(tpr_upper, fpr_lower, DELTA)

    print(f"DP-ICL 结果:")
    print(f"  - TPR (True Positive Rate): {tpr:.4f}")
    print(f"  - FPR (False Positive Rate): {fpr:.4f}")
    print(f"  - 审计出的经验 Epsilon: {empirical_eps_dpicl:.4f}\n")


    # --- 实验2: 传统 Private Voting ---
    print("="*20 + " 实验2: 运行传统 Private Voting (基于ID) " + "="*20)
    traditional_voter = TraditionalPrivateVoting(
        class_labels=["True", "False"],
        epsilon=EPSILON,
        delta=DELTA
    )

    # 计算 TPR
    tp_count_trad = 0
    for _ in range(T):
        result = traditional_voter(exemplars_D_ids, query_id, NUM_SUBSETS)
        if result == "True":
            tp_count_trad += 1
            
    # 计算 FPR
    fp_count_trad = 0
    for _ in range(T):
        result = traditional_voter(exemplars_D_prime_ids, query_id, NUM_SUBSETS)
        if result == "True":
            fp_count_trad += 1

    tpr_trad = tp_count_trad / T
    fpr_trad = fp_count_trad / T
    tpr_lower_trad, tpr_upper_trad = clopper_pearson(tp_count_trad, T, CI_ALPHA)
    fpr_lower_trad, fpr_upper_trad = clopper_pearson(fp_count_trad, T, CI_ALPHA)
    empirical_eps_trad = calculate_empirical_epsilon(tpr_upper_trad, fpr_lower_trad, DELTA)

    print(f"传统 Private Voting 结果:")
    print(f"  - TPR (True Positive Rate): {tpr_trad:.4f}")
    print(f"  - FPR (False Positive Rate): {fpr_trad:.4f}")
    print(f"  - 审计出的经验 Epsilon: {empirical_eps_trad:.4f}\n")

    # --- 最终对比 ---
    print("="*20 + " 最终对比 " + "="*20)
    print(f"目标隐私预算 Epsilon: {EPSILON}")
    print(f"DP-ICL 的实证 Epsilon: {empirical_eps_dpicl:.4f} " + 
          ("<= 预算内" if empirical_eps_dpicl <= EPSILON else ">> 超出预算!"))
    print(f"传统 Private Voting 的实证 Epsilon: {empirical_eps_trad:.4f} " +
          ("<= 预算内" if empirical_eps_trad <= EPSILON else ">> 超出预算!"))
