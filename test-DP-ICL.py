import numpy as np
from collections import Counter
import math
import re
import dspy
import csv
import pdb
import mia_2
from scipy.stats import beta
from mia_2 import EnhancedClassifier, MIAModule
# ==============================================================================
# 1. 模拟模型 (Mock Models) - 请用您自己的模型替换这些函数
# ==============================================================================
with open("/mnt/8t_data/yuyang/ag_news_csv/test.csv", 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    data = list(reader)[1:]

types = [None, "World", "Sports", "Business", "Sci/Tech"]

target_lm = dspy.LM("ollama_chat/llama3.1:8b6k",
                         api_key="sk-LGORY9rX516364f6fd78T3BLbKFJ0cD7f508E7aF41C0a769", 
                         base_url="http://localhost:11434",
                         temperature=0.1)

attack_lm = dspy.LM("ollama_chat/llama3.1:8b6k",
                        api_key="sk-LGORY9rX516364f6fd78T3BLbKFJ0cD7f508E7aF41C0a769", 
                        base_url="http://localhost:11434",
                        temperature=0.1)

# attack_lm = dspy.LM("together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
#                         api_key="2f87bd2dace135491c98dd468ff09d757e30b0dbde9cfd091b2859020b9d0bc4", 
#                         # base_url="http://localhost:11434",
#                         temperature=0.1)
dspy.configure(lm=attack_lm)

def mock_llm_classifier(contexts: str, original_news_sample: str) -> str:
    mia_module = MIAModule(
        attack_lm=attack_lm
    )

    classifier= EnhancedClassifier(
        target_lm=target_lm,
        compiled_mia_module=mia_module
    )
    
    result = classifier.forward(
        contexts=contexts,
        original_news_sample_text=original_news_sample
    )
    # pdb.set_trace()
    classification = getattr(result, "is_member", False)

    """模拟一个LLM分类器。实际应调用您的LLM API。"""
    if classification:
        return "True"
    else:
        return "False"

def mock_llm_generator(prompt: str) -> str:
    """模拟一个LLM生成器。实际应调用您的LLM API。"""
    base_sentences = [
        "Joyce found a great deal on a link shared by Michael and Edson.",
        "Joyce shared a link about a cheap ticket deal that Edson is now booking.",
        "Michael and Edson thought the link Joyce shared was a good deal.",
        "The deal found by Joyce was for a cheap ticket."
    ]
    if "suggestions" in prompt:
        # 模拟基于关键词生成
        keywords = prompt.split("suggestions:")[1].strip().replace('"', '').split(", ")
        return f"Based on keywords like {keywords[0]} and {keywords[1]}, the summary is that a good deal was found."
    return np.random.choice(base_sentences)

def mock_embedding_model(text: str) -> np.ndarray:
    """模拟一个文本到嵌入的模型。实际应调用您的Embedding API。"""
    # 生成一个固定维度的随机向量并归一化，模拟 l2 norm 为 1 的情况 
    vec = np.random.rand(128)
    return vec / np.linalg.norm(vec)

# ==============================================================================
# 2. DP-ICL 文本分类实现
# ==============================================================================

class DPICLForClassification:
    """
    通过私有投票（Report-Noisy-Max with Gaussian Noise）实现用于文本分类的DP-ICL。
    该实现遵循论文的 Section 3.1 和 Algorithm 1 [cite: 114, 471]。
    """
    def __init__(self, llm_callable, class_labels: list, epsilon: float, delta: float):
        """
        初始化分类器。
        :param llm_callable: 一个接受prompt字符串并返回单个类别标签的函数。
        :param class_labels: 所有可能的类别标签列表，如 ["Positive", "Negative"]。
        :param epsilon: 差分隐私参数 Epsilon。
        :param delta: 差分隐私参数 Delta。
        """
        self.llm = llm_callable
        self.class_labels = class_labels
        self.num_classes = len(class_labels)
        self.label_to_id = {label: i for i, label in enumerate(class_labels)}
        self.id_to_label = {i: label for i, label in enumerate(class_labels)}
        self.epsilon = epsilon
        self.delta = delta

    def _private_voting(self, votes: list[str]) -> str:
        """
        对投票结果应用 Report-Noisy-Max with Gaussian Noise 机制 。
        """
        # 步骤 1: 创建输出直方图 [cite: 115, 117]
        histogram = np.zeros(self.num_classes)
        for vote in votes:
            if vote in self.label_to_id:
                histogram[self.label_to_id[vote]] += 1

        # 计算高斯噪声的标准差
        # 灵敏度 (sensitivity) Δ 为 sqrt(2)，因为改变一个exemplar最多影响两个计数（一增一减）。
        sensitivity = math.sqrt(2)
        # 根据论文 Theorem 3  和公式 [cite: 529] 计算 sigma
        sigma = (2 * math.sqrt( math.log(1.25 / self.delta))) / self.epsilon

        # 步骤 2: 为每个计数值添加高斯噪声 [cite: 118, 127]
        noisy_histogram = histogram + np.random.normal(0, sigma, self.num_classes)
        # pdb.set_trace()
        # 步骤 3: 释放噪声最大的类别 (Release Noisy Argmax) [cite: 119, 128]
        winner_index = np.argmax(noisy_histogram)
        return self.id_to_label[winner_index]

    def __call__(self, exemplars: list[str], query_str: str, num_subsets: int) -> str:
        """
        执行完整的DP-ICL流程。
        """
        if not exemplars or num_subsets <= 0:
            raise ValueError("Exemplars 列表不能为空且 num_subsets 必须为正数。")
        
        # 阶段 1: 将exemplars分区 [cite: 101, 471]
        # 为简单起见，这里采用简单的分块。论文中提到了Poisson采样 [cite: 108]。
        partitioned_exemplars = np.array_split(exemplars, num_subsets)

        # 阶段 2 & 3: 为每个分区构建prompt并查询LLM [cite: 102, 103]
        votes = []
        for subset in partitioned_exemplars:
            # 格式化prompt，包含示例和查询
            # prompt = "\n".join(subset) + f"\n\nQuery: {query}\nAnswer:"
            contexts = "\n".join(subset)
            # 获取模型输出 
            
            prediction = self.llm(contexts=contexts, original_news_sample=query_str)
            votes.append(prediction)
        # pdb.set_trace()
        # 阶段 4: 私有聚合 [cite: 104]
        return self._private_voting(votes)


# ==============================================================================
# 3. DP-ICL 文本生成实现
# ==============================================================================
def _cosine_similarity(v1, v2):
    """计算两个向量的余弦相似度。"""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

class DPICLForGeneration:
    """
    为语言生成任务实现DP-ICL。
    实现了 Embedding Space Aggregation (ESA) 和 Keyword Space Aggregation (KSA) 两种方法。
    """
    def __init__(self, llm_callable, embedding_model_callable, epsilon: float, delta: float):
        self.llm = llm_callable
        self.embedding_model = embedding_model_callable
        self.epsilon = epsilon
        self.delta = delta

    def _embedding_space_aggregation(self, sentences: list[str], public_candidates: list[str]) -> str:
        """
        实现 Embedding Space Aggregation (ESA) [cite: 184, 472]。
        遵循论文 Figure 4(a)  和 Algorithm 2  的流程。
        """
        # 步骤 1: 将所有输出句子投影到嵌入空间 
        embeddings = [self.embedding_model(s) for s in sentences]

        # 步骤 2: 计算均值并添加高斯噪声 [cite: 178]
        mean_embedding = np.mean(embeddings, axis=0)
        
        # 灵敏度为 1/N，其中 N 是子集数量（即句子数量），因为每个嵌入的l2范数为1 。
        sensitivity = 1.0 / len(sentences)
        sigma = (sensitivity * math.sqrt(2 * math.log(1.25 / self.delta))) / self.epsilon
        
        noisy_mean_embedding = mean_embedding + np.random.normal(0, sigma, mean_embedding.shape)

        # 步骤 3: 从公共候选中找到与私有化嵌入最相似的句子并返回 [cite: 189, 195, 477]
        candidate_embeddings = [self.embedding_model(c) for c in public_candidates]
        similarities = [_cosine_similarity(noisy_mean_embedding, emb) for emb in candidate_embeddings]
        
        best_candidate_index = np.argmax(similarities)
        return public_candidates[best_candidate_index]

    def _keyword_space_aggregation_ptr(self, sentences: list[str], k_range: tuple = (5, 15)) -> str:
        """
        实现 Keyword Space Aggregation (KSA) with Propose-Test-Release (PTR) 。
        遵循论文 Figure 4(b) [cite: 179], Algorithm 3, 5, 6 的逻辑 [cite: 486, 501, 504]。
        """
        # 步骤 1 & 2: 分解句子，创建关键词直方图 [cite: 179]
        words = re.findall(r'\w+', ' '.join(sentences).lower())
        # 简单的停用词列表，可以扩展
        stop_words = set(['a', 'an', 'the', 'is', 'it', 'to', 'and', 'in', 'on', 'of'])
        meaningful_words = [word for word in words if word not in stop_words and not word.isdigit()]
        histogram = Counter(meaningful_words)
        
        # 步骤 3: 私下选择关键词
        
        # 对应 Algorithm 6: FindBestK 
        # 使用指数机制（通过Gumbel噪声实现）来私下选择最佳的k值 [cite: 502, 566]
        sorted_counts = sorted(histogram.values(), reverse=True)
        gaps = {k: (sorted_counts[k-1] - sorted_counts[k] if k < len(sorted_counts) else 0) 
                for k in range(1, len(sorted_counts))}
        
        # 灵敏度为 2 [cite: 576]
        # 注意：这里我们为选择k分配一部分隐私预算，比如 epsilon/2
        k_selection_epsilon = self.epsilon / 2
        
        # 添加正则化器，限制k的范围 [cite: 505]
        utilities = {}
        for k, gap in gaps.items():
            if k_range[0] <= k <= k_range[1]:
                gumbel_noise = np.random.gumbel(loc=0, scale=4 / k_selection_epsilon)
                utilities[k] = gap + gumbel_noise
        
        if not utilities: # 如果没有有效的k
             return self.llm("Summarize the main idea.") # 回退
        
        best_k = max(utilities, key=utilities.get)

        # 对应 Algorithm 5: TopKwithPTR 
        # 使用剩余预算进行 PTR 测试
        ptr_epsilon = self.epsilon / 2
        ptr_delta = self.delta
        
        # 如果第k个和第k+1个候选词的票数差距大于2，我们可以直接发布top-k结果 
        dk = gaps.get(best_k, 0)
        
        # 私下测试 dk > 2 [cite: 500]
        # 为简化，我们直接实现其核心思想。如果差距足够大，则认为测试通过。
        # 严格实现需要对 noisy_dk 进行比较。
        # test_threshold = 2 + np.random.normal(0, math.sqrt(4 * sigma_ptr**2)) # 简化处理
        if dk > 2:
            top_k_keywords = [item[0] for item in histogram.most_common(best_k)]
        else:
            # 测试失败，终止或使用零样本学习 
            return self.llm("Summarize the main idea without any private context.")
            
        # 步骤 4: 使用选定的关键词重构最终输出 [cite: 181]
        keyword_str = ", ".join(f'"{w}"' for w in top_k_keywords)
        # 根据论文修改prompt [cite: 704]
        new_prompt = f"Summarize the main idea with the following word suggestions: {keyword_str}"
        return self.llm(new_prompt)

    def __call__(self, exemplars: list[str], query: str, num_subsets: int, 
                 method: str = 'ESA', public_candidates: list[str] = None) -> str:
        """
        执行完整的DP-ICL生成流程。
        :param method: 'ESA' 或 'KSA'。
        :param public_candidates: 仅在 method='ESA' 时需要，由零样本预测生成的候选句子列表 。
        """
        if not exemplars or num_subsets <= 0:
            raise ValueError("Exemplars 列表不能为空且 num_subsets 必须为正数。")

        # 阶段 1, 2, 3
        partitioned_exemplars = np.array_split(exemplars, num_subsets)
        output_sentences = []
        for subset in partitioned_exemplars:
            prompt = "\n".join(subset) + f"\n\nQuery: {query}\nAnswer:"
            prediction = self.llm(prompt)
            output_sentences.append(prediction)

        # 阶段 4: 私有聚合
        if method == 'ESA':
            if not public_candidates:
                raise ValueError("ESA 方法需要提供 public_candidates。")
            return self._embedding_space_aggregation(output_sentences, public_candidates)
        elif method == 'KSA':
            # 论文提到了 Joint EM 和 PTR [cite: 71, 212]。这里我们实现了PTR。
            # Joint EM 的实现非常复杂，论文也指出他们使用了现有代码 [cite: 494]。
            return self._keyword_space_aggregation_ptr(output_sentences)
        else:
            raise ValueError("方法必须是 'ESA' 或 'KSA'。")

def clopper_pearson(c0, c1, T, a):
    """
    计算Clopper-Pearson置信区间的上下界。

    参数:
        c1: 成功的次数。
        T: 实验总次数。
        a: 置信水平 (例如, 0.95 表示 95% 的置信水平)。

    返回:
        一个元组，包含置信区间的下界和上界。
    """
    alpha = 1 - a
    lower_bound = beta.ppf(alpha / 2, c1, T - c1 + 1)
    upper_bound = beta.ppf(1 - alpha / 2, c0 + 1, T - c0)
    return lower_bound, upper_bound

# ==============================================================================
# 4. 示例用法
# ==============================================================================
if __name__ == '__main__':
    # --- 文本分类任务示例 ---
    print("="*20 + " 文本分类任务示例 " + "="*20)
    subset_data = data[:100]  # 仅使用前100条数据进行测试
    classification_exemplars = [subset_data[i][2] for i in range(1, 10)]
    query_classify = subset_data[0][2]
    
    dp_classifier = DPICLForClassification(
        llm_callable=mock_llm_classifier,
        class_labels=["True", "False"],
        epsilon=1.0,
        delta=1e-3
    )
    
    T = 1000
    private_result_classify = dp_classifier(
        exemplars=classification_exemplars,
        query_str=query_classify,
        num_subsets=1
    )
    print(f"查询: {query_classify}")
    print(f"经过DP-ICL保护的分类结果: {private_result_classify}\n")
    count = 0
    for _ in range(T):
        private_result_classify = dp_classifier(
            exemplars=classification_exemplars,
            query_str=query_classify,
            num_subsets=1
        )
        # print(f"查询: {query_classify}")
        # print(f"经过DP-ICL保护的分类结果: {private_result_classify}\n")
        if private_result_classify == "True":
            count += 1
    print(f"经过DP-ICL保护的分类结果为 True 的比例: {count / T:.2f}\n")


    classification_exemplars = [subset_data[i][2] for i in range(1, 2)]
    query_classify = subset_data[0][2]
    
    dp_classifier = DPICLForClassification(
        llm_callable=mock_llm_classifier,
        class_labels=["True", "False"],
        epsilon=20.0,
        delta=1e-3
    )
    
    T = 1000
    private_result_classify = dp_classifier(
        exemplars=classification_exemplars,
        query_str=query_classify,
        num_subsets=1
    )
    print(f"查询: {query_classify}")
    print(f"经过DP-ICL保护的分类结果: {private_result_classify}\n")
    count_2 = 0
    for _ in range(T):
        private_result_classify = dp_classifier(
            exemplars=classification_exemplars,
            query_str=query_classify,
            num_subsets=1
        )
        # print(f"查询: {query_classify}")
        # print(f"经过DP-ICL保护的分类结果: {private_result_classify}\n")
        if private_result_classify == "True":
            count_2 += 1
    print(f"经过DP-ICL保护的分类结果为 True 的比例: {count_2 / T:.2f}\n")

    lower, upper = clopper_pearson(count,count_2, T, a=0.95)
    print(f"Clopper-Pearson 置信区间下界: {lower:.4f}")
    print(f"Clopper-Pearson 置信区间上界: {upper:.4f}") 
    eps = math.log(upper/lower)
    print(f"计算出的 epsilon: {eps:.4f}")
    
