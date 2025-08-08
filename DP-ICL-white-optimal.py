import numpy as np
import math
import csv
import dspy
from scipy.stats import beta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# 确保 roc_curve 已被导入
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve 
from sklearn.ensemble import GradientBoostingClassifier

# ==============================================================================
# 1. DSPy 和 LLM 设置 (与之前相同)
# ==============================================================================
try:
    attack_lm = dspy.LM("ollama_chat/llama3.1:8b",
                        base_url="http://localhost:11434",
                        temperature=0.1)
    dspy.configure(lm=attack_lm)
except Exception as e:
    print(f"无法配置DSPy LM: {e}")
    print("将使用模拟LLM进行演示。")
    class MockLM(dspy.LM):
        def __init__(self):
            super().__init__("mock-model")
        def __call__(self, prompt, **kwargs):
            return [dspy.Prediction(completions=[dspy.Completion(text=np.random.choice(["World", "Sports", "Business", "Sci/Tech"]))])]
    attack_lm = MockLM()
    dspy.configure(lm=attack_lm)

# ... (NewsClassification, llm_classifier, DPICLForClassification, clopper_pearson 函数保持不变) ...

class NewsClassification(dspy.Signature):
    """将新闻文章分类到所提供的类别之一。"""
    context = dspy.InputField(desc="新闻文章及其类别的少样本示例。")
    article = dspy.InputField(desc="要分类的新闻文章。")
    category = dspy.OutputField(desc="世界、体育、商业、科技/技术之一。")

dspy_predictor = dspy.Predict(NewsClassification)

def llm_classifier(contexts: list[str], query_article: str, exemplars_labels: list[str]) -> str:
    formatted_context = ""
    for text, label in zip(contexts, exemplars_labels):
        formatted_context += f"文章: {text}\n答案: {label}\n\n"
    result = dspy_predictor(context=formatted_context, article=query_article)
    predicted_category = result.completions[0].category.split('\n')[0].strip()
    valid_labels = ["World", "Sports", "Business", "Sci/Tech"]
    if predicted_category not in valid_labels:
        return np.random.choice(valid_labels)
    return predicted_category

class DPICLForClassification:
    def __init__(self, llm_callable, class_labels: list, epsilon: float, delta: float):
        self.llm = llm_callable
        self.class_labels = class_labels
        self.num_classes = len(class_labels)
        self.label_to_id = {label: i for i, label in enumerate(class_labels)}
        self.id_to_label = {i: label for i, label in enumerate(class_labels)}
        self.epsilon = epsilon
        self.delta = delta

    def _private_voting(self, votes: list[str]) -> tuple[str, np.ndarray]:
        histogram = np.zeros(self.num_classes)
        for vote in votes:
            if vote in self.label_to_id:
                histogram[self.label_to_id[vote]] += 1
        sensitivity = math.sqrt(2)
        if self.epsilon == 0:
            sigma = float('inf')
        else:
            sigma = (sensitivity * math.sqrt(2 * math.log(1.25 / self.delta))) / self.epsilon
        noisy_histogram = histogram + np.random.normal(0, sigma, self.num_classes)
        winner_index = np.argmax(noisy_histogram)
        return (self.id_to_label[winner_index], noisy_histogram)

    def __call__(self, exemplars: list[tuple[str, str]], query_str: str, num_subsets: int) -> tuple[str, np.ndarray]:
        if not exemplars or num_subsets <= 0:
            raise ValueError("示例列表不能为空且num_subsets必须为正数。")
        shuffled_indices = np.random.permutation(len(exemplars))
        partition_indices = np.array_split(shuffled_indices, num_subsets)
        votes = []
        for subset_indices in partition_indices:
            if len(subset_indices) == 0:
                continue
            subset_exemplars = [exemplars[i][0] for i in subset_indices]
            subset_labels = [exemplars[i][1] for i in subset_indices]
            prediction = self.llm(
                contexts=subset_exemplars,
                query_article=query_str,
                exemplars_labels=subset_labels
            )
            votes.append(prediction)
        return self._private_voting(votes)

def clopper_pearson(successes, trials, confidence_level):
    if trials == 0: return (0.0, 1.0)
    alpha = 1 - confidence_level
    lower = beta.ppf(alpha / 2, successes, trials - successes + 1)
    upper = beta.ppf(1 - alpha / 2, successes + 1, trials - successes)
    return (lower if not np.isnan(lower) else 0.0, upper if not np.isnan(upper) else 1.0)

# ==============================================================================
# 4. 主程序
# ==============================================================================
if __name__ == '__main__':
    # --- 数据和任务设置 ---
    try:
         with open("/mnt/8t_data/yuyang/ag_news_csv/test.csv", 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            all_data = [(row[0], row[1], row[2]) for row in list(reader)[1:]]
    except FileNotFoundError:
        print("错误: 未找到`ag_news_csv/test.csv`。请确保数据集位于正确的路径。")
        exit()

    output_file = open('output-whitebox-optimal.txt', 'w', encoding = 'utf-8')
    
    print("="*20 + " DP-ICL 分类与隐私审计 " + "="*20, file=output_file)
    class_map = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}
    
    # --- 隐私审计参数 ---
    target_record_text = "Real Madrid has signed a new football player."
    query_article_text = target_record_text 
    query_article_true_label = "Sports" # 更正标签以匹配内容

    base_exemplars_data = all_data[1:10] # 使用9个基础样本
    target_exemplar = (target_record_text, query_article_true_label)
    base_exemplars = [(rec[2], class_map[int(rec[0])]) for rec in base_exemplars_data]

    exemplars_with_target = [target_exemplar] + base_exemplars
    exemplars_without_target = base_exemplars

    dp_classifier = DPICLForClassification(
        llm_callable=llm_classifier,
        class_labels=list(class_map.values()),
        epsilon=1.0, 
        delta=1e-3
    )
    
    # --- 白盒攻击 ---
    print(f"\n--- 运行白盒审计 (基于直方图) 对于 ε={dp_classifier.epsilon} ---", file=output_file)
    
    N_attack_samples = 50000 # 减少样本数以便快速演示
    print(f"正在为攻击模型生成 {N_attack_samples} 对直方图...", file=output_file)
    
    histograms_with_target = []
    histograms_without_target = []
    
    for i in range(N_attack_samples):
        _, noisy_hist_with = dp_classifier(exemplars=exemplars_with_target, query_str=query_article_text, num_subsets=10)
        histograms_with_target.append(noisy_hist_with)
        _, noisy_hist_without = dp_classifier(exemplars=exemplars_without_target, query_str=query_article_text, num_subsets=9)
        histograms_without_target.append(noisy_hist_without)
        print(f"  已生成样本 {(i+1)*2}/{N_attack_samples*2}", end='\r')
    print("\n数据生成完成。", file=output_file)

    X = np.array(histograms_with_target + histograms_without_target)
    y = np.array([1] * N_attack_samples + [0] * N_attack_samples)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    print(f"\n在 {len(X_train)} 个样本上训练逻辑回归分类器...", file=output_file)
    attack_classifier = LogisticRegression(random_state=42)
    attack_classifier.fit(X_train, y_train)

    # --- 寻找最优攻击阈值 (ROC分析) ---
    print("\n--- 寻找最优攻击阈值 (ROC分析) ---", file=output_file)
    # 1. 获取模型对每个样本属于正例(类别'1')的预测概率
    y_pred_probabilities = attack_classifier.predict_proba(X_test)[:, 1]

    # 2. 计算ROC曲线的点
    fprs, tprs, thresholds = roc_curve(y_test, y_pred_probabilities)

    # 3. 遍历所有阈值，找到使经验epsilon最大化的点
    max_empirical_epsilon = 0.0
    optimal_threshold = 0.0
    best_tpr = 0.0
    best_fpr = 0.0
    confidence = 0.95
    for fpr, tpr, threshold in zip(fprs, tprs, thresholds):
        # 避免除以零或log(0)的错误
        if fpr > 0 and tpr > 0:
            p_with_target_interval = clopper_pearson(tpr*N_attack_samples*0.3, N_attack_samples*0.3, confidence)
            p_without_target_interval = clopper_pearson(fpr*N_attack_samples*0.3, N_attack_samples*0.3, confidence)
            lower_p1 = p_with_target_interval[0]
            upper_p2 = p_without_target_interval[1]
            current_epsilon = math.log(lower_p1 / upper_p2)
            if current_epsilon > max_empirical_epsilon:
                max_empirical_epsilon = current_epsilon
                optimal_threshold = threshold
                best_tpr = tpr
                best_fpr = fpr

    print(f"搜索完成。攻击者能达到的最大经验 ε: {max_empirical_epsilon:.4f}", file=output_file)
    print(f"  - 最优决策阈值: {optimal_threshold:.4f}", file=output_file)
    print(f"  - 在此阈值下的真阳性率 (TPR): {best_tpr:.4f}", file=output_file)
    print(f"  - 在此阈值下的假阳性率 (FPR): {best_fpr:.4f}", file=output_file)

    # --- 使用最优阈值进行最终评估 ---
    print("\n--- 使用最优阈值进行最终的白盒攻击评估 ---", file=output_file)
    y_pred_optimal = (y_pred_probabilities >= optimal_threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred_optimal)
    report = classification_report(y_test, y_pred_optimal, target_names=['非成员 (D\')', '成员 (D)'])
    cm = confusion_matrix(y_test, y_pred_optimal)

    print(f"在最优阈值下的攻击分类器准确率: {accuracy:.4f}", file=output_file)
    print("\n详细分类报告:", file=output_file)
    print(report, file=output_file)
    # print("\n混淆矩阵:", file=output_file)
    # print(cm, file=output_file)
    tn, fp, fn, tp = cm.ravel()
    
    # 计算 TPR 和 FPR
    # TPR = TP / (TP + FN)
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    # FPR = FP / (FP + TN)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    print(f"真阴性 (TN): {tn}, 假阴性 (FN): {fn}, 真阳性 (TP): {tp}, 假阳性 (FP): {fp}", file=output_file)
    print(f"真阳性率 (TPR): {tpr:.4f}", file=output_file)
    print(f"假阳性率 (FPR): {fpr:.4f}", file=output_file)
    print(f"实际检验eps:{math.log(tpr/fpr):.4f}", file=output_file)
    T = N_attack_samples

    
    p_with_target_interval = clopper_pearson(tp, tp + fn, confidence)
    p_without_target_interval = clopper_pearson(fp, fp + tn, confidence)
    
    print(f"\n{confidence*100}% CI for P(With Target):   ({p_with_target_interval[0]:.4f}, {p_with_target_interval[1]:.4f})",file = output_file)
    print(f"{confidence*100}% CI for P(Without Target): ({p_without_target_interval[0]:.4f}, {p_without_target_interval[1]:.4f})",file = output_file)

    lower_p1 = p_with_target_interval[0]
    upper_p2 = p_without_target_interval[1]
    empirical_epsilon_whitebox = 0.0
    if upper_p2 > 0:
        empirical_epsilon = math.log(lower_p1 / upper_p2)
        print(f"\nEmpirically Estimated Epsilon (lower_bound): {empirical_epsilon:.4f}", file = output_file)
    elif tpr > 0 and fpr == 0:
        print("\n无法计算有限的经验性 ε，因为 FPR 为 0 (完美的区分能力)。这表明存在严重的隐私泄露。", file=output_file)
    else:
        print("\n无法计算经验性 ε，因为 TPR 或 FPR 为 0。", file=output_file)
    print("\n解释:", file=output_file)
    print("以上结果展示了该分类器能发起的'最强'攻击。最大经验ε是本次审计的核心发现，代表了在当前设定下最坏情况的实际隐私泄露水平。", file=output_file)

    output_file.close()
    print("\n白盒审计完成。结果已保存到 'output-whitebox-optimal.txt'。")