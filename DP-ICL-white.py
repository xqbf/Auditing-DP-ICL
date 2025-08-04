import numpy as np
import math
import csv
import dspy
from scipy.stats import beta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
# ==============================================================================
# 1. DSPy Setup (LLM Configuration)
# ==============================================================================
# As in your original code, configure the language models.
# Note: Using a real API key is required for this to run.
# The key provided is a placeholder and will not work.
try:
    # attack_lm = dspy.LM("together_ai/meta-llama/Llama-3-8b-chat-hf",
    #                      api_key="2f87bd2dace135491c98dd468ff09d757e30b0dbde9cfd091b2859020b9d0bc4", # Placeholder API Key
    #                      temperature=0.1)
    attack_lm = dspy.LM("ollama_chat/llama3.1:8b6k",
                        api_key="sk-LGORY9rX516364f6fd78T3BLbKFJ0cD7f508E7aF41C0a769", 
                        base_url="http://localhost:11434",
                        temperature=0.1)
    dspy.configure(lm=attack_lm)
except Exception as e:
    print(f"Could not configure DSPy LM. Please ensure you have a valid API key. Error: {e}")
    # Using a mock LLM for demonstration if API fails
    print("Using a Mock LLM for demonstration purposes.")
    class MockLM(dspy.LM):
        def __init__(self):
            super().__init__("mock-model")
        def __call__(self, prompt, **kwargs):
            # A simple mock that randomly returns a class label
            return dspy.Prediction(completions=[dspy.Completion(text=np.random.choice(["World", "Sports", "Business", "Sci/Tech"]))])
    attack_lm = MockLM()
    dspy.configure(lm=attack_lm)


# Define the signature for the classification task based on the paper's format.
class NewsClassification(dspy.Signature):
    """Classify the news article into one of the provided categories."""
    context = dspy.InputField(desc="Few-shot examples of news articles and their categories.")
    article = dspy.InputField(desc="The news article to classify, if you have seen the news article in those examples, you need to output the corresponding category in those examples directly.")
    category = dspy.OutputField(desc="One of: World, Sports, Business, Sci/Tech.")

# Create a reusable DSPy predictor module
dspy_predictor = dspy.Predict(NewsClassification)
# dspy_predictor=dspy.ChainOfThought(NewsClassification)

def llm_classifier(contexts: list[str], query_article: str, exemplars_labels: list[str]) -> str:
    """
    Simulates an LLM classifier using a few-shot prompt, as described in the paper.
    This function now performs actual classification instead of membership inference.
    """
    formatted_context = ""
    for text, label in zip(contexts, exemplars_labels):
        formatted_context += f"Article: {text}\nAnswer: {label}\n\n"
    
    result = dspy_predictor(context=formatted_context, article=query_article)
    
    predicted_category = result.category.split('\n')[0].strip()
    valid_labels = ["World", "Sports", "Business", "Sci/Tech"]
    if predicted_category not in valid_labels:
        return np.random.choice(valid_labels)
    return predicted_category

# ==============================================================================
# 2. DP-ICL Implementation (Unchanged)
# ==============================================================================
class DPICLForClassification:
    """
    Implements DP-ICL for text classification using private voting (Report-Noisy-Max with Gaussian Noise).
    """
    def __init__(self, llm_callable, class_labels: list, epsilon: float, delta: float):
        self.llm = llm_callable
        self.class_labels = class_labels
        self.num_classes = len(class_labels)
        self.label_to_id = {label: i for i, label in enumerate(class_labels)}
        self.id_to_label = {i: label for i, label in enumerate(class_labels)}
        self.epsilon = epsilon
        self.delta = delta

    def _private_voting(self, votes: list[str], output_file_handle=None) -> tuple[str, np.ndarray]:
        """Applies the Report-Noisy-Max with Gaussian Noise mechanism."""
        histogram = np.zeros(self.num_classes)
        for vote in votes:
            if vote in self.label_to_id:
                histogram[self.label_to_id[vote]] += 1

        sensitivity = math.sqrt(2)
        sigma = (sensitivity * math.sqrt(2 * math.log(1.25 / self.delta))) / self.epsilon
        
        noisy_histogram = histogram + np.random.normal(0, sigma, self.num_classes)
        
        winner_index = np.argmax(noisy_histogram)

        if output_file_handle:
             print(histogram, noisy_histogram, end = " ", file=output_file_handle)

        return (self.id_to_label[winner_index], noisy_histogram)

    def __call__(self, exemplars: list[tuple[str, str]], query_str: str, num_subsets: int, output_file_handle=None) -> tuple[str, np.ndarray]:
        """Executes the full DP-ICL process."""
        if not exemplars or num_subsets <= 0:
            raise ValueError("Exemplars list cannot be empty and num_subsets must be positive.")
        
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
        
        return self._private_voting(votes, output_file_handle)

# ==============================================================================
# 3. Privacy Audit Helper Functions (Unchanged)
# ==============================================================================
def clopper_pearson(successes, trials, confidence_level):
    if trials == 0: return (0.0, 1.0)
    alpha = 1 - confidence_level
    lower = beta.ppf(alpha / 2, successes, trials - successes + 1)
    upper = beta.ppf(1 - alpha / 2, successes + 1, trials - successes)
    return (lower if not np.isnan(lower) else 0.0, upper if not np.isnan(upper) else 1.0)

# ==============================================================================
# 4. Example Usage and Privacy Audits
# ==============================================================================
if __name__ == '__main__':
    # --- Data and Task Setup ---
    try:
        with open("/mnt/8t_data/yuyang/ag_news_csv/test.csv", 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            all_data = [(row[0], row[1], row[2]) for row in list(reader)[1:]]
    except FileNotFoundError:
        print("Error: `ag_news_csv/test.csv` not found. Please ensure the dataset is in the correct path.")
        exit()

    output_file = open('output-whitebox.txt', 'w', encoding = 'utf-8')
    
    print("="*20 + " DP-ICL Classification & Privacy Audit " + "="*20, file=output_file)
    class_map = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}
    
    # --- Privacy Audit Parameters ---
    target_record_text = "Real Madrid has signed a new football player."
    query_article_text = target_record_text 
    query_article_true_label = "Business" 

    # Create base exemplars and the target exemplar
    base_exemplars_data = all_data[1:20] # Using 9 other records
    target_exemplar = (target_record_text, query_article_true_label)
    base_exemplars = [(rec[2], class_map[int(rec[0])]) for rec in base_exemplars_data]

    # D: Dataset WITH the target record
    exemplars_with_target = [target_exemplar] + base_exemplars
    # D': Dataset WITHOUT the target record
    exemplars_without_target = base_exemplars

    # --- DP-ICL Instantiation ---
    dp_classifier = DPICLForClassification(
        llm_callable=llm_classifier,
        class_labels=list(class_map.values()),
        epsilon=1.0, 
        delta=1e-3
    )
    
    # --- Audit 1: Black-Box Attack (Original Method) ---
    # T = 1000 
    # print(f"\n--- Running Black-Box Audit (Output-based) with {T} trials for ε={dp_classifier.epsilon} ---", file=output_file)
    
    # count_with_target = 0
    # for i in range(T):
    #     predicted_label, _ = dp_classifier(
    #         exemplars=exemplars_with_target, query_str=query_article_text, num_subsets=8
    #     )
    #     if predicted_label == query_article_true_label:
    #         count_with_target += 1
    #     print(f"  Trial {i+1}/{T} (With Target)", end='\r')
    # print()

    # count_without_target = 0
    # for i in range(T):
    #     predicted_label, _ = dp_classifier(
    #         exemplars=exemplars_without_target, query_str=query_article_text, num_subsets=8
    #     )
    #     if predicted_label == query_article_true_label:
    #         count_without_target += 1
    #     print(f"  Trial {i+1}/{T} (Without Target)", end='\r')
    # print()
    
    # print("\n\n--- Black-Box Audit Results ---", file=output_file)
    # prob_with_target = count_with_target / T
    # prob_without_target = count_without_target / T
    # print(f"P(Correct Prediction | With Target)  = {prob_with_target:.4f}", file=output_file)
    # print(f"P(Correct Prediction | Without Target)= {prob_without_target:.4f}", file=output_file)
    
    # confidence = 0.95
    # p_with_target_interval = clopper_pearson(count_with_target, T, confidence)
    # p_without_target_interval = clopper_pearson(count_without_target, T, confidence)
    
    # upper_p1, lower_p2 = p_with_target_interval[1], p_without_target_interval[0]
    # if lower_p2 > 0:
    #     empirical_epsilon = math.log(upper_p1 / lower_p2)
    #     print(f"Empirically Estimated Epsilon (black-box): {empirical_epsilon:.4f}", file=output_file)
    # else:
    #     print("Could not compute empirical epsilon (black-box)", file=output_file)
        
    # print("\n" + "="*65, file=output_file)


    # ==============================================================================
    # 5. NEW: White-Box Membership Inference Attack on Noisy Histograms
    # ==============================================================================
    print(f"\n--- Running White-Box Audit (Histogram-based) for ε={dp_classifier.epsilon} ---", file=output_file)
    
    N_attack_samples = 50000 # Number of samples for training the attack model (half with, half without)
    print(f"Generating {N_attack_samples} pairs of histograms for the attack model...", file=output_file)
    
    # --- Step 1: Generate Data ---
    histograms_with_target = []
    histograms_without_target = []
    
    for i in range(N_attack_samples):
        # Generate one histogram for D (with target)
        _, noisy_hist_with = dp_classifier(
            exemplars=exemplars_with_target, query_str=query_article_text, num_subsets=10
        )
        histograms_with_target.append(noisy_hist_with)
        
        # Generate one histogram for D' (without target)
        _, noisy_hist_without = dp_classifier(
            exemplars=exemplars_without_target, query_str=query_article_text, num_subsets=10
        )
        histograms_without_target.append(noisy_hist_without)
        print(f"  Generated sample {2*(i+1)}/{2*N_attack_samples}", end='\r')
    print("\nData generation complete.", file=output_file)

    # --- Step 2: Prepare Dataset for Scikit-learn ---
    # Create feature matrix X (the histograms) and label vector y (0 or 1)
    X = np.array(histograms_with_target + histograms_without_target)
    y = np.array([1] * N_attack_samples + [0] * N_attack_samples) # 1 = member, 0 = non-member
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\nTraining a Logistic Regression classifier on {len(X_train)} samples...", file=output_file)

    # --- Step 3: Train and Evaluate the Attack Classifier ---
    attack_classifier = LogisticRegression(random_state=42)
    attack_classifier.fit(X_train, y_train)

    # --- 修改阈值 ---
    # 1. 获取模型对每个样本属于正例(类别'1')的预测概率
    # predict_proba() 返回一个数组，[概率属于0, 概率属于1]，我们取第二列
    y_pred_probabilities = attack_classifier.predict_proba(X_test)[:, 1]

    # 2. 设置你自己的新阈值
    # 默认阈值是 0.5。降低阈值会把更多样本预测为正例(1)，提高TPR/召回率。
    custom_threshold = 0.65  # 你可以改成任何你想要的值，比如 0.3, 0.6 等

    # 3. 根据新阈值生成最终预测结果
    # 如果概率 >= custom_threshold，则预测为 1，否则为 0
    y_pred = (y_pred_probabilities >= custom_threshold).astype(int)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Non-Member (D\')', 'Member (D)'])

    # --- Step 4: Report Results ---
    print("\n--- 白盒攻击结果 ---", file=output_file)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"攻击分类器准确率: {accuracy:.4f}", file=output_file)
    print("\n详细分类报告:", file=output_file)
    report = classification_report(y_test, y_pred, target_names=['非成员 (D\')', '成员 (D)'])
    print(report, file=output_file)

    # --- 新增: 从 TPR/FPR 计算经验性 Epsilon ---
    print("\n--- 经验性 Epsilon 计算 (白盒) ---", file=output_file)
    
    # 计算混淆矩阵以获得TP, FP, TN, FN
    cm = confusion_matrix(y_test, y_pred)
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

    confidence = 0.95
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
    print(f"这个经验性 ε ({empirical_epsilon_whitebox:.4f}) 是从攻击者的角度量化的实际隐私损失。", file=output_file)
    print(f"它应该与理论上设定的 ε ({dp_classifier.epsilon}) 进行比较。如果经验值远低于理论值，说明机制在该攻击下的表现比最坏情况的保证要好。", file=output_file)

    output_file.close()

    print("\n白盒审计完成。结果已保存到 'output-whitebox.txt'。")