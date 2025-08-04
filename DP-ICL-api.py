import numpy as np
import math
import csv
import dspy
from scipy.stats import beta

# ==============================================================================
# 1. DSPy Setup (LLM Configuration)
# ==============================================================================
# As in your original code, configure the language models.
# target_lm = dspy.LM(...) # You can use a different model for the main task if needed.
# attack_lm = dspy.LM("together_ai/meta-llama/Llama-3-8b-chat-hf",
#                      api_key="2f87bd2dace135491c98dd468ff09d757e30b0dbde9cfd091b2859020b9d0bc4", # Please replace with your actual API key
#                      temperature=0.1)
attack_lm = dspy.LM("ollama_chat/llama3.1:8b6k",
                        api_key="sk-LGORY9rX516364f6fd78T3BLbKFJ0cD7f508E7aF41C0a769", 
                        base_url="http://localhost:11434",
                        temperature=0.1)
dspy.configure(lm=attack_lm)

# Define the signature for the classification task based on the paper's format.
class NewsClassification(dspy.Signature):
    """Classify the news article into one of the provided categories."""
    context = dspy.InputField(desc="Few-shot examples of news articles and their categories.")
    article = dspy.InputField(desc="The news article to classify.")
    category = dspy.OutputField(desc="One of: World, Sports, Business, Sci/Tech.")

# Create a reusable DSPy predictor module
dspy_predictor = dspy.Predict(NewsClassification)
# dspy_predictor=dspy.ChainOfThought(NewsClassification)

def llm_classifier(contexts: list[str], query_article: str, exemplars_labels: list[str]) -> str:
    """
    Simulates an LLM classifier using a few-shot prompt, as described in the paper[cite: 18, 648].
    This function now performs actual classification instead of membership inference.

    :param contexts: A list of exemplar article texts.
    :param query_article: The article text to be classified.
    :param exemplars_labels: The corresponding labels for the exemplar articles.
    :return: The predicted class label string.
    """
    # Format the few-shot context according to the AGNews template in Table 6.
    # Template: "Article: {text} Answer: {label}"
    formatted_context = ""
    for text, label in zip(contexts, exemplars_labels):
        formatted_context += f"Article: {text}\nAnswer: {label}\n\n"
    
    # Use the DSPy predictor to get the classification
    result = dspy_predictor(context=formatted_context, article=query_article)
    
    # Simple post-processing to ensure the output is one of the valid labels.
    # A more robust solution might involve forcing the output token via logit_bias if the API supports it.
    predicted_category = result.category.split('\n')[0].strip()
    valid_labels = ["World", "Sports", "Business", "Sci/Tech"]
    if predicted_category not in valid_labels:
        # If the model gives an invalid output, we can treat it as a random guess or failure.
        # For simplicity, we'll just pick one randomly.
        return np.random.choice(valid_labels)
    return predicted_category

# ==============================================================================
# 2. DP-ICL Implementation (Unchanged from your code)
# ==============================================================================
class DPICLForClassification:
    """
    Implements DP-ICL for text classification using private voting (Report-Noisy-Max with Gaussian Noise).
    This implementation correctly follows Section 3.1 and Algorithm 1 of the paper[cite: 114, 471].
    """
    def __init__(self, llm_callable, class_labels: list, epsilon: float, delta: float):
        self.llm = llm_callable
        self.class_labels = class_labels
        self.num_classes = len(class_labels)
        self.label_to_id = {label: i for i, label in enumerate(class_labels)}
        self.id_to_label = {i: label for i, label in enumerate(class_labels)}
        self.epsilon = epsilon
        self.delta = delta

    def _private_voting(self, votes: list[str]) -> tuple[str, np.ndarray]:
        """Applies the Report-Noisy-Max with Gaussian Noise mechanism."""
        # Step 1: Create an output histogram [cite: 115, 117]
        histogram = np.zeros(self.num_classes)
        for vote in votes:
            if vote in self.label_to_id:
                histogram[self.label_to_id[vote]] += 1

        # Sensitivity (Δ) is sqrt(2) as changing one exemplar affects at most two counts
        sensitivity = math.sqrt(2)
        # Calculate sigma based on the Gaussian mechanism's privacy guarantee from Theorem 3 [cite: 527]
        sigma = (sensitivity * math.sqrt(2 * math.log(1.25 / self.delta))) / self.epsilon
        
        # Step 2: Add Gaussian noise to each count [cite: 118, 127]
        noisy_histogram = histogram + np.random.normal(0, sigma, self.num_classes)
        
        # Step 3: Release the class with the highest noisy count [cite: 119, 128]
        winner_index = np.argmax(noisy_histogram)
        return (self.id_to_label[winner_index], noisy_histogram)

    def __call__(self, exemplars: list[tuple[str, str]], query_str: str, num_subsets: int) -> tuple[str, np.ndarray]:
        """Executes the full DP-ICL process."""
        if not exemplars or num_subsets <= 0:
            raise ValueError("Exemplars list cannot be empty and num_subsets must be positive.")
        
        # Phase I: Partition the exemplars [cite: 101, 471]
        # For simplicity, using simple splitting. The paper also discusses Poisson sampling[cite: 108].
        shuffled_indices = np.random.permutation(len(exemplars))
        partition_indices = np.array_split(shuffled_indices, num_subsets)
        
        # Phase II & III: Prompt the model for each partition [cite: 102, 103]
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
        
        # Phase IV: Privately aggregate the votes [cite: 104]
        return self._private_voting(votes)

# ==============================================================================
# 3. Privacy Audit Helper Functions
# ==============================================================================
def clopper_pearson(successes, trials, confidence_level):
    """
    Calculates the Clopper-Pearson confidence interval for a binomial proportion.
    This is a standard statistical method.
    """
    if trials == 0:
        return (0.0, 1.0)
    alpha = 1 - confidence_level
    lower = beta.ppf(alpha / 2, successes, trials - successes + 1)
    upper = beta.ppf(1 - alpha / 2, successes + 1, trials - successes)
    return (lower if not np.isnan(lower) else 0.0, upper if not np.isnan(upper) else 1.0)

# ==============================================================================
# 4. Example Usage and Privacy Audit
# ==============================================================================
if __name__ == '__main__':
    
    
    # --- Data and Task Setup ---
    try:
        with open("/mnt/8t_data/yuyang/ag_news_csv/test.csv", 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            # Format: [(class_id, title, description), ...]
            all_data = [(row[0], row[1], row[2]) for row in list(reader)[1:]]
    except FileNotFoundError:
        print("Error: `ag_news_csv/test.csv` not found. Please ensure the dataset is in the correct path.")
        exit()

    # Class labels from the paper for AGNews 
    class_map = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}
    
    # --- Privacy Audit Parameters ---
    # We will test if including a specific exemplar (target_record)
    # leaks information about its true class.
    target_record = all_data[0] # The record to include/exclude
    base_exemplars_data = all_data[1:20] # 5 other records as base context
    
    query_article_text = target_record[2]
    query_article_true_label = class_map[int(target_record[0])]

    # Convert data into (text, label) tuples
    target_exemplar = (query_article_text, query_article_true_label)
    base_exemplars = [(rec[2], class_map[int(rec[0])]) for rec in base_exemplars_data]

    # D: Dataset WITH the target record
    exemplars_with_target = [target_exemplar] + base_exemplars
    # D': Dataset WITHOUT the target record
    exemplars_without_target = base_exemplars

    # --- DP-ICL Instantiation ---
    dp_classifier = DPICLForClassification(
        llm_callable=llm_classifier,
        class_labels=list(class_map.values()),
        epsilon=1.0,  # As used in the paper's experiments [cite: 233]
        delta=1e-3    # A common delta value for DP
    )
    f = open('output-api.txt', 'w',encoding = 'utf-8')
    print("="*20 + " DP-ICL Classification & Privacy Audit " + "="*20, file = f)
    # --- Run Audit Simulation ---
    T = 1000 # Number of trials for the audit. Increase for more stable results.
    print(f"Running privacy audit with {T} trials for ε={dp_classifier.epsilon}...",file = f)
    
    # Experiment 1: Target record IS IN the context
    print("\nRunning Experiment 1 (target record INCLUDED in exemplars)...", file = f)
    count_with_target = 0
    for i in range(T):
        predicted_label, _ = dp_classifier(
            exemplars=exemplars_with_target,
            query_str=query_article_text,
            num_subsets=10 # Partition 6 exemplars into 3 subsets of 2
        )
        if predicted_label == query_article_true_label:
            count_with_target += 1
        print(f"  Trial {i+1}/{T}: Predicted '{predicted_label}'", end='\r',file = f)

    # Experiment 2: Target record IS NOT IN the context
    print("\n\nRunning Experiment 2 (target record EXCLUDED from exemplars)...", file = f)
    count_without_target = 0
    for i in range(T):
        # We need to make sure the number of exemplars is consistent.
        # Here we use a different set of 5+1 exemplars for a fair comparison.
        # Or, we can proceed with a smaller set for D'. Let's do that for a direct test.
        predicted_label, _ = dp_classifier(
            exemplars=exemplars_without_target,
            query_str=query_article_text,
            num_subsets=10 # Partition 5 exemplars into 2 subsets
        )
        if predicted_label == query_article_true_label:
            count_without_target += 1
        print(f"  Trial {i+1}/{T}: Predicted '{predicted_label}'", end='\r',file = f)
    
    print("\n\n--- Audit Results ---",file = f)
    
    # Calculate probabilities of correctly classifying the query article
    prob_with_target = count_with_target / T
    prob_without_target = count_without_target / T

    print(f"Query Article: '{query_article_text}'",file = f)
    print(f"Query Article True Label: '{query_article_true_label}'",file = f)
    print(f"P(Correct Prediction | With Target)  = {prob_with_target:.4f} ({count_with_target}/{T})",file = f)
    print(f"P(Correct Prediction | Without Target)= {prob_without_target:.4f} ({count_without_target}/{T})", file = f)

    # --- Empirical Epsilon Calculation ---
    # We calculate the confidence interval for each probability to get a more robust
    # estimate of the privacy loss.
    confidence = 0.95
    p_with_target_interval = clopper_pearson(count_with_target, T, confidence)
    p_without_target_interval = clopper_pearson(count_without_target, T, confidence)
    
    print(f"\n{confidence*100}% CI for P(With Target):   ({p_with_target_interval[0]:.4f}, {p_with_target_interval[1]:.4f})",file = f)
    print(f"{confidence*100}% CI for P(Without Target): ({p_without_target_interval[0]:.4f}, {p_without_target_interval[1]:.4f})", file = f)
    
    # The empirical epsilon is the log of the ratio of probabilities.
    # To find the worst-case, we take the upper bound of the "with target" prob
    # and the lower bound of the "without target" prob.
    # log( Pr(E|D) / Pr(E|D') ) <= epsilon
    lower_p1 = p_with_target_interval[0]
    upper_p2 = p_without_target_interval[1]

    if upper_p2 > 0:
        empirical_epsilon = math.log(lower_p1 / upper_p2)
        print(f"\nEmpirically Estimated Epsilon (lower_bound): {empirical_epsilon:.4f}", file = f)

    print("\nNote: A lower empirical epsilon suggests better privacy protection. " \
          "If it's close to the theoretical epsilon you set, the mechanism is behaving as expected.",file = f)