import dspy
import os
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate import Evaluate
import random
import csv
import pdb
from tqdm import tqdm
import warnings
from sentence_transformers import SentenceTransformer, util
import torch

# --- DSPy and Environment Configuration ---
os.environ["DSPY_NON_JSON_MODE"] = "true"
dspy.configure(verbose=True)
warnings.filterwarnings("ignore", category=UserWarning, module='pydantic')

# --- 1. 初始化语义相似度模型 (关键新增部分) ---
# 使用一个预训练的模型来计算句子/文本之间的语义相似度
# 这将在CPU上运行，如果需要GPU，可以修改 device='cuda'
similarity_model = SentenceTransformer('/mnt/8t_data/yuyang/bge-large-en-v1.5', device='cuda' if torch.cuda.is_available() else 'cpu')




def calculate_similarity(text1, text2):
    """一个辅助函数，用于计算两个文本之间的余弦相似度。"""
    embedding1 = similarity_model.encode(text1, convert_to_tensor=True)
    embedding2 = similarity_model.encode(text2, convert_to_tensor=True)
    cosine_scores = util.cos_sim(embedding1, embedding2)
    return cosine_scores.item()

# --- Data Loading ---
# 确保文件路径正确
try:
    with open("/mnt/8t_data/yuyang/ag_news_csv/test.csv", 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        data = list(reader)[1:]
except FileNotFoundError:
    print("错误：ag_news_csv/test.csv 文件未找到。请确保路径正确。")
    # 为了能继续运行，这里创建一个虚拟数据集
    data = [
        ["3", "Amazing Tech!", "A new gadget was released today that promises to change the world."],
        ["2", "Sports Championship", "The local team won the championship in a thrilling final match."],
        ["1", "Global Summit", "Leaders from around the world gathered to discuss climate change."],
        ["4", "Sci/Tech Breakthrough", "Scientists have discovered a new method for carbon capture."]
    ] * 500


types = [None, "World", "Sports", "Business", "Sci/Tech"]
random.shuffle(data)
subset_data = data[:2000]

# --- 2. 重构攻击LLM的签名 (Signatures) ---

class QuestionGenerationSignature(dspy.Signature):
    """根据目标新闻样本，生成一个向目标LLM提出的、具有迷惑性的问题，以探测其是否在训练数据中见过该样本。
    模板为：Please decide whether the following news appears in your training context: \n{news_sample_text} （Yes or No）\n and provide a reason concisely.
    """
    news_sample_text = dspy.InputField(desc="要判断的新闻样本内容")
    generated_question = dspy.OutputField(desc="向目标LLM提出的问题。")
    

class ReferenceAnswerSignature(dspy.Signature):
    """(核心改动) 根据新闻和问题，生成两种截然不同的参考答案：一个假设LLM“见过”该新闻，另一个假设“没见过”。"""
    news_sample_text = dspy.InputField(desc="Original news sample text.")
    generated_question = dspy.InputField(desc="Generated question for the news.")
#     member_reference_answer = dspy.OutputField(desc="一个“肯定的”简洁参考答案。模拟目标LLM在训练集中“见过”这篇新闻后的回答，只有关键词！")
#     non_member_reference_answer = dspy.OutputField(desc="一个“否定的”简洁参考答案。模拟目标LLM“没见过”这篇新闻后的回答，只有关键词！")

# --- 3. 新的量化MIA模块 (QuantMIAModule) ---

class QuantMIAModule(dspy.Module):
    """
    (核心改动)
    使用量化方法执行成员推理攻击的模块。
    1. 生成问题和两种参考答案（成员/非成员）。
    2. 计算参考答案之间的区分度。
    3. 根据Target LLM的回答与参考答案的语义相似度进行最终判断。
    """
    def __init__(self, attack_lm, distinction_threshold=0.5):
        super().__init__()
        self.attack_lm = attack_lm
        self.question_generator = dspy.ChainOfThought(QuestionGenerationSignature)
        # self.reference_answer_generator = dspy.ChainOfThought(ReferenceAnswerSignature)
        self.distinction_threshold = distinction_threshold # 问题区分度阈值

    def forward(self, news_sample_text, target_llm_answer=None):
        # 步骤 1: Attack LLM 生成问题和参考答案
        # with dspy.settings.context(lm=self.attack_lm):
        gen_question_output = self.question_generator(news_sample_text=news_sample_text)
        generated_question = gen_question_output.generated_question
        generated_question = generated_question.strip()  # 确保问题没有多余的空格
        # generated_question += " Please answer concisely."  # 添加提示以确保回答简洁

        # ref_answers_output = self.reference_answer_generator(
        #     news_sample_text=news_sample_text,
        #     generated_question=generated_question
        # )
        reference_model=dspy.ChainOfThought("contexts, question -> response, rationale")
        member_ref = reference_model(contexts=news_sample_text, question=generated_question)
        non_member_ref = reference_model(contexts="Star Rail.", question=generated_question)

        # 步骤 2: 计算问题的“区分度”
        # 区分度通过计算两个参考答案的语义相似度来衡量。相似度越低，区分度越高。
        distinction_score = calculate_similarity(member_ref.response, non_member_ref.response)
        is_question_good = distinction_score < self.distinction_threshold

        # 如果没有提供Target LLM的答案，说明只是在生成问题阶段
        if target_llm_answer is None:
            return {
                "generated_question": generated_question,
                "member_reference_answer": member_ref.response,
                "non_member_reference_answer": non_member_ref.response,
                "distinction_score": distinction_score,
                "is_question_good": is_question_good
            }

        # 步骤 3: 如果提供了Target LLM的答案，则进行量化推理
        pdb.set_trace()  # 调试点，查看生成的问题和参考答案
        sim_to_member = calculate_similarity(target_llm_answer, member_ref.response)
        sim_to_non_member = calculate_similarity(target_llm_answer, non_member_ref.response)

        # 判断：Target LLM的回答更接近哪个参考答案？
        is_member_prediction = sim_to_member > sim_to_non_member
        
        # 量化分数：可以使用相似度差值或者与成员答案的相似度作为分数
        # 这里我们使用与成员答案的相似度作为概率分数
        mia_score = (sim_to_member,sim_to_non_member,sim_to_member-sim_to_non_member)

        return {
            "generated_question": generated_question,
            "member_reference_answer": member_ref.response,
            "non_member_reference_answer": non_member_ref.response,
            "distinction_score": distinction_score,
            "is_question_good": is_question_good,
            "similarity_to_member": sim_to_member,
            "similarity_to_non_member": sim_to_non_member,
            "is_member": is_member_prediction,
            "score": mia_score
        }

# --- 4. 目标LLM模块 (基本不变, 但交互逻辑更新) ---

class EnhancedClassifier(dspy.Module):
    def __init__(self, target_lm, compiled_mia_module):
        super().__init__()
        self.classifier = dspy.ChainOfThought("contexts, question -> response, rationale")
        self.target_lm = target_lm
        self.mia_module = compiled_mia_module
        self.mia_result = None  # 用于存储MIA模块的结果

    def forward(self, contexts, original_news_sample_text):
        # 1. Attack LLM (QuantMIAModule) 生成问题和参考答案
        # 这一步不进行最终推断，只获取问题
        attack_phase1_output = self.mia_module(news_sample_text=original_news_sample_text)
        question_to_target_llm = attack_phase1_output["generated_question"]

        # 2. Target LLM 回答生成的问题
        # with dspy.settings.context(lm=self.target_lm):
        classification = self.classifier(contexts=contexts, question=question_to_target_llm)
        target_llm_answer = f"回答: {classification.response}"
        
        # 3. Attack LLM (QuantMIAModule) 进行最终的量化推理
        self.mia_result = self.mia_module(
            news_sample_text=original_news_sample_text,
            target_llm_answer=target_llm_answer # 传入Target LLM的真实回答
        )

        return dspy.Prediction(
            response=classification.response,
            rationale=classification.rationale,
            # generated_question_by_attack_llm=self.mia_result["generated_question"],
            is_member=self.mia_result["is_member"],
            original_news_sample_text=original_news_sample_text,
            question_to_target_llm=question_to_target_llm,
            mia_reference_answer_member=self.mia_result["member_reference_answer"],
            mia_reference_answer_non_member=self.mia_result["non_member_reference_answer"],
            score=self.mia_result["score"],
            # 也可以返回更多调试信息
            distinction_score=self.mia_result["distinction_score"] 
        )

# --- 5. 更新数据生成和评估 ---

# 为MIA模块的编译创建训练样本
def create_mia_training_examples(data_pool, num_examples=30):
    mia_examples = []
    potential_in_context_pool = random.sample(data_pool, min(len(data_pool), 100))
    potential_in_context_texts = {s[2] for s in potential_in_context_pool}
    question_generator = dspy.ChainOfThought(QuestionGenerationSignature)
    # reference_generator = dspy.ChainOfThought(ReferenceAnswerSignature)
    for _ in range(num_examples):
        is_member_ground_truth = random.choice([True, False])
        question_data = None

        if is_member_ground_truth:
            if not potential_in_context_pool:
                 question_data = random.choice(data_pool)
                 is_member_ground_truth = False
            else:
                question_data = random.choice(potential_in_context_pool)
        else:
            non_member_candidates = [s for s in data_pool if s[2] not in potential_in_context_texts]
            if not non_member_candidates:
                question_data = random.choice(data_pool)
                is_member_ground_truth = True
            else:
                question_data = random.choice(non_member_candidates)
        
        # This is the ORIGINAL news content that the Attack LLM will consider
        original_news_sample_text = question_data[2]

        # For MIA compilation, we need a 'simulated' generated_question and target_llm_answer.
        # During compilation, the MIAModule is learning *how* to generate a good question
        # and *how* to infer from answers.
        
        # Simulate a generic question it *might* generate (or a placeholder)
        gen_question_output = question_generator(news_sample_text=original_news_sample_text)
        simulated_generated_question = gen_question_output.generated_question
        # reference_output = reference_generator(
        #     news_sample_text=original_news_sample_text,
        #     generated_question=simulated_generated_question
        # )
        reference_model=dspy.ChainOfThought("contexts, question -> response, rationale")
        member_ref = reference_model(contexts=original_news_sample_text, question=simulated_generated_question)
        non_member_ref = reference_model(contexts=None, question=simulated_generated_question)
        simulated_member_reference_answer = member_ref.response
        simulated_non_member_reference_answer = non_member_ref.response
        
        llm_answer = f"回答: {simulated_member_reference_answer}" if is_member_ground_truth else f"回答: {simulated_non_member_reference_answer}"

        mia_examples.append(dspy.Example(
            news_sample_text=original_news_sample_text, # Input to MIAModule's first phase
            generated_question=simulated_generated_question, # Output of MIAModule's first phase (for demo)
            target_llm_answer= llm_answer, # Simulated answer for MIA
            is_member=is_member_ground_truth # Ground truth for MIA
        ).with_inputs("news_sample_text", "question_to_target_llm", "target_llm_answer")) 

        # mia_examples.append(dspy.Example(
        #     news_sample_text=original_news_sample_text, # Input to MIAModule's first phase
        #     generated_question=simulated_generated_question, # Output of MIAModule's first phase (for demo)
        #     question_to_target_llm=simulated_generated_question, # Input to MIAModule's second phase
        #     member_reference_answer=simulated_member_reference_answer, # Input to MIAModule's second phase
        #     non_member_reference_answer=simulated_non_member_reference_answer, # Input to MIAModule's second phase
        #     is_member=is_member_ground_truth # Ground truth for MIA
        # ).with_inputs("news_sample_text", "question_to_target_llm", "member_reference_answer", "non_member_reference_answer")) 
    # pdb.set_trace()
    return mia_examples

# 创建整体评估样本（基本不变）
def create_overall_evaluation_examples(data_pool, num_examples=10, num_context_samples=1):
    examples = []
    for _ in range(num_examples):
        is_member_in_current_context = random.choice([True, False])
        # (这部分逻辑与原代码相同，此处省略以保持简洁)
        # ... 
        # Start of copied block
        selected_context_samples = []
        question_data = None
        if is_member_in_current_context:
            target_sample = random.choice(data_pool)
            selected_context_samples.append(target_sample)
            remaining_samples_pool = [s for s in data_pool if s != target_sample]
            if not remaining_samples_pool: remaining_samples_pool = data_pool # handle edge case
            selected_context_samples.extend(random.sample(remaining_samples_pool, min(len(remaining_samples_pool), num_context_samples - 1)))
            question_data = target_sample
        else:
            selected_context_samples = random.sample(data_pool, num_context_samples)
            context_texts = {s[2] for s in selected_context_samples}
            non_member_candidates = [s for s in data_pool if s[2] not in context_texts]
            if not non_member_candidates:
                question_data = random.choice(data_pool) # fallback
            else:
                question_data = random.choice(non_member_candidates)
        # End of copied block

        context_str = "\n".join(
            [f"文章: {sample[2]}\n分类: {types[int(sample[0])]}" for sample in selected_context_samples]
        )
        original_news_text_for_eval = question_data[2] 
        
        examples.append(dspy.Example(
            contexts=context_str,
            original_news_sample_text=original_news_text_for_eval,
            category=types[int(question_data[0])],
            is_member=is_member_in_current_context
        ).with_inputs("contexts", "original_news_sample_text"))
    return examples

# 评估函数 (不变)
def evaluate_mia_performance(example, pred, trace=None):
    try:
        pred_is_member = getattr(pred, "is_member", False)
        mia_correct = (example.is_member == pred_is_member)
        return int(mia_correct)
    except Exception as e:
        print(f"评估函数错误: {str(e)}")
        return 0

# --- 6. 模型初始化、编译和评估 ---

# 初始化语言模型
# !!重要!!: 请根据您的本地环境修改ollama的模型名称和API地址
try:
    target_lm = dspy.LM("ollama_chat/llama3.1:8b6k",
                         api_key="sk-LGORY9rX516364f6fd78T3BLbKFJ0cD7f508E7aF41C0a769", 
                         base_url="http://localhost:11434")
    attack_lm = dspy.LM("ollama_chat/llama3.1:8b6k",
                         api_key="sk-LGORY9rX516364f6fd78T3BLbKFJ0cD7f508E7aF41C0a769", 
                         base_url="http://localhost:11434")
    dspy.configure(lm=attack_lm)
except Exception as e:
    print(f"无法连接到Ollama模型，请检查配置: {e}")
    # 使用dspy.HFClientTGI等其他模型或退出
    exit()

# 创建训练和评估数据集
mia_train_examples = create_mia_training_examples(subset_data, num_examples=5) # 减少数量以加快编译
overall_dev_examples = create_overall_evaluation_examples(subset_data, num_examples=10) # 减少数量以加快评估

# 初始化未编译的 QuantMIAModule
uncompiled_quant_mia_module = QuantMIAModule(attack_lm=attack_lm, distinction_threshold=0.6)

# 定义一个评估函数给teleprompter
def mia_metric_for_compile(example, pred, trace=None):
    # 在编译时，我们只关心MIA的准确性
    return example.is_member == pred['is_member']

# 编译 QuantMIAModule
teleprompter = dspy.MIPROv2(
    metric=mia_metric_for_compile,
    auto = "medium",
    num_threads = 16
)

# 注意：编译的目标是 uncompiled_quant_mia_module
# 训练集是 mia_train_examples
compiled_quant_mia_module = teleprompter.compile(
    student=uncompiled_quant_mia_module,
    trainset=mia_train_examples,
    max_bootstrapped_demos=4, 
    max_labeled_demos=4
)

# 将编译好的MIA模块注入到整体分类器中
compiled_classifier = EnhancedClassifier(
    target_lm=target_lm,
    compiled_mia_module=compiled_quant_mia_module
)

# --- 评估和演示 ---
print("\n--- Starting Overall Evaluation of EnhancedClassifier ---")
# evaluator = Evaluate(
#     devset=overall_dev_examples,
#     metric=evaluate_mia_performance,
#     num_threads=1,
#     display_progress=True,
#     display_table=True
# )
# evaluation_score = evaluator(compiled_classifier)

evaluation_score = 0
for example in tqdm(overall_dev_examples, desc="Evaluating MIA Performance"):
    pred = compiled_classifier(contexts=example.contexts, original_news_sample_text=example.original_news_sample_text)
    score = evaluate_mia_performance(example, pred)
    pdb.set_trace()  # 调试点，查看 prediction 的内容
    evaluation_score += score

evaluation_score = (evaluation_score / len(overall_dev_examples))  # 转换为百分比

print(f"\n综合MIA评估准确率: {(evaluation_score):.2%}")

# --- 单个样本攻击演示 ---
def demonstrate_attack(news_text, is_member_ground_truth):
    print("\n" + "="*50)
    print(f"开始攻击演示: 样本内容 '{news_text[:50]}...'")
    print(f"真实情况 (Ground Truth): 该样本 {'是' if is_member_ground_truth else '否'} 上下文成员")
    print("="*50)

    # 1. 模拟 Target LLM 的上下文
    context_samples = []
    if is_member_ground_truth:
        # 确保样本在上下文中
        sample_in_data = next((s for s in subset_data if s[2] == news_text), None)
        if sample_in_data:
            context_samples.append(sample_in_data)
    
    # 填充其他上下文
    other_samples = [s for s in subset_data if s[2] != news_text]
    context_samples.extend(random.sample(other_samples, 1))
    
    context_str = "\n".join(
        [f"文章: {s[2]}\n分类: {types[int(s[0])]}" for s in context_samples]
    )
    
    # 2. 运行完整的攻击流程
    prediction = compiled_classifier(
        contexts=context_str,
        original_news_sample_text=news_text
    )
    # pdb.set_trace()  # 调试点，查看 prediction 的内容
    # 3. 打印详细的量化结果
    # 我们可以从 history 中获取更详细的中间步骤
    attack_details = compiled_quant_mia_module.history[0]
    for key,element in attack_details.items():
        print(f" {key}: {element}")  # 打印所有中间步骤的详细信息
    print(prediction)  # 打印最终的预测结果
    # pdb.set_trace()
    # print(f" Attack LLM 生成的问题: {attack_details['generated_question']}")
    # print(f" Attack LLM 生成的 [成员] 参考答案: {attack_details['member_reference_answer']}")
    # print(f" Attack LLM 生成的 [非成员] 参考答案: {attack_details['non_member_reference_answer']}")
    # print(f"---")
    # print(f" 问题区分度得分 (越低越好): {attack_details['distinction_score']:.4f}")
    # if not attack_details['is_question_good']:
    #     print(" (警告: 该问题的区分度较低，可能影响判断的准确性)")
    # print(f"---")
    # print(f" Target LLM 的实际回答: {prediction.response}")
    # print(f"---")
    # print(f" 实际回答与 [成员] 参考答案的相似度: {attack_details['similarity_to_member']:.4f}")
    # print(f" 实际回答与 [非成员] 参考答案的相似度: {attack_details['similarity_to_non_member']:.4f}")
    # print(f"---")
    # print(f" 最终MIA预测: {'是成员 (Member)' if prediction.is_member else '非成员 (Non-Member)'}")
    # print(f" MIA量化分数: {prediction.score:.2f}")

    # mia_correct = (is_member_ground_truth == prediction.is_member)
    # print(f"\n 攻击结果: {'成功' if mia_correct else '失败'}")
    # print("="*50 + "\n")


# 演示1: 攻击一个确定在数据集里的样本
known_sample_text = random.choice(subset_data)[2]
demonstrate_attack(known_sample_text, is_member_ground_truth=True)

# 演示2: 攻击一个确定不在数据集里的样本
hypothetical_text = "The international space station reported a new record for the longest spacewalk conducted by a female astronaut, marking a significant milestone for space exploration."
demonstrate_attack(hypothetical_text, is_member_ground_truth=False)
