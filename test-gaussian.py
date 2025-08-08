# 完整代码示例

from scipy.stats import norm
import math

# --- 参数定义 ---
# 均值为 0
mu = 0

delta = 1e-5
epsilon = 1.0  # 隐私预算
sensitivity = math.sqrt(2)
sigma = (sensitivity * math.sqrt(2 * math.log(1.25 / delta))) / epsilon

def compute_probability_greater_than_A(mu, sigma, A):
    """
    计算高斯分布 N(mu, sigma^2) 中 P(X > A) 的概率。
    """
    # 创建正态分布对象
    dist = norm(loc=mu, scale=sigma)
    
    # 使用生存函数 (survival function, sf) 计算 P(X > A)
    return dist.sf(A)
# 阈值为 8
A = 22

computed_probability_1 = compute_probability_greater_than_A(mu, sigma, A)
print(f"P(X > {A}) = {computed_probability_1:.4f}")
A +=2
computed_probability_2 = compute_probability_greater_than_A(mu, sigma, A)
print(f"P(X > {A}) = {computed_probability_2:.4f}")

print(f"Advantage: {math.log(computed_probability_1 / computed_probability_2):.4f}")