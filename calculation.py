from scipy.stats import beta
import math
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

T = 1000  # 实验总次数
c0 = 540
c1 = 440   # 成功的次数
a = 0.95 # 置信水平

# 计算置信区间
lower, upper = clopper_pearson(c0,c1, T, a)

# 打印结果
print(f"实验总次数: {T}")
print(f"成功的次数: {c1}")
print(f"置信水平: {a}")
print(f"Clopper-Pearson 置信区间下界: {lower:.4f}")
print(f"Clopper-Pearson 置信区间上界: {upper:.4f}")


eps = math.log(upper/lower)
print(f"计算出的 epsilon: {eps:.4f}")