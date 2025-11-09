from openai import OpenAI
import numpy as np

def euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)
client = OpenAI(api_key="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")

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    # 转换成 numpy array，避免 list 相减报错
    return np.array(response.data[0].embedding, dtype=np.float32)

if __name__ == "__main__":
    pairs = [
    ('The old clock chimed a forgotten, dusty tune.',
         "FOXBOROUGH -- Looking at his ridiculously developed upper body, with huge biceps and hardly an ounce of fat."),
        
        # Pair 1: 语义相似（时钟）
        ('The ancient timepiece sang a quiet, lost melody.',
         'The dusty grandfather clock announced the hour with a frail chime.'),
        
        # Pair 2: 语义相似（体育）
        ('The athlete showcased incredible strength, lifting the heavy weights with ease.',
         'His muscular physique was the result of years of dedicated training and hard work.'),

        # Pair 3: 句式相似，主题不同
        ('[Incorrect statement,]',
         '[]'),

        # Pair 4: 几乎相同（最小差异）
        ('Yes.',
         'No.')
]

for w1, w2 in pairs:
    v1 = get_embedding(w1)
    v2 = get_embedding(w2)
    dist = euclidean_distance(v1, v2)
    print(f"{w1} vs {w2} -> distance = {dist:.4f} (>= sqrt(2)? {dist >= np.sqrt(2)})")  