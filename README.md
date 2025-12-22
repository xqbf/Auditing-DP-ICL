# Auditing DP-ICL (Formalization coming soon.)

This is the code repository for paper Tight and Practical Privacy Auditing for Differentially Private In-Context Learning.

We have implemented auditing codes for both text classification and generation tasks. 

# Datasets

We here list the datasets used in our paper.
|datasets|url|type|
| :----: | :----: | :----: |
|AGnews|https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset|text-classification|
|TREC|https://www.kaggle.com/datasets/thedevastator/the-trec-question-classification-dataset-a-longi|text-classification|
|Samsum|https://huggingface.co/datasets/knkarthick/samsum|text-generation|
|Xsum|https://huggingface.co/datasets/EdinburghNLP/xsum|text-generation|

# Implementations
We here list the corresponding relationship between each folder and experiment we described in paper.
|folder name|experiment|
| :----: | :----: |
|classification/mislabel|Mislabel mia baseline for text classification task|
|classification/auditing|Both black-box access and white-box access for text classification task|
|classification/utility|Utility test on AGnews dataset|
|generation/combine|Both black-box access and white-box access for text generation task|
|generation/baseline|Both black-box baseline and white-box baseline for text generation task|
|generation/utility|utility test for text generation task|


