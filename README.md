# Auditing DP-ICL (Formalization coming soon.)

This is the code repository for paper \textit{Tight and Practical Privacy Auditing for Differentially Private In-Context Learning}.

We have implemented auditing codes for both text classification and generation tasks. 

# Implementations
We here list the corresponding relationship between each folder and experiment we described in paper.
|folder name|algorithm|
| :----: | :----: |
|classification/mislabel|mislabel mia baseline for text classification task|
|classification/white|white-box access for text classification task|
|classification/single|black-box access and utility test for text classification task|
|generation/combine|Both black-box access and white-box access for text generation task|
|generation/baseline|Both black-box baseline and white-box baseline for text generation task|
|generation/utility|utility test for text generation task|
|generation/text-embedding|compute embedding distance between texts|

To run this codes, you need to change the model path and data path in both source code files and batch files. 

For each source code file, there is a corresponding batch file for running it.
