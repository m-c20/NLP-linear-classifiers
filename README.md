# NLP Classification: Naive Bayes vs Logistic Regression

## Problem Setup
This experiment aims to explore the performance of Naive Bayes (NB) and Logistic Regression (LR) in classifying semantics and spelling within sentences. Two different tests were conducted. Task1 (semantics): Distinguishing sentences that contain universal quantifiers (for example "All...") versus particular quantifiers (for example "None..."). Task2 (spelling): Sorting sentences that start with at least three words that start with the same letter (e.g. "the tall tree...") versus otherwise (for instance "blue car..."). For each task, the sentences were pre-processed in multiple ways: POS tagging, stopword removal, and n-gram features. We hypothesize that when data are pre-processed using uni-gram features, both models will achieve accuracy above 55% in Task1. Furthermore, we expect both models to achieve the highest F1 score when the data are pre-processed using POS tagging in Task2 since the sentence features in Task2 require more adjectives within sentences that start with at least three words that start with the same letter. 

## Dataset & Procedure
The dataset used in this experiment is a mix of sentences obtained from general AI-generated sources, online free corpus, and human written sentences. The data were manually inspected by humans and modified to ensure correctness and to increase variability.

For each Task1 and Task2 the procedure is as follows, the dataset is labeled by adding 0 or 1 to each sentence in the data. Each sentence is labeled 1 (positive) if it contains the target feature, or 0 (negative) otherwise. Labeling is automated by `script1.py`. After that, the sentence is preprocessed (e.g. stopword removal). Then, the classifiers process the files to split the data with their corresponding labels into training and testing set. At the later phase, the data were vectorized and the proposed NLP models were applied. Finally, accuracy, recall, and F1 scores were reported. 

## Parameter Settings
In both experiment one and two, the data were split into two sets: 80% training and 20% testing. Python's random seed is set to 42 to ensure reproducibility. Two linear classification models were selected: Naive Bayes and Logistic Regression (max_iter set to 1000). Three pre-processing settings were applied: unigram features, POS tagging, and stopword removal. Three tests were conducted per task. The results are reported in the table below.

## Results & Conclusions
The test results indicate that both classifiers achieved their highest F1 scores in Task1 when the data were pre-processed with uni-gram features, while the performance dropped when the data were pre-processed with POS tagging. This may be attributed to the fact that universal and particular quantifiers are more effectively captured through frequency based representation rather than POS tagging which abstracts away the meaning of quantifiers. In contrast, the classifiers demonstrated the highest performance on Task2 when the data were pre-processed with POS tagging. This is an expected outcome, POS tagging facilitates the identification of sentences that are adjective heavy. A pattern that is more likely to occur in sentences that start with three words that start with the same letter.  
Notably, both classifiers exhibited near random performance (approximately 50% accuracy) in Task 2 when stopword removal or uni-gram features were applied, with an exception in which LR with uni-gram pre-processing reporting 71% accuracy. 

## Limitations
As general NLP studies state, both Logistic Regression and Naive Bayes in NLP have non trivial data requirements. The data used in this study are limited with 140 sentences for Task1 and 233 sentences for Task2. The data used in the experiments were generated using general AI sources, free internet corpus and human writing, which are biased. For instance, all generative AI models used tended to produce sentences with "repeated syntax order", while human written samples were authored by a non-native English speaker. Such factors may influence linguistic patterns learned by the classifiers.  

## Results Table

| Dataset   | Method                | Acc.  | Rec.  | F1    |
|-----------|----------------------|-------|-------|-------|
| Dataset 1 | NB (uni-gram)        | 0.96  | 0.90  | 0.947 |
|           | NB (POS Tags)        | 0.64  | 0.60  | 0.571 |
|           | NB (stopword removal)| 0.80  | 0.70  | 0.737 |
|           | LR (uni-gram)        | 1.00  | 1.00  | 1.000 |
|           | LR (POS Tags)        | 0.68  | 0.80  | 0.667 |
|           | LR (stopword removal)| 0.84  | 0.80  | 0.800 |
| Dataset 2 | NB (uni-gram)        | 0.55  | 0.54  | 0.588 |
|           | NB (POS Tags)        | 0.83  | 0.93  | 0.867 |
|           | NB (stopword removal)| 0.55  | 0.57  | 0.604 |
|           | LR (uni-gram)        | 0.60  | 0.71  | 0.678 |
|           | LR (POS Tags)        | 0.89  | 1.00  | 0.918 |
|           | LR (stopword removal)| 0.51  | 0.43  | 0.511 |

**Formulas:**

