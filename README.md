# Power Analysis for Interleaving Experiments



This is source code for the following paper:

Hosein Azarbonyad and Evangelos Kanoulas, "Power Analysis for Interleaving Experiments by means of Offline Evaluation", In proceedings of The 2nd ACM International Conference on the Theory of Information Retrieval, 2016.

The goal is to analyze the correlation of offline and online evaluation information retrieval measures and provide insights of the number of required impressions to discover that two systems (A and B) are significantly different in online setting using the results achieved in offline evaluation setting. 

# Prerequisites
* Python 2.7
* [Pyclick](https://github.com/markovi/PyClick)
* Numpy
* Scipy

# Usage
All offline measures are implemented in common.py file. You can add your own measures to this file. runExperiment.py contains an step by step example. Pyclick is used for training click models on Yandex Relevance Prediction challenge click log. runExperiment.py demonstrates how to learn the parameters of User Browsing Model (UBM) click model. Any other click model can be used similarly.





