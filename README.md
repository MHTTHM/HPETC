# Traffic Classification Code - O2OS

## 0. Table of Contents & Links

- [1. Increase ratio](#1-ol_classifier_ratio)
- [2. Baseline](#2-baseline)
- [3. Nop-O2OS](#3-nop-o2os-method)
- [4. Borderline Tech.](#4-borderline-optimization)
- [5. Self-adaptive Tech.](#5-self-adaptive-optimization)
- [6. Ensemble Tech.](#6-ensemble-optimization)
- [7. O2OS-entirely](#7-entirety-optimization)
- [8.1 Supplements-Featrues](#statistical-featrues)
- [8.2 Supplements-Models](#model-parameters)

## 1. OL_Classifier_ratio

Code for two-stage classification (online identification and offline classification).

The **modelpath** is the location where well-trained models are stored. The models are saved in the format of `1_XX.m` and `2_XX.m`, which are used for the first-stage online identification and the second-stage offline classification, respectively. The features used are `pl_iat` and `select_features` in `conf.py`.

**tor** and **nor** are directories for storing the testing datasets for TOR and normal traffic, respectively.

## 2. Baseline

The original traffic classification method. Modify the data reading part in `Classifier.py` and run the code. The models are saved in the "models" folder.

## 3. NOP-O2OS Method

Initial classification performance of O2OS.

- Model Generation: Modify the `modelpath` to specify the model storage path in `OL_Classifier_models.py`. Set **borderline=0** and **adaboostout = 0**.

- Model Testing: Modify **ensembleout = 0** in `OL_Classifier_ratio.py`. Select the test dataset paths for NOR and the model path, and modify the values of `lowbound` and `uppbound`.

## 4. Borderline Optimization

Generate samples using the borderline method and modify the distribution of the test dataset to bias the classifier.

- Model Generation: Modify the `modelpath` to specify the model storage path in `OL_Classifier_models.py`. Set **borderline=1** and **adaboostout = 0**. You can modify the sample quantity to be generated and which class to increase, which will bias the classifier towards that class.

- Model Testing: Modify **ensembleout = 0** in `OL_Classifier_ratio.py`. Select the test dataset paths for NOR and the model path, and modify the values of `lowbound` and `uppbound`.

## 5. Self-adaptive Optimization

Adopting the Adaboost concept, the training samples for the second-stage offline classification are not normal samples, but rather TOR samples and the misclassified samples from the first-stage online identification.

- Model Generation:

1. In `OL_Classifier_models.py`, **ensure that the normal training traffic in `norpath` is abundant**. Select an appropriate number of normal samples for training and save the misclassified results. Modify the `modelpath` to specify the model storage path, and set **borderline=0** and **adaboostout = 1**. Modify `adaboostpath` to store the misclassified results.

2. Run `OL_Classifier_models_adaboost.py`. Modify the `norpath` and `modelpath`, where `modelpath` should be the same as the previous one. After running, go to the model path. The newly generated models are named in the format of 2_*.m, where * is a number. The order is c45, gbdt, knn, rf30.

- Model Testing: Modify **ensembleout = 0** in `OL_Classifier_ratio.py`. Select the test dataset paths for NOR and the model path, and modify the values of `lowbound` and `uppbound`.

## 6. Ensemble Optimization

Adopting the ensemble learning concept to integrate the results from different classifiers.

- Preliminary Classification Results: In `OL_Classifier_ratio.py`, run a pre-existing model, which can be any of the previous baseline, borderline, or adaboost methods. However, set **ensembleout = 1** to save the original labels and classification results in `ensemble.csv`.

- Integration of Classification Results: In `ensemble.py`, modify the `file` to the path of `ensemble.csv`. There are three ensemble strategies: naive, slightly, and strong, representing different rejection granularities. Modify the option in `conf.py` to select ensemble strategy.

1. Naive: Majority voting. If there is a 2:2 tie or all four classification results are different, classify it as normal.
2. Slightly: Weak negation. If the classification is normal, classify it as normal; otherwise, follow the majority voting principle.
3. Strong: Strong negation. If the four voting results do not exceed three votes, classify it as normal.

## 7. Entirety Optimization

In the O2OS system, combining the borderline method, adaboost method, and ensemble method. Use the borderline method to modify the training sample distribution and bias the classifier towards normal. Use the adaboost method to focus the second-stage offline recognition on identifying the misclassified results from the first-stage. Use the ensemble method to integrate the classification results.

- Model Generation:

1. In `OL_Classifier_models.py`, modify the `modelpath` to specify the model storage path. Set **borderline=1** and **adaboostout = 1**. You can modify the sample quantity to be generated and which class to increase, which will bias the classifier towards that class. **Ensure that the normal training traffic in `norpath` is abundant**.

2. Run `OL_Classifier_models_adaboost.py`. Modify the five `norpath` variables and `modelpath`, where `modelpath` should be the same as the previous one. After running, go to the model path. The newly generated models are named in the format of 2_*.m, where * is a number. The order is c45, gbdt, knn, rf30.

- Preliminary Classification Results: In `OL_Classifier_ratio.py`, modify the respective paths. Set **borderline=0** and **adaboostout = 1**, and output the classification label results.

- Integration of Classification Results: In `ensemble.py`, modify the `file` to the path of `ensemble.csv`. There are three integration strategies: naive, slightly, and strong, representing different rejection granularities.

1. naïve: Applies the majority voting principle and, in the event of a tie, selects the class with the highest precision among the classifiers, considering classifier bias.
2. slightly: Classifies a sample as normal if the consensus vote of the classifiers is less than 3.;
3. strong: Considers a sample as normal in the online phase if any one of the four classifiers assigns a normal classification, while in the offline phase, the decision is made based on majority voting.

## 8. Supplements

### statistical featrues

| Category | Feature Names | Description |
|----------|---------------|-------------|
| **TCP Parameters** | B_tcpMSS, B_tcpBtm, A_tcpBtm, A_tcpTmS, A_tcpTmER, B_tcpTmER, A_tcpInitWinSz, B_tcpTmS, A_tcpRTTAckTripMin, A_tcpMaxWinSz, A_tcpAveWinSz, B_tcpAveWinSz, B_tcpPSeqCnt, B_tcpInitWinSz, A_tcpOptPktCnt, A_tcpSeqSntBytes, A_tcpMinWinSz, A_tcpMSS, A_tcpPSeqCnt, A_tcpWS, A_tcpPAckCnt, A_tcpWinSzDwnCnt | These features provide information about TCP connection parameters and states, such as Maximum Segment Size (MSS), congestion control-related window parameters (Btm, TmS, TmER), initial window size, minimum round-trip time (RTTAckTripMin), maximum window size, average window size, TCP option count, TCP sequence sent bytes, minimum window size, TCP window scaling, packet acknowledgement count, and window size decrease count. |
| **IP Parameters** | A_ipMinTTL, A_ipMaxTTL, B_ipMinTTL, B_ipMaxTTL, B_ipMaxTTL, A_ipMaxdIPID | These features describe IP protocol parameters, including minimum and maximum time-to-live (TTL) values and maximum differentiated IP identification (dIPID). |
| **Packet Length and Assembly** | A_dsMaxPl, A_dsRangePl, A_dsSkewPl, A_dsMedianPl, A_dsMeanPl, A_dsExcPl, A_dsMeanPl, A_pktAsm | These features provide information about packet length and assembly, such as maximum length, length range, skewness, median length, mean length, exceptional lengths, and packet assembly. |
| **Packet Statistics** | B_numBytesRcvd, A_tcpOptPktCnt, A_numPktsRcvd, A_numPktsSnt, B_bytAsm, A_avePktSize, A_numBytesSnt, B_tcpFlwLssAckRcvdBytes, A_tcpFlwLssAckRcvdBytes, B_numPktsRcvd, B_numBytesSnt | These features offer statistical information about packet counts and byte counts, including received byte counts, TCP option packet count, received packet count, sent packet count and byte count, TCP flow loss acknowledgement received byte count, and more. |
| **Inter-Arrival Time of Packets** | A_dsRobStdIat, B_dsIqdIat | These features describe statistical information about the inter-arrival time of packets, including the stability of the interval time (RobStdIat) and interquartile distance (IqdIat). |
| **Other Features** | A_maxPktSz, B_maxPktSz, B_dsLowQuartileIat, A_dsStdPl | These features include maximum packet size, low quartile inter-arrival time, and standard deviation of packet length. |

`A_XXX` represents the direction of the client to the server, `B_XXX` the opposite.

### model parameters

| model | parameters |
| ---- | -------- |
| C45 | criterion='entropy', splitter='best', max_depth=None |
| Cart | criterion='gini', splitter='best', max_depth=None, max_features='sqrt' |
| KNN | n_neighbors=6, weights='distance',metric=1, n_jobs=8 |
| lrc | penalty='l2' |
| RF | n_estimators=30, criterion='gini', bootstrap=True, n_jobs=8, class_weight=None, min_samples_leaf=1, max_features=None |
| GBDT | loss='deviance', learning_rate=0.01, subsample=1, n_estimators=n_estimators, criterion='friedman_mse' |
| Adaboost | n_estimators=60, learning_rate=0.1 |
| GNB | priors=None, var_smoothing=1e-09 |
| LDA | solver='eigen', priors=None |
| QDA | priors=None, reg_param=0.0, tol=1e-04 |
| SVM | kernel=kernel, probability=probability |

| parameters | CNN | LSTM |
| ---- | ---- | ---- |
| Optimizer | SGD, ADam | SGD, ADam |
| Learning rate | 0.001 | 0.0001-0.001 |
| Batch size | 64 | 64 |
| Training epochs | 32 | 32-64 |
| Number of layers | 3CNN+3MLP | 64-256 |
| Hidden layers  | 64, 128, 256, 128, 256, 64 | 64-256 |
| Activation | relu | tanh,relu |