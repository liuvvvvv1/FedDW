# FedDW: Distilling Weights through Consistency Optimization in Heterogeneous Federated Learning

## Authors
- **Jiayu Liu** (Harbin Engineering University, China)  
- **Yong Wang** (Harbin Engineering University, China)  
- **Nianbin Wang** (Harbin Engineering University, China)  
- **Jing Yang** (Harbin Engineering University, China)  
- **Xiaohui Tao** (University of Southern Queensland, Australia)  

## Abstract
Federated Learning (FL) is an innovative distributed machine learning paradigm that enables neural network training across devices without centralizing data. While this addresses issues of data sharing and privacy, challenges arise due to data heterogeneity and the increasing scale of networks, which impact model performance and training efficiency. This paper introduces a novel framework, **FedDW**, to address these challenges through consistency optimization. The key contributions are as follows:

- Proposed the concept of Deep Learning Encrypted (DLE) data and its applications.
- Discovered the consistency relationship between soft labels derived from knowledge distillation and the classifier layer parameter matrix.
- Introduced the **FedDW** framework, demonstrating improved performance under non-IID data conditions.

Code is available at: [GitHub Repository](https://github.com/liuvvvvv1/FedDW)

---

## Methodology

### 1. Background
Federated Learning struggles with data heterogeneity (non-IID distributions), leading to significant differences in local updates and reduced global model performance. Current methods often rely on inter-client information sharing, risking privacy violations.

### 2. DLE Data and Consistency Optimization
- **Definition**: DLE data represents encrypted training data features that cannot be reconstructed back to the original data.
- **Consistency Optimization**: Regularizes the relationship between soft labels and classifier layer parameters to mitigate the impact of heterogeneity.




### 3. **FedDW** Framework
**FedDW** is a lightweight and efficient optimization framework:
1. Clients locally generate soft label matrices and upload them to the server.
2. The server aggregates global soft labels and forms a consistency regularization target.
3. Clients adjust classifier layer parameters based on the global regularization target, reducing non-IID issues.

---

## Experimental Results



---

## Citation
If you find this paper interesting, please refer to the full text: [PDF Download](Link to the paper).

