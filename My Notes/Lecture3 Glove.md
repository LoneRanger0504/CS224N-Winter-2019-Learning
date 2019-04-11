#Lecture3 Glove：Global Vectors for Word Representation

##Objection function of Glove:
$$
J(θ)=\frac{1}{2}\sum^W_{i,j=1}f(P_{ij})(u^T_iv_j-logP_{ij})^2
$$
优点：

- 训练速度快
- 可扩展至庞大的语料库
- 小规模语料/向量下表现良好

