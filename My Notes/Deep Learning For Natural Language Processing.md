#Lecture2：Word Vector Representations:Word2Vec

1. Word meaning

   one-hot 向量表示：[0,0,0,...,1,0,0,.....] 无法表示相似关系，点积为0

   **分布式相似性**：通过查看上下文来表示单词的含义，使用向量进行分布式表示

   如何去学习一个关于词嵌入的神经网络模型：

   首先定义一个模型用于预测一个单词Wt和上下文的概率关系：
   $$
   p(context|w_t) =  ...
   $$
   再定义一个损失函数：
   $$
   J = 1 - p(w_{-t}|w_t)\\其中w_t表示t位置的word，w_{-t}表示除去t之外的其他单词
   $$
   通过在大量语料中计算不同位置t对应的概率和loss,通过改变单词的表示不断减小loss

2. *Word2Vec introduction*

   基本思想：**使用词义 理论，来预测每个单词及其上下文单词**

   两种算法：

   - Skip-grams（SG）：通过给定目标来预测上下文单词（与位置无关）

     在每个估算步骤中，将一个单词作为中心词，在某种窗口大小下预测其上下文可能出现的单词，具体做法：对每个单词从1到T，预测在一个半径m范围内的附近单词出现的概率并使其最大化。

     

   - Continuous Bag of Words（CBOW）：连续词袋模型，从上下文词袋中预测目标单词

   两种中等效率的训练算法：

   - Hierarchical softmax
   - Negative sampling

3. *Research highlight*

4. *Word2Vec objective function gradients*

   目标函数：最大化当前给定中心词的上下文单词的出现概率
   $$
   J'(θ) = \prod^T_{t=1}\prod_{-m≦j≦m,j≠0}p(w_{t+j}|w_t;θ) （1）
   $$
   使用负对数似然估计：
   $$
   J(θ) = -1/T\sum^T_{t=1}\sum_{-m≦j≦m,j≠0}logp(w_{t+j}|w_t)  （2）\\其中θ表示优化的参数
   $$
   相关细节：
   $$
   p(o|c) = \frac{exp(u_o^Tv_c)}{\sum^v_{w=1}exp(u_w^Tv_c)}  （3）\\其中c和o是词汇空间中的索引，即单词类型；u_o是与索引o和c的上下文单词相关联的向量\\v_c是与中心词相关联的向量
   $$
   我们对（3）式进行对数求偏导：
   $$
   \frac{∂}{∂v_c}(log\frac{exp(u_o^Tv_c)}{\sum^v_{w=1}exp(u_w^Tv_c)})=\frac{∂}{∂v_c}log\exp(u_o^Tv_c) - log\sum^v_{w=1}exp(u_w^Tv_c)
   $$
   前一项就是Uo,后一项使用链式求导规则进行求导 得到
   $$
   \sum_{x=1}^v\frac{exp(u_x^Tv_c)}{\sum_{x=1}^vexp(u_w^Tv_c)}.u_x \\进一步表示成\sum_{x=1}^vp(x|c).u_x
   $$
   即最终的优化结果为 
   $$
   u_o - \sum_{x=1}^vp(x|c).u_x \\其中u_o表示实际观察到的输出上下文单词的向量，p(x|c)表示对应每个单词x对应于在出现上下文c的情况下的概率值，\\作为期望值，乘以u_x，进行求和
   $$
   

5. *Optimization refresher*

   Stochastic Gradient Descent
   $$
   θ^{new} = θ^{old} - α▽_θJ_t(θ)
   $$
   

6. *Usefulness of Word2Vec*