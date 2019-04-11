## Lecture2：Word Vector Representations:Word2Vec

#### 1.Word meaning

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



#### 2.*Word2Vec introduction*

基本思想：**使用词义理论，来预测每个单词及其上下文单词**

两种算法：

- Skip-grams（SG）：通过给定目标来预测上下文单词（与位置无关）

  在每个估算步骤中，将一个单词作为中心词，在某种窗口大小下预测其上下文可能出现的单词，具体做法：对每个单词从1到T，预测在一个半径m范围内的附近单词出现的概率并使其最大化。

- Continuous Bag of Words（CBOW）：连续词袋模型，从上下文词袋中预测目标单词



两种中等效率的训练算法：

- Hierarchical softmax

  层次softmax，使用霍夫曼树加快训练速度

- Negative sampling

  ##### Negative Sampling主要思想：

  利用正例和一定量的负例训练一个二分类的LogisticRegressions模型.

  论文中的目标函数公式：$J(\theta)=\frac{1}{T}\sum^{T}_{t=1}J_t(\theta)$ 其中：
  $$
  J_t(\theta) = log\sigma(u_o^Tv_c) + \sum_{i=1}^{k}\Epsilon_{j~P(w)}[log\sigma(-u^T_jv_c)]\\
  其中 \sigma(x) = \frac{1}{1+e^{-x}}
  $$
  

#### 3.*Research highlight*



#### 4.*Word2Vec objective function gradients*

目标函数：最大化当前给定中心词的上下文单词的出现概率
$$
Likelihood = L(θ) = \prod^T_{t=1}\prod_{-m≦j≦m,j≠0}p(w_{t+j}|w_t;θ) （1）
$$
使用负对数似然估计,转化为最小化J(θ)：
$$
J(θ) =-\frac{1}{T}log(L(θ)) =-\frac{1}{T}\sum^T_{t=1}\sum_{-m≦j≦m,j≠0}logP(w_{t+j}|w_t;θ)  （2）\\其中\theta表示优化的参数
$$
相关细节：
$$
p(o|c) = \frac{exp(u_o^Tv_c)}{\sum^v_{w=1}exp(u_w^Tv_c)}  （3）\\其中c和o是词汇空间中的索引，即单词类型；u_o是与索引o和c的上下文单词相关联的向量\\v_c是与中心词相关联的向量
$$
(3)式中的分子使用指数函数保证点积非负，看做概率处理

其中$u_o^Tv_c$表示$o$与$c$的向量点积，$u^Tv=u.v=\sum^n_{i=1}u_iv_i$

(3)式中的分母对整个词汇表进行归一化以给出概率分布

(3)式其实就是一种softmax，将实数值$x_i$映射为一个概率分布$p_i$,即
$$
softmax(x_i) = \frac{exp(x_i)}{\sum^{n}_{j=1}exp(x_j)}=p_i
$$
我们对（3）式进行对数求偏导：
$$
\frac{∂}{∂v_c}(log\frac{exp(u_o^Tv_c)}{\sum^v_{w=1}exp(u_w^Tv_c)})=\frac{∂}{∂v_c}log\exp(u_o^Tv_c) -\frac{∂}{∂v_c} log\sum^v_{w=1}exp(u_w^Tv_c)
$$
前一项对数与指数抵消，结果就是$\frac{∂}{∂v_c}u_o^Tv_c$,求偏导结果为$u_o$

后一项使用链式求导规则进行求导得到。

**推导过程：**

当做复合函数求偏导，记$log\sum_{w=1}^{v}exp(u_o^Tv_c)=f(g(v_c))$则有：
$$
\begin{align}
原式=&\frac{1}{g(v_c)}.\frac{∂}{∂v_c}g(v_c)\\=&\frac{1}{\sum_{w=1}^vexp(u_o^Tv_c)}.\sum_{x=1}^{v}\frac{∂}{∂v_c}exp(u_x^Tv_c)\\=&\frac{1}{\sum_{w=1}^vexp(u_o^Tv_c)}.\sum_{x=1}^{v}exp(u_x^Tv_c)\frac{∂}{∂v_c}(u_x^Tv_c)\\=&\frac{1}{\sum_{w=1}^vexp(u_o^Tv_c)}.\sum_{x=1}^{v}exp(u_x^Tv_c).(u_x)\\
=&\sum_{x=1}^v\frac{exp(u_x^Tv_c)}{\sum_{x=1}^vexp(u_w^Tv_c)}.u_x \\ \\&进一步表示成\sum_{x=1}^vp(x|c).u_x
\end{align}
$$
即最终的优化结果为 
$$
\frac{∂}{∂v_c}logP(o|c)=u_o - \sum_{x=1}^vp(x|c).u_x \\
$$
其中$u_o$表示实际观察到的输出上下文单词的向量，$p(x|c)$表示对应每个单词x对应于在出现上下文c的情况下的概率值，作为期望值，乘以$u_x$，进行求和



#### 5.*Optimization: Gradient Descent*

我们有了一个loss函数$J(\theta)$，我们的目标就是最小化loss函数，而使用的方法就是梯度下降算法。

**核心思想：对当前的$\theta​$的值，计算$J(\theta)​$的梯度，然后在沿着负梯度的方向走一小步，重复此过程，即为梯度下降算法**

Gradient Descent

从矩阵角度更新等式：
$$
θ^{new} = θ^{old} - α▽_θJ_t(θ)\\
其中\alpha表示学习率
$$
从单个参数角度更新等式：
$$
\theta^{new} = \theta^{old} - \alpha\frac{\part}{\part\theta^{old}_{j}}J(θ)
$$
Q:使用上式会存在什么问题呢？

A:$J(\theta)$是对整个语料的所有窗口的loss函数，对其求梯度计算量太大

**Solution：使用随机梯度下降(Stochastic Gradient Descent):**

**每次随机采样部分窗口，使用梯度下降算法**

~~~python
while True:
    window = sample_window(corpus)
    theta_grad = evaluate_gradient(J,window,theta)
    theta = theta - alpha * theta_grad
~~~



#### 6.*Usefulness of Word2Vec*

Word2Vec最大的问题在于无法解决多义词问题，同一个单词具有多个意思时，在不同的语境中的含义其实是不同的，但是Word2Vec并不能较好地解决这一点。



#### *7. Glove*

**核心思想：使用一个共现矩阵X**

两种方式：窗口和整个文档

窗口：类似于w2v，对每个单词使用一个滑动窗口，同时获取语法信息和语义信息

单词-文档共现矩阵：

但存在很多问题：词汇越来越多，维度越来越高，后续的分类模型存在稀疏性问题，模型不够鲁棒

如何解决呢：存储包含了尽可能多的重要信息的低维向量，通常为25-1000维，那么如何降维呢：**奇异值分解**

对任意一个矩阵X，都可以分解为$U\Epsilon V^T$,其中$\Epsilon$表示对角矩阵 

**目标函数：**

$w_i\cdot{w_j}=logP(i|j)$

$J=\sum^{V}_{i,j=1}f(X_{ij})(w^T_i\tilde{w_j}+b_i+\tilde{b_j}-logX_{i,j})^2$







