---
title: "模型评估和选择"
date: 2016-08-18 21:08
---

[TOC]

### 基本形式
$$f(\boldsymbol x)=\boldsymbol\omega^T\boldsymbol x + b$$
其中$\boldsymbol\omega=(\omega_1;\omega_1;...;\omega_d)$
### 线性回归
数据集D,样本由d个属性描述.此时我们试图学得$$f(\boldsymbol x_i)=\boldsymbol\omega^T\boldsymbol x_i + b$$
使得$f(\boldsymbol x_i) \simeq y_i$。

为便于讨论,我们把$\boldsymbol\omega$和$b$吸收入向量形式$\boldsymbol{\hat\omega}=(\boldsymbol\omega;b)$, 相应的, 把数据集D表示为一个$m \times (d + 1)$大小的矩阵$\boldsymbol X$ ,其中每行对应于一个示例,该行前d个元素对应于示例的d个属性值,最后一个元素恒置为1,即
$$\boldsymbol X=
\begin{pmatrix}
x_{11}& x_{12}& \cdots& x_{1d}& 1\\
x_{21}& x_{22}& \cdots& x_{2d}& 1\\
\vdots& \vdots& \ddots& \vdots& \vdots\\
x_{m1}& x_{m2}& \cdots& x_{md}& 1
\end{pmatrix}=
\begin{pmatrix}
\boldsymbol x_{1}^T& 1\\
\boldsymbol x_{2}^T& 1\\
\vdots& \vdots\\
\boldsymbol x_{m}^T& 1
\end{pmatrix}
$$
再把标记也写成向量形式$y=(y_1;y_2;\cdots;y_m)$,则有
$$\boldsymbol{{\hat\omega}^*}=\mathop{\arg\min}\limits_{w}(\boldsymbol y-\boldsymbol X{\hat\omega})^T(\boldsymbol y-\boldsymbol X{\hat\omega})$$
令$E_{\boldsymbol{\hat\omega}}=(\boldsymbol y-\boldsymbol X{\hat\omega})^T(\boldsymbol y-\boldsymbol X{\hat\omega})$，对$\boldsymbol{\hat\omega}$求导得到
$$\frac{\partial E_{\boldsymbol{\hat\omega}}}{\partial \boldsymbol{\hat\omega}}=2{\boldsymbol X}^T(\boldsymbol X{\hat\omega}-\boldsymbol y)$$
令上式为零可得$\boldsymbol{\hat\omega}$最优解的闭式解:

- 若$\boldsymbol X^T\boldsymbol X$为满秩矩阵或正定矩阵，有
$$\boldsymbol{\hat\omega}=(\boldsymbol X^T\boldsymbol X)^{-1}\boldsymbol X^T\boldsymbol y$$
- 若$\boldsymbol X^T\boldsymbol X$不满秩(如$p >> n$)，可解出多个$\boldsymbol{\hat\omega}$，都能使均方误差最小化。择哪一个解作为输出, 将由学习算法的归纳偏好决定,常见的做法是引入正则化(regularization)项。

线性回归的变化：可考虑对$y$的函数做回归，如$lny$。一般地,考虑单调可微函数$g(\centerdot)$, 令$$y= g^{-1}({\boldsymbol{\hat\omega}}^T\boldsymbol x+b)$$这样得到的模型称为“广义线性模型”(generalized linear model)。

###对数几率回归
回归任务 => 分类任务
分类任务的输出标记为$y\in\{0, 1\}$, 而线性回归的预测值$z={\boldsymbol{\omega}}^T\boldsymbol x+b$为实数，因此我们需将实值$z$转换为0/1值.最理想的是“单位 阶跃函数”(unit-step function)
$$abc=
\begin{equation}
\begin{cases}
0,& z<0\\
0.5,& z=0\\
1,& z>0
\end{cases}
\end{equation}
$$即若预测值$z$大于零就判为正例,小于零则判为反例,预测值为临界值零则可任意判别。
