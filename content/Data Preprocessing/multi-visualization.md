---
title: "多变量可视化"
date: 2016-08-18 21:08\\\\\\\\
---

[TOC]

### Pandas Code
[Example Code in Kraggle](../static/code/multi-visualization.ipynb)

### Andrews Curves

From [Wiki](https://en.wikipedia.org/wiki/Andrews_plot):

In data visualization, an Andrews plot or Andrews curve is a way to visualize structure in high-dimensional data. 

Each data point $x=\{x_{1},x_{2},\ldots x_{d}\}$ defines a finite Fourier series:

$$f_{x}(t)={\frac {x_{1}}{\sqrt {2}}}+x_{2}\sin(t)+x_{3}\cos(t)+x_{4}\sin(2t)+x_{5}\cos(2t)+\ldots$$

This function is then plotted for $-\pi <t<\pi$ . Thus each data point may be viewed as a line between $-\pi$ and $\pi$. This formula can be thought of as the projection of the data point onto the vector:

$$(\frac{1}{\sqrt 2},\sin(t),\cos(t),\sin(2t),\cos(2t),\ldots)$$

If there is structure in the data, it may be visible in the Andrews' curves of the data.

The result applied to Iris data:

![](../static/images/andrews-curves.png)

可以看到Andrews curves反映出Iris的数据具有两种明显不同的模式。

### Parallel coordinates plots
Parallel coordinates plots each feature on a separate column, then draws lines connecting the features for each data sample.

![](../static/images/parallel-coordinates.png)

### Radviz
Radviz is a neat non-linear multi-dimensional visualization technique that can display data on three or more attributes in a 2-dimensional projection. 

- The visualized attributes are presented as anchor points equally spaced around the perimeter of a unit circle. 
- Data instances are shown as points inside the circle, with their positions determined by a metaphor from physics: 
	+ Each point is held in place with springs that are attached at the other end to the attribute anchors. 
	+ The stiffness of each spring is proportional to the value of the corresponding attribute and the point ends up at the position where the spring forces are in equilibrium. 
- Prior to visualization, attribute values **are scaled to lie between 0 and 1**.
- Data instances that are **close to a set of feature anchors have higher values** for these features than for the others.

![](../static/images/radviz.png)

可以看到radviz graph也反映出Iris的数据具有两种明显不同的模式。

*按*：当数据维数很多的时候，是否会导致archor point过多，难以使用？

### Lag Plot && Auto Correlation Plot
对时间序列的相关性分析时，可使用这两种图形。具体例子见[这里](http://pandas.pydata.org/pandas-docs/stable/visualization.html#lag-plot)

