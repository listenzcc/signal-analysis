---
layout: single
title: 频谱泄露的特点
date: 2025-1-7
author: listenzcc
categories: signalProcessing spectrum
toc: true
---

本文尝试说明频谱泄露的程度只取决于信号截断的位置，而几乎与信号的总长度无关，并且频谱泄露的程度随截断位置呈现周期性变化的趋势。

为了说明这个命题，我使用 Manim 动画分别绘制了长度约为`200`和`800`的序列被非整周期截断时的功率谱。结果表明频谱泄露现象与序列总长度无关，而仅与截断位置有关。

[toc]

## 重要结论

由于对于绝对可积函数\(f(t)\)，始终有\(f(t)(u(t) - u(t - T))\)的傅里叶变换为

\[
\tag{1} \frac{1}{2\pi}F(\omega)*\left[\left(1 - e^{-j\omega T}\right)\left(\frac{1}{j\omega}+\pi\delta(\omega)\right)\right]
\]

因此，频谱泄露的程度只取决于信号截断的位置，而几乎与信号的总长度无关，并且频谱泄露的程度随下式呈现周期性变化的趋势

\[
\omega T \in (2(n-1)\pi, 2n\pi)
\]

![spectrum_leak_pic1](/assets/spectrum_leak_pic1.png)

我使用 Manim 动画分别绘制了长度约为`200`和`800`的序列被非整周期截断时的功率谱，动画中的曲线分别是截断位置与整周期相差`0-19`个采样点时的频谱密度曲线，它们不是冲激函数，这就是频谱泄露现象。
动画中从左向右的曲线代表200长序列的频谱、从右向左的曲线代表800长序列的频谱。结果表明频谱泄露现象与序列总长度无关，而仅与截断位置有关。

<video width="320" height="240" controls>
  <source src="/assets/SpectrumLeak.mp4" type="video/mp4">
</video>

## 解决 \(f(t)(u(t)−u(t−T))\) 的傅里叶变换

本问题可根据傅里叶变换的性质，通过已知函数的傅里叶变换来求解\(f(t)(u(t) - u(t - T))\)的傅里叶变换。以下是具体步骤：

1. **回顾傅里叶变换的相关性质**
    - **乘积定理**：若\(f_1(t)\)的傅里叶变换为\(F_1(\omega)\)，\(f_2(t)\)的傅里叶变换为\(F_2(\omega)\)，那么\(f_1(t)f_2(t)\)的傅里叶变换为\(\frac{1}{2\pi}F_1(\omega)*F_2(\omega)\)，其中“$*$”表示卷积运算。
    - **时移性质**：若\(f(t)\)的傅里叶变换为\(F(\omega)\)，则\(f(t - t_0)\)的傅里叶变换为\(e^{-j\omega t_0}F(\omega)\)。

2. **求\(u(t)\)和\(u(t - T)\)的傅里叶变换**
    - 已知单位阶跃函数\(u(t)\)的傅里叶变换\(U(\omega)=\frac{1}{j\omega}+\pi\delta(\omega)\)。
    - 对于\(u(t - T)\)，根据时移性质，其傅里叶变换为\(U_T(\omega)=e^{-j\omega T}\left(\frac{1}{j\omega}+\pi\delta(\omega)\right)\)。

3. **求\(f(t)(u(t) - u(t - T))\)的傅里叶变换**

设\(f(t)\)的傅里叶变换为\(F(\omega)\)，根据乘积定理，\(f(t)(u(t) - u(t - T))\)的傅里叶变换为：

\[
\begin{align*}
&\frac{1}{2\pi}F(\omega)*\left[U(\omega)-U_T(\omega)\right]\\
=&\frac{1}{2\pi}F(\omega)*\left[\left(\frac{1}{j\omega}+\pi\delta(\omega)\right)-e^{-j\omega T}\left(\frac{1}{j\omega}+\pi\delta(\omega)\right)\right]\\
=&\frac{1}{2\pi}F(\omega)*\left[\left(1 - e^{-j\omega T}\right)\left(\frac{1}{j\omega}+\pi\delta(\omega)\right)\right]
\end{align*}
\]

即\(f(t)(u(t) - u(t - T))\)的傅里叶变换为\(\frac{1}{2\pi}F(\omega)*\left[\left(1 - e^{-j\omega T}\right)\left(\frac{1}{j\omega}+\pi\delta(\omega)\right)\right]\)。

## 阶跃函数的差的傅里叶变换

### 步骤一：回顾单位阶跃函数及其傅里叶变换

利用已知的基本信号的傅里叶变换公式以及傅里叶变换的性质来求解\(u(t) - u(t - T)\)的傅里叶变换，以下是详细的步骤：
单位阶跃函数\(u(t)\)的定义为：

\[
u(t)=
\begin{cases}
0, & t<0 \\
1, & t \geq 0
\end{cases}
\]

单位阶跃函数\(u(t)\)的傅里叶变换\(U(\omega)\)为：

\[
U(\omega)=\frac{1}{j\omega}+\pi\delta(\omega)
\]

其中\(\omega\)为角频率，\(\delta(\omega)\)是狄拉克函数，\(j\)是虚数单位（\(j^2 = -1\)）。

### 步骤二：利用时移性质求\(u(t - T)\)的傅里叶变换

根据傅里叶变换的时移性质，如果\(f(t)\)的傅里叶变换为\(F(\omega)\)，那么\(f(t - t_0)\)的傅里叶变换为\(e^{-j\omega t_0}F(\omega)\)。

对于\(u(t - T)\)，这里\(t_0 = T\)，已知\(u(t)\)的傅里叶变换为\(U(\omega)=\frac{1}{j\omega}+\pi\delta(\omega)\)，所以\(u(t - T)\)的傅里叶变换\(U_T(\omega)\)为：

\[
U_T(\omega)=e^{-j\omega T}\left(\frac{1}{j\omega}+\pi\delta(\omega)\right)
\]

### 步骤三：求\(u(t) - u(t - T)\)的傅里叶变换

设\(f(t)=u(t) - u(t - T)\)，根据傅里叶变换的线性性质，若\(f_1(t)\)的傅里叶变换为\(F_1(\omega)\)，\(f_2(t)\)的傅里叶变换为\(F_2(\omega)\)，那么\(f_1(t) - f_2(t)\)的傅里叶变换为\(F_1(\omega) - F_2(\omega)\)。

所以\(f(t)=u(t) - u(t - T)\)的傅里叶变换\(F(\omega)\)为：

\[
\begin{align*}
F(\omega)&=U(\omega)-U_T(\omega)\\
&=\left(\frac{1}{j\omega}+\pi\delta(\omega)\right)-e^{-j\omega T}\left(\frac{1}{j\omega}+\pi\delta(\omega)\right)\\
&=\left(1 - e^{-j\omega T}\right)\left(\frac{1}{j\omega}+\pi\delta(\omega)\right)
\end{align*}
\]

综上，\(u(t) - u(t - T)\)的傅里叶变换为\(\left(1 - e^{-j\omega T}\right)\left(\frac{1}{j\omega}+\pi\delta(\omega)\right)\)。 

