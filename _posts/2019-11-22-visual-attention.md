---
title:  Visual Attention
categories:
- paper-reading
tags:
- attention
---

&emsp;&emsp;因为在做硕士课题的过程中碰到了一些时序上的多帧处理, 就采用了`Encoder-Decoder + ConvLSTM cell`来做时序上的演化. 调研了一些文献, 发现相关的工作确实比较少, 大多集中在`Image to caption`, `video object segmentation`, `semantic instance segmentation`方面, 且大部分发表在早期的2015-2016年间, 随后都转向`attention`的研究工作了. RNN存在占用显存大、超长时记忆缺陷的问题, 做NLP的早已全面抛弃RNN转向`attention`的怀抱, 这些年CV方面也慢慢地向`attention`靠拢, 以后是个趋势, 所以学习了一下`attention`到底是个什么东西, 下面做一些记录.
<!-- more -->

***
>+ # 我理解的`Attention`

&emsp;&emsp;虽然自己完全不懂NLP, 但是`attention`最初起源于NLP的应用, 所以有必要在这里提一下. 这里举一个常见的例子, 在NLP的不定长时序输入中, `I come from {China}, ... and i speak {Chinese}.`, 这里的`Chinese`其实跟前面出现过的`China`关系最大, 跟离它最近的`speak`反而关系不是最大, 但在RNN/LSTM中, 由于存在着长期遗忘的特性, 当`Chinese`和`China`之间的距离很远时, 它们二者的相互影响比较弱, 可能会被误判成`English`这类的. 所以为了解决这个问题, 需要人为地加强`Chinese`和`China`之间的联系, 让它们之间的权重比较大并占据主导地位, 即`pay more attention to "China"`, 这就是字面上`attention`的理解.  
&emsp;&emsp;在CV中也存在同样的问题, 图像中可能存在多个主体, 但我们最想关注的可能只是其中某一个主体, 这个时候就需要针对该主体做突出的强调, 让我们的网络更多地去关注(`attention`)该特定的主体. 直白地说, `attention map`就是权重图或者说是热力图. 那这跟`convolution feature map`有什么区别呢? 好像没太大区别, 本质是差不多的. `feature map`有或大或小的权值, 对应着给输入施以不同大小的权重, 但区别就在于`feature map`作用于整个`global image`, 其中存在着很多不相干的信息冗余. 而在`attention map`中, 由`feature map`中的权重经`Softmax`后产生的`attention score`介于0-1之间, 很小的`attention score`会把不相干的信息去除, 把相干的信息权重进一步提高, 相当于在`feature map`的基础上做二次强调处理. 下图便是一些直观的`attention map`.  
![](/assets/images/attention/1.png)

***
&emsp;&emsp;下面, 就根据三篇具体的文章看看`attention`在CV中具体是怎么应用的.  

>+ # Diagnose like a Radiologist: Attention Guided Convolutional Neural Network for Thorax Disease Classification

&emsp;&emsp;这是一个图像分类问题, 通过`attention`机制让模型更多地去关注某一特定区域的图像, 来判断病理图像到底属于哪一种疾病.

>> ### 网络结构及训练策略

![](/assets/images/attention/2.png)
&emsp;&emsp;三个分支: `Global branch`, `Local branch`和`Fusion branch`. 其中`Local branch`被认为是`attention`的具体实现. 模型采用分段训练的方法, 先单独训练`global branch`, 训练完成后`global branch`最末端的`feature`被认为是每一类疾病的`probability score`, 或者说就是`attention map`或`heat map`. 当然, 这里不是简单地直接采用`convolution feature map`, 而是做了一个适当转化:  
![](/assets/images/attention/3.png)
&emsp;&emsp;在每个pixel的位置上, 取所有`channel`上的最大值作为`attention map`的值, 有点像`point-wise convolution`的思想, 只不过这里只是挑出了最大值, 以此来表达最重要的联系. 得到`attention map`后就好说了, 依据一个阈值来把`attention map`二值化, 得到一个mask:  
![](/assets/images/attention/4.png)
![](/assets/images/attention/5.png)
&emsp;&emsp;再依据mask对`Global input`做crop, 即只抠出`attention map`中`attention score`最高的区域送入`Local branch`, 这些区域是影响最终分类的最重要区域. 然后再单独训练`Local branch`. 最后, 还要单独训练一次`Fusion branch`.  

>> ### 模型评价

&emsp;&emsp;比较硬核的`attention`, 直接以0/1的`attention score`来分辨重要性, 这是一种`hard attention`. 再站在更高的角度想一想, 这是不是一种类似于`ResNet`的`skip connection`?   
![](/assets/images/attention/6.png)
&emsp;&emsp;`Global branch`相当于`Residual module`, `Local branch`相当于`skip connection`. 但二者明显的区别是, `ResNet`认为`skip connection`是用来加强梯度的流动, 最后两个分支合并时用的`add`而不是`multiply`. 而这里的分支合并用的是`multiply`.  
&emsp;&emsp;再来想一想, 在图像分割领域, `FCN/U-Net/DeepLab`中不同level的`feature concatenate`是不是也是类似于这种`attention`的形式? 只不过它们都没有刻意去"突出重要的部分, 削弱冗余的部分", 只认为是一种`local texture information`和`global semantic information`的聚合.  
![](/assets/images/attention/7.png)

论文链接: [Diagnose like a Radiologist: Attention Guided Convolutional Neural Network for Thorax Disease Classification](https://arxiv.xilesou.top/pdf/1801.09927.pdf)

***
>+ # Squeeze-and-Excitation Networks (CVPR 2018)

&emsp;&emsp;前面提到的是一种`hard attention`, 与之类似的还有`soft attention`. 接下来的这篇文章通过改造`ResNet`的`residual module`, 实现了一种类似于`ResBlock`的`attention block`, 突出`the channel relationship`.  

>+ ### 网络结构

![](/assets/images/attention/8.png)
&emsp;&emsp;是不是很像`ResBlock`? 其中包含着四个操作.  
&emsp;&emsp;**$F_{tr}$:** 其实就是一个普通的`convolution`, 但是作者强调`the channel dependencies are implicitly embedded in it, but these dependencies are entangled with the spatial correlation captured by the filters.`, 意思是`channel dependencies`和`spatial correlation`纠缠在一起, 所以为了突出`the channel relationship`就有了后面的这些模块.  
&emsp;&emsp;**$F_{squeeze}$:** 名字起得花里胡哨, 其实就是一个`channel-wise global average pooling`. 并给每个`channel`起了个包装名叫`local descriptors`.  
&emsp;&emsp;**$F_{excitation}$:** 两层`FC layers`, 作用在前面$F_{squeeze}$的输出上, 相当于把`channel wise`的信息给打乱并连接到一起. 后面接一个`Sigmoid`对权重做一个二值分化.  
&emsp;&emsp;**$F_{scale}$:** 其实就是一个`channel-wise multiply`, 把每一个`channel`上学习到的不同的权重赋予`skip-connection`的输入.  
&emsp;&emsp;文中针对该模块的功能, 给出的解释是`we propose a mechanism that allows the network to perform feature recalibration, through which it can learn to use global information to selectively emphasise informative features and suppress less useful ones.` 关键词是`feature recalibration`, 这即是`attention`所在.  

>+ ### 模型思考

&emsp;&emsp;与上一篇中提到的猜想类似, `attention`确实类似于`ResNet`中的`skip connection`. 所以作者也将此模块嵌入到了`ResBlock`和`Inception block`中.  
![](/assets/images/attention/9.png)
![](/assets/images/attention/10.png)
&emsp;&emsp;仔细看一看, `attention`的机制具体体现在`Global pooling`和`Sigmoid`这两个操作上, 让大的权重更大  来占据主导地位, 让小的权重更小来削弱信息冗余. 这些权重是通过两层`FC layers`来自己学习到的, 是一种端到端的训练, 不需要像上一篇文章中那样需要分几个阶段来生成`attention map`.   

论文链接: [Squeeze-and-Excitation Networks](http://openaccess.thecvf.com/content_cvpr_2018/papers/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.pdf)

***

>+ # Stand-Alone Self-Attention in Vision Models (NIPS 2019)

&emsp;&emsp;与上一篇类似但不同, 上一篇的`squeeze-excitation block`需要依附于传统`convolution`, 本篇提出了一种新的`attention block`可以完全代替`convolution`操作.  

>+ ### 网络结构

![](/assets/images/attention/11.png)
&emsp;&emsp;同`convolution`类似, 在一个邻域内, 针对输入做线性变换得到`queries`, `keys`和`values`, 然后在这三者的基础上做矩阵操作, 得到最终的结果. 这三者的公式如下, $W_Q, W_K, W_V$即是学得的参数.  
![](/assets/images/attention/12.png)
&emsp;&emsp;还需要考虑空间信息, 因此邻域内的像素距离也被计算在内.  
![](/assets/images/attention/13.png)
&emsp;&emsp;最终的计算公式如下, 在每个pixel上都执行该操作, 权值是共享的. 但在每个channel上是独立进行的.  
![](/assets/images/attention/14.png)

>+ ### 模型思考

&emsp;&emsp;其实这篇文章不是很懂, NIPS上的很多文章都比较简短, 细节也没说清楚, 需要比较多的预备知识. 比如这里的`queries`, `keys`和`values`到底是啥, 我目前还不太了解, 这里有一个[lilianweng's blog](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html#whats-wrong-with-seq2seq-model)和[Jay Alammar's blog](http://jalammar.github.io/illustrated-transformer/)可供日后学习. 总之, 模块中也存在`skip connection`, 也有`Softmax`等二值分化操作.  

论文链接: [Stand-Alone Self-Attention in Vision Models](http://papers.nips.cc/paper/8302-stand-alone-self-attention-in-vision-models.pdf)

***
&emsp;&emsp;虽然`attention`相关知识挺深的, 但是本文的目的不在于覆盖到所有范围, 只是了解一下`attention`的思想到底是什么, 至于具体应用, 还需要以后继续学习. 很多以前不懂的文章, 现在也能够理解其中`attention mechanism`了. 比如很久以前读过的轻量级语义分割模型[旷视-EECV-2018-BiSeNet](http://openaccess.thecvf.com/content_ECCV_2018/papers/Changqian_Yu_BiSeNet_Bilateral_Segmentation_ECCV_2018_paper.pdf), 里面就用到了`attention block`, 现在就可以理解了.  
![](/assets/images/attention/15.png)


<br
/>
##### 参考资料:  
Self-Attention In Computer Vision: <https://towardsdatascience.com/self-attention-in-computer-vision-2782727021f6>  
The fall of RNN / LSTM: <https://towardsdatascience.com/the-fall-of-rnn-lstm-2d1594c74ce0>  



