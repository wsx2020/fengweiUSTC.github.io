---
title:  Paper review - Faster RCNN/YOLO/SSD & Paper Reading - ICCV 2019 "ThunderNet - Towards Real-time Generic Object Detection"
categories:
- paper-reading
tags:
- detection
- RPN
---

&emsp;&emsp;旷视detection组的一篇轻量级two-stage目标检测论文, 起的名字很好听, ThunderNet, 所以就特意找出来看一看. 以前接触detection比较少, 就趁这个机会把一些经典的object detection论文找出来读一读, 主要有two-stage的Faster-RCNN和ont-stage的YOLO、SDD, 它们奠定了一些基本的思路和框架, 新发表的论文基本是在此基础上做延伸, 有需要的时候再细看, 下面是一些总结. (全文4000多字, 涉及4篇paper, 27张图, 加载起来可能比较慢.)

***
>+ # 问题概述

&emsp;&emsp;通用目标检测, 任务是检测出图片中存在物体的边界框坐标, 且需给出对应物体的分类. 直观地想象, 针对这样一个任务, 如果采用end-to-end的形式训练, 需要有一个网络, 它的输入是图片, 输出是可能存在物体的位置以及类别, 这是one-stage模型, 有无数个可能的位置以及很多种类别的可能, 难度是比较大的. 如果采用two-stage模型, 先让一个网络(region proposal network/RPN)负责找出可能存在物体的区域, 它只关心该区域是否存在物体, 以及该区域的大致坐标位置, 而不关系物体的分类具体是什么. 接下来把这些候选区域送进下一个分类网络(detector), 这个网络只需要关心候选框内的物体分类是什么即可, 具有较强烈的"候选区域内一定存在物体"的先验条件. 两部分网络各司其职, 就能有较好的效果, 但是也存在流程繁琐的缺点.
<center>具有代表性的经典paper</center>

![](/assets/images/detection/2.png)
<center>沿着时间线上的发展</center>

![](/assets/images/detection/1.png)
&emsp;&emsp;下面分别针对two-stage的经典模型: Faster-RCNN和one-stage的代表模型: YOLO/SDD, 记录一些基本的常识性内容.

***
>+ # Faster-RCNN (NIPS 2015)

&emsp;&emsp;MSRA出品, 作者列表中有孙剑、何凯明、任少卿等大佬, 这三位也是ResNet的部分作者. 其中任少卿是科大信院与MARS连培的博士, 我作为菜鸡真的是感叹人与人的差距比人与狗的差距还要大.
>> ### 网络结构  

![](/assets/images/detection/3.png)
&emsp;&emsp;主要有四部分: Backbone, RPN, ROI Pooling, Classifier.    
&emsp;&emsp;**Backbone:** 把固定大小为1000x600的input, 降采样为原来的1/16(用了4个2x2的pooling), 输出维度为512x60x40.  
&emsp;&emsp;**RPN:**   
&emsp;&emsp;-- 对每一个pixel提出9个候选框anchor, 这9种anchor如下图所示(长宽比0.5、1、2, 缩放比8、16、32), 一共有60x40x9=21600个anchor. 
![](/assets/images/detection/4.png)
&emsp;&emsp;- 有两个分支, 一路做regression, 输出anchor的四个坐标(中心点x、y, 宽高w、h), 维度是36x60x40, 36即是9种anchor的4个坐标点(其实不是绝对坐标, 而是坐标的偏差, 类似于残差的思想, 在后面loss一节会再做详细介绍); 另一路做classification, 输出该anchor是否存在物体(0或1), 维度是18x60x40, 18即是9种anchor被softmax分成0或1的两类.
&emsp;&emsp;-- 这些anchor数量太多了, 需要进行剔除. 首先去掉超出feature边界的框, 然后标记与Ground Truth之间IoU超过0.7的框即为正例保留, 标记与Ground Truth之间IoU小于0.3的框为反例保留, 其余框剔除. 然后在剩下的框中, 正例随机取128个, 反例随即取128个进行训练. (不同的是, 在inference时, 如果有anchor越界了, 不会进行剔除, 会对其进行clip, 限定在图像区域内.)  
&emsp;&emsp;-- RPN最后还有一个Proposal层, 细节也比较多. proposal接受经过残差修正后的新anchor(如何修正见下节loss介绍), 再次进行剔除(越界剔除、宽高过小的anchor剔除), 接下来进行非极大值抑制, 旨在去掉重复区域较多的anchors提高效率. 然后按照前面的输出softmax score从大到小对anchors进行排序, 选择前面topN的anchors(即只保留正例anchors). 至此, anchor就是比较干净和较准确的框了, 可以送入下一步进行分类识别.  
>>> -- 非极大值抑制(Non-Maximun Suppression/NMS): 
![](/assets/images/detection/5.png)
a. 对所有anchor的得分从大到小排序, 如上图[0.98, 0.83, 0.81, 0.75, 0.67].  
b. 以当前最大值0.98为参考, 依次判断其它anchor是否与参考框的IoU超过某一阈值(文中取的是0.7), 剔除重复面积过大的框, 如删除[0.83, 0.75], 保留最大值0.98并以后不参与筛选.  
c. 找出当前最大值0.81, 重复步骤b, 直到没有框可供筛选.
![](/assets/images/detection/6.png)
&emsp;&emsp;如图最后只留下了每一轮的最大值0.98和0.81.  

&emsp;&emsp;**ROI Pooling:** 输入是由RPN筛选出的大约300个anchors坐标(300x4), 以及backbone生成的维度为512x60x40的feature. 前面过程中幸存下来的这些anchors, 宽高比可能不一致, 需要把它们映射成固定大小的feature, 方便后面带有全连接的分类器进行分类. 方法比较粗暴, 即把anchor在feature上所对应的区域, 分割成7x7的块儿, 每个块内取最大值, 即得到300x512x7x7的feature.
![](/assets/images/detection/7.png)
>>> -- RoI Pooling可能存在的问题:  
&emsp;&emsp; 一, anchor映射回feature会存在量化误差, 如50/3=16.66, 实际中只能取16; 二, 映射后的feature的候选区域宽度不是7的整数倍时, pooling也会带来量化误差, 对结果产生影响, 如第一步中的16, 16/7=2.28, 实际中只能取2.   
&emsp;&emsp;因此这里有替代方案--RoIAlign: 一, 保留映射时的浮点数; 二, pooling时, 对虚拟中心点采用双线性插值, 再进行pooling.
![](/assets/images/detection/8.png)

&emsp;&emsp;**Classifier:** 输入是300x512x7x7的anchor feature, 输出有两路: 一路是每一个anchor的类别概率, 维度是300x81(有81个类); 另一路是每个anchor的坐标偏移修正法量, 维度是300x(81x4), 对框的位置进行精细修正. 这两个输出即是我们最终想要的"物体类别"和"物体位置".

>> ### 损失函数

&emsp;&emsp;RPN和Classifier是分开训练的, 因为目的不一样. RPN目的是找出存在物体的框, 只关心是否存在物体以及框的粗修正, 不关心框内是什么物体; Classifier只认为送给它的输入框内一定存在物体, 有一个很强的先验假设, 然后它在此基础上放心地去预测这个物理到底属于哪一类, 同时进行精细框修正.  
&emsp;&emsp;**RPN loss:**   
&emsp;&emsp;-- 一个是对物体是否存在的分类loss, 原文中采用的是log loss, 其实就是cross entropy loss; 一个是对边界框偏移量进行修正的regression loss, 原文中采用的是smooth L1 loss, 俗称huber loss. 由于两部分loss的数量级差了将近10倍, 所以取$\lambda=10$进行量级平衡.
![](/assets/images/detection/9.png)
&emsp;&emsp;-- 再来看, anchor的四个坐标表示到底是什么, 摘抄原文中的一段如下, 即用$(t_x, t_y, t_w, t_h)$表示偏移量, 它们越小表示box重叠地越好. 回归收敛后, 可使用$(t_x, t_y, t_w, t_h)$转换得到真正的$(x_a, y_a, w_a, h_a)$.
![](/assets/images/detection/10.png)
&emsp;&emsp;**Classfier loss:** 基本同上, 一个分类loss, 对应着不同的类别(不同的是, RPN分类只有0和1两类, 而这里有81类); 一个是回归loss, 对应着精细框位置修正.  
&emsp;&emsp;**训练过程:** 一, 导入ImageNet预训练模型, 单独训练RPN; 二, 单独训练Classifier, 它的输入是上一步训练好的RPN提供的proposal; 三, 再次单独训练RPN, 这次会先把Classifier的部分Conv层权重导入到RPN并冻结, 这里就是共享权重的地方, 节省了计算时间; 四, 单独训练Classifier, 这里在上一步中共享给了RPN的权重会被冻结, 输入是上一步训练好的RPN提供的proposal. 本质就是一种迭代循环, Classifier不断地共享它的部分权重给RPN, 二者轮流fine tune, 只不过文中只循环了两次, 作者说两次循环就够了, 再往下循环下去对性能也没有太大提升.  
<br
/>
&emsp;&emsp;至此two-stage的Faster RCNN便介绍完了, 可以看到流程比较复杂, 而且NIPS原文有一些细节也没讲清楚, 这里参考了一些资料.  
&emsp;&emsp;Faster RCNN原文: [Faster RCNN](http://papers.nips.cc/paper/5638-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks.pdf)  
&emsp;&emsp;知乎专栏: [一文读懂Faster RCNN](https://zhuanlan.zhihu.com/p/31426458)  
&emsp;&emsp;一篇博客: [Faster RCNN 学习笔记](https://www.cnblogs.com/wangyong/p/8513563.html)

***
>+ # YOLO (CVPR 2016)

&emsp;&emsp;比Faster RCNN稍微晚了一点, 作者之一是Ross Girshick, 这个人同时也是Faster RCNN、Fast RCNN和RCNN的作者, 发表RCNN时在UC Berkeley, 发表Fast RCNN和Faster RCNN时去了微软, 发表现在这篇YOLO时是在Facebook AI, 简直给跪了.
>> ### 网络结构 

![](/assets/images/detection/11.png)
&emsp;&emsp;网络结构平平无奇, 就是个类似于VGG一卷到底的结构. 主打轻量级end-to-end的模式, 没有复杂的RPN候选框机制, 文章开头就先把two-stage方法的复杂且低效吐槽了一遍, 强调该文的方法只需`you only look once`(YOLO), 有快速、低背景错误率和实际场景泛化能力强的优点.    
&emsp;&emsp;下面来看该方法的特殊思路. 如果直接让模型既输出物体边界框, 又输出物体类别, 这是一个无法收敛的任务, 无法正确指明优化的方向, 所以得做一些特殊的处理.
![](/assets/images/detection/12.png)
&emsp;&emsp;首先, 强行把输入图片划分成SxS的网格(文中取7x7). 接下来有一个分支, 负责预测每个网格属于哪个类别(对应上图class probability map), 这里是不需要预测位置信息的, 输出维度是SxSxC, C是类别的个数; 另外一个分支, 在每个网格上提出B个anchor(文中取B=2), 预测的是这些anchor区域是不是有一个物体的可能性, 只考虑是不是物体, 不考虑物体的类别是什么, 这与RPN只负责检测是否存在物体有点类似. 另外, 还预测这B个anchor的4个坐标点的位置, 所以输出维度是SxSx(B$\ast$5). 最后这两个分支合起来, 输出维度就是SxSx(B$\ast$5+C). 虽然是这么分析, 但是具体到网络中, 直接把最后一维4096的向量reshape成SxSx(B$\ast$5+C), 重点放在了loss上.

>> ### 损失函数

&emsp;&emsp;由上面的分析知, `class probability`那一路负责预测类别, 它有每个网格内必定存在物体的先验, 所以预测的是$P(Class_i\mid Object)$; 另一路`bounding boxes + confidence`预测是否存在物体和位置, 可表示为$P(Object)\times IoU_{pred}^{truth}$. 最后将二者合并, 预测的就是我们的目的: 类别和位置.
![](/assets/images/detection/13.png)
&emsp;&emsp;既然是两个任务, 损失函数也要兼顾两部分任务. 一, 对于四个位置点(x, y, w, h)的预测, 采用了MSE loss, 特殊地, 大框位置偏一点对总体IoU影响不大, 但是小框位置偏一点对最后IoU 影响很大, 所以要提高小框影响的权重, 也是就是降低大框的权重, 所以loss里用的是$\sqrt w, \sqrt h$. 二, 对于anchor confidence的预测, 它们的重要性不如位置点的预测,位置点先预测准了, confidence自然就上去了, 所以做权重调整, 文中给的是$\lambda_{coord}=5, \lambda_{haveObject}=1, \lambda_{noObject}=0.5$. 三, 对于网格类别的预测, 文中没有采用one-hot编码, 而是每一类都预测一个浮点型概率, 最后采用MSE loss. 如下图, 第一二行对应位置回归, 三四行对应是否存在物体的confidence回归, 第五行对应网格点物体类别回归.
![](/assets/images/detection/14.png)
&emsp;&emsp;最后, 对所有修正后的anchor进行非最大值抑制, 这个在Faster RCNN中也有介绍, 不再赘述.

>> ### 模型效果

&emsp;&emsp;虽然效果不是最佳的, 但是运行速度FPS显著高于其它, 达到了效果和性能的综合最佳. 同时, 与Fast RCNN相比, 虽然它的位置出错率占比更高, 但是它的背景出错率占比更低, 也就是它的False Positive更低, 不容易把背景认成Object. 
![](/assets/images/detection/15.png)
![](/assets/images/detection/16.png)
<br
/>
&emsp;&emsp;YOLO原文: [YOLO](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf)

***
>+ # SDD (ECCV 2016)

&emsp;&emsp;稍晚于YOLO, 因此也借鉴了一些YOLO one-stage的思想, 感觉是出于技术的后发优势, 达到了性能优于Faster RCNN且速度优于YOLO的效果. 文章做的工作很好, 但是不得不吐槽一下文章的写作真是一言难尽, 很多细节没有表述清楚, 还有很多冗余的废话, 用了很多长句. 看了一下第一作者, 是个中国学生写的, 也就不觉得奇怪了.
>> 示例一: `we evaluate a small set of default boxes of different aspect ratios at each location in several feature maps with different scales.`  
示例二: `By combining predictions for all default boxes with different scales and aspect ratios from all locations of many feature maps,`

&emsp;&emsp;不禁让我想起自己的论文, 其实也犯了很多这样的错误, 滥用长句, 一个句子里"of、for、in、with、from、by"这些介词全用上了, 其实完全拆开写成短句更清晰. 还有就是用了很多主动语态, 公式里的符号没有紧接着完全点明含义. 在英文写作这块, 很多非母语作者也表述得很地道, 读起来完全没有理解障碍, 写的时候应该也是花了一番心思的, 所以以后看文章要多留心native的表达方式.

>> ### 网络结构

![](/assets/images/detection/17.png)
&emsp;&emsp;跟YOLO相比, 无非就是在不同level的feature上都提出anchor, 以此来捕捉不同尺度大小的物体. 具体的来说, 浅层感受域小的feature用来检测小目标, 深层感受域大的feature用来检测大目标. 同时, 不同level的feature由于大小不同, anchor的大小也根据feature的大小来相应成比例变化.
![](/assets/images/detection/22.png)
![](/assets/images/detection/18.png)

>> ### 损失函数

&emsp;&emsp;跟YOLO有点类似, 分为类别预测loss和位置回归loss, 其中类别预测loss用的是softmax loss, 位置回归loss用的是smooth L1 loss.
![](/assets/images/detection/19.png)
![](/assets/images/detection/20.png)
![](/assets/images/detection/21.png)

>> ### 训练策略

&emsp;&emsp;-- Matching strategy: feature上的每个pixel都有好几个anchor, 所以anchor总数很多, 需要进行筛选, 与Ground Truth的IoU超过阈值0.5, 才会认为是正例.  
&emsp;&emsp;-- Hard negative mining: 反例往往多于正例, 这会导致在训练时loss被反例所支配, 所以要进行比例调整, 按照confidence score排序, 挑选前面的部分正例, 使得正例:反例=1:3.  
&emsp;&emsp;-- Data augmentation: 三种方式, 一种是输入原图, 另一种是在原图上随机取patch, 最后一种是在Ground Truth周围随机取patch(IoU=0.1, 0.3, 0.5, 0.7, 0.9), 这种patch在后面利用率很高. 后面还有0.5概率随机flip.

>> ### 模型效果及评价

![](/assets/images/detection/23.png)
&emsp;&emsp;mAP比Faster RCNN要好, FPS还比YOLO要高. 总体感觉, 整体思想其实跟YOLO没什么本质区别, 只是增加了更多的数据增广手段, 如多level的feature、随机取patch, 提升了不同尺度下物体的效果. YOLO中对anchor未做长宽比限制, 也就是说每个anchor的形状都是随机无假设的, 而SSD做了不同宽高比的anchor假设, 这里就存在一个先验假设, 假设了物体的不同尺寸, 这个假设对结果有一定的提升. 说白了, 就是trick比较多, 对结果提升有好处, 坏处是很多超参(如宽高比、feature的选择)的调整是手动的, 依据经验的.  
<br
/>
&emsp;&emsp;SSD原文: [SSD](https://arxiv.xilesou.top/pdf/1512.02325.pdf)

***
>+ # ThunderNet (ICCV 2019)

&emsp;&emsp;旷视做了很多模型轻量化的探索, 比如著名的shuffleNet系列, 主要是针对将来部署在端上(手机或摄像头)的实时模型需求. 这篇ThunderNet就是在这种业务背景下催生出来的, 虽然是two-stage, 但是能够在ARM平台上real-time运行.
![](/assets/images/detection/24.png)

>> ### 网络结构

&emsp;&emsp;**Backbone Part:** 320x320固定大小的输入. 传统detection模型的backbone都是从一个classifier网络(例如VGG)迁移而来, 但文中指出`classification`和`detection`任务的性质不同, 具体来说, `classification`需要大的感受域和high level的feature来进行分类, 不关心local的纹理细节, 而`detection`需要low-level的纹理细节(或者说空间信息)来进行定位,  同时又要求大的感受域使得看到物体的范围更大. 因此`detection`所用的网络应该是又浅又大的(层数少且感受域大), 文中的backbone采用了ShuffleNet V2, 但是为了增加low-level的重要性, 将其中3x3的Conv换成了5x5的Conv, 还在low-level feature上增加了更多的channel数. 以上便是被称为SNet的Backbone.  
&emsp;&emsp;**Detection Part:**   
&emsp;&emsp;-- Context Enhancement Module: backbone的输出感受域还是不够大, 在这里, 为了进一步扩大感受域, 采用了金字塔型的多级feature融合方法, 即在backbone的末级用1x1的Conv和上采样得到local context information和global context information.  
![](/assets/images/detection/25.png)
&emsp;&emsp;-- RPN & subnet: RPN与Faster RCNN中的结构差不多, subnet仅仅是一个卷积层和一个全连接层.   
&emsp;&emsp;-- Spatial Attention Module: 这个模块是一个与RPN并行的分支旁路, 它的目的是利用RPN所学得的前景和背景的区别, 增强网络对前景的关注, 降低对背景的关注度. 同时, 它也是一种short cut, 能增强梯度的流动.  
![](/assets/images/detection/26.png)

>> ### 损失函数

&emsp;&emsp;文中并未提及loss函数, 猜测和同样是two-stage的Faster RCNN的loss差不多.

>> ### 模型效果

![](/assets/images/detection/27.png)

>> ### 模型评价

&emsp;&emsp;因为是旷视系的文章, 所以文章倾向于使用或引用孙剑老师门下的文章结构(如GCN、shuffleNet、Lighthead r-cnn等). 同时, 依靠一些对网络模型的理解和经验, 压缩了一些channel数减小计算压力, 增大了一些kernel大小增大了感受域, 设计了一些模块提取local和global的空间信息, 并调整了特征图的注意力分布, 在容量上考虑了backbone和detector的平衡, 各种考量综合起来达到了想要的目的. 虽然可能在模型创新上不是break through, 但是具有很强的实用价值, 可以实时运行在ARM上. 所以要多看文章, 理解神经网络模型的意义, 各个feature对应着什么, 各种任务需要的是什么, 才能合理"搭积木".  

<br
/>
&emsp;&emsp;ThunderNet原文: [ThunderNet](https://arxiv.xilesou.top/pdf/1903.11752.pdf)

