---
title:  Matplotlib -- SCI论文绘图配置
categories:
- computer-skills
---

&emsp;&emsp;前一阵子忙着修改论文, 最近完成了初稿, 为了后期能复现一些结果, 就把文献、代码、数据以及后处理等过程都做了存档. 其中, 所有的的示意图都是用PPT制作的, 所有的数据图都是用`Matplotlib`画的, 二者自由度都很高, 这也意味着有些地方需要人为精细地控制, 下面针对写`Paper`的大概流程配置和`Matplotlib`制作科研论文用图的设置做一些记录.

***
>+ ### 科研Pipeline

1. 文献阅读  
&emsp;&emsp;`NoteExpress`, 之间用过一段时间的`EndNote`, 体验很不好, 主要是因为对中文的支持不好, 而且加载起来比较慢. 后来用了国产的`NoteExpress`, 可以随文献记录一些中文笔记, 文献也可以按中文文件夹名分类存放, 用起来比较顺手.  
2. 写Code  
&emsp;&emsp;`VSCode`, 这个无可替代没得说. 炼丹的话, 用`vim`来修改任务脚本, `tmux`开多个`panel`下多个丹. 一般不喜欢在`vim`上直接写代码, 主要不喜欢键盘操作, 复制粘贴移动光标也不如鼠标方便. 习惯是在本机上用`VSCode`写好代码, 再在`Xshell+Xftp`(Window)里上传到服务器, 写一个脚本来跑任务. 同时只在本机上维护代码的`git repo`, 因为服务器上是无法连接外网的, 所以这里就需要十分注意`remote`和`host`端的代码同步, 所有的修改都只在`host`端完成, 确保不分叉. (`VSCode`虽然也有`Remote-SSH`插件, 可以做到`remote`和`host`端代码自动同步, 但亲测不是很好用.)  
3. 数据后处理  
&emsp;&emsp;`Jupyter Notebook`, 把数据存成`.h5`或`.npy`文件, 在`Jupyter`中加载, 使用`%matplotlib inline`在浏览器页面上画图, 然后使用`plt.savefig("name.pdf/.eps")`或`plt.imsave("name.pdf/.eps", file)`来保存矢量格式的图片. 注意, 如果使用`xeLaTex`编译的话需要的是`.eps`格式, 使用`pdfLaTex`编译的话需要的是`.pdf`格式. `.pdf`格式的图片在编译时候速度很快, 人生苦短, 所以建议用`pdfLaTex`. (`VSCode`虽然也原生支持了`Jupyter`, 但亲测不好用, 有很多`bug`, 而且图形化的东西还是在浏览器界面上看着舒服.)   
&emsp;&emsp;针对由`PPT`制作的示意图, 需要将整个`.pptx`保存成`.pdf`文件才能保留矢量信息, 如果想要去除大面积的边框空白或者想转化成`.eps`格式, 需要用`Adobe Acrobat DC`转化一下(尊重版权, 我爱zheng版).  
4. 论文写作  
&emsp;&emsp;`VSCode+LaTex插件`, 针对这个组合之前专门写过一篇[配置教程](https://fengweiustc.github.io/computer-skills/2019/11/17/vscode/). `WinEdt/TeXLive`之类的编辑器, 写着没有`VSCode`有感觉.  

&emsp;&emsp;总的来说, 就是尽量摆脱对特定软件的依赖, 能在`VSCode`上插件化的尽量插件化, 能用代码库的尽量调库, 能统一数据格式的尽量统一格式, 能统一编程语言的尽量统一成`Python`.


>+ ### `Matplotlib`常用设置

1. 为了与正文字体保持一致, 图和公式中任意地方出现的字体应该使用`Times New Roman`. 曾经在这里还踩了一个`bug`, 切换字体后变成了`Italic`的`Times New Roman`, 而不是想要的朴素版`Times New Roman`. 使用命令`matplotlib.get_cachedir()`查到所有字体库的路径是`'C:\\Users\\fengwei\\.matplotlib'\\fontList.json`, 发现一个键`Times New Roman`居然同时对应着四种格式的字体(`style`和`weight`不同), 强行修改键值为一对一, 这才正常了.  
```python
    # 修改图中的默认字体
    import matplotlib.pyplot as plt
    plt.rc('font',family='Times New Roman') 
    # 修改公式中默认字体
    from matplotlib import rcParams
    rcParams['mathtext.default'] = 'regular'
```

2. 修改图片长宽大小及比例, 修改子图的行列数, 修改图片内容距边缘的相对位置, 修改子图的间距.  
```python
    plt.figure(figsize=(7,5))
    plt.subplots(nrows=3, ncols=3, figsize=(6, 6))
    plt.subplots_adjust(left=0.2, right=0.85, top=0.9, bottom=0.15, wspace=0.01, hspace=0.1)
```

3. 修改坐标轴字体大小, 修改坐标轴显示间隔.  
```python
    # 经测试, 发现坐标轴刻度字体大小采用16, Label字体大小采用22, legend大小采用`x-large`, 线宽采用2比较合适, 即使在文章排版后经过缩放也能保证看得清.
    plt.xticks(fontsize=16); plt.yticks(fontsize=16); plt.tick_params(labelsize=16)
    # 修改X轴刻度显示间隔为0.5的倍数(例如0.5, 1.0, 1.5, 2.0, ...)
    from matplotlib.pyplot import MultipleLocator
    plt.gca().xaxis.set_major_locator(MultipleLocator(0.5))
```

4. 修改图例的大小以及位置, 是否需要边框, 甚至精细调整相对位置.  
```python
    plt.legend(fontsize='x-large', loc=0, frameon=False, bbox_to_anchor=(0.575, 0.38))
```

5. 修改颜色柱(`Colorbar`)四个角的位置以及颜色.  
```python
    fig, axes = plt.figure()
    image = axes[1][2].imshow(file)
    cbar_ax = fig.add_axes([0.93, 0.66, 0.015, 0.2])
    fig.colorbar(image, cax=cbar_ax, cmap='Greys_r')
```

&emsp;&emsp;下面附上一张速查表, 右键新页面打开全图可以缩放查看:
![](/assets/images/matplotlib/1.png)
