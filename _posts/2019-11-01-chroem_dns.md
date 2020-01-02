---
title: Chrome清除DNS缓存
categories:
- computer-skills
---

&emsp;&emsp;一直想给自己的个人网站换个简短点儿的域名. 当初在注册github账号时, 花式地想了无数个ID都被占用了, 最后无奈用了fengweiUSTC这个臃长又土到掉渣的名字, 而github pages的域名前缀又必须与ID名称一致, 所以就有了现在的域名: [fengweiustc.github.io][fengweiustc]. 后来, 想自定义一个域名: [wayne.com][wayne], 发现域名已经被注册了, 链接自动重定向到境外一家售卖加油机的网站, 索性不折腾了直接改回来. 这时候出现问题了, 在github setting中删除custom domain后, 使用其他的浏览器(iphone safari和IE)都能正常访问[fengweiustc.github.io][fengweiustc]这个站点, 但是在chrome中会依然自动跳转到[wayne.com][wayne], 在网上搜索一番, 发现是chrome的DNS缓存机制的原因, 接下来讲述一下如何清除DNS缓存恢复原站点的访问.
<!-- more -->

***
* 清除DNS cache
> 在chrome地址栏, 输入`chrome://net-internals/#dns`会出现如下图所示`Clear host cache`按键, 点击清除cache即可.

![](/assets/images/chorme-dns/chrome-dns.png)
&emsp;&emsp;然而, 问题依然没有得到解决, 接下来尝试其它方案.

* 清除浏览数据
> 在`设置 -> 高级 -> 隐私设置和安全性 -> 清除浏览数据`中, 会有如下选项:

![](/assets/images/chorme-dns/chrome-cache.png)
&emsp;&emsp;选中`缓存的图片和文件`, 清除即可, 这下子问题得到了解决, 之前错误地以为需要清除cookie, 看来并没有搞清楚其中的关系. Anyway, 能解决问题就行.

***
&emsp;&emsp;想把这个网站长期维护下去, 目前感觉还需要添加的一些模块: 评论功能, 搜索功能, 站点统计功能, 一点点地来吧.

[fengweiustc]: https://fengweiustc.github.io/
[wayne]: https://wayne.com/

[comment]: comment
<!--
comment
-->