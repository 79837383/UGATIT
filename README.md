## U-GAT-IT &mdash; Official TensorFlow Implementation
### : Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation

<div align="center">
  <img src="./assets/teaser.png">
</div>

### [Paper](https://arxiv.org/abs/1907.10830) | [Official Pytorch code](https://github.com/znxlwm/UGATIT-pytorch)

The results of the paper came from the **Tensorflow code**

> **U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation**<br>
> **Junho Kim (NCSOFT)**, Minjae Kim (NCSOFT), Hyeonwoo Kang (NCSOFT), Kwanghee Lee (Boeing Korea)
>
> **Abstract** *We propose a novel method for unsupervised image-to-image translation, which incorporates a new attention module and a new learnable normalization function in an end-to-end manner. The attention module guides our model to focus on more important regions distinguishing between source and target domains based on the attention map obtained by the auxiliary classifier. Unlike previous attention-based methods which cannot handle the geometric changes between domains, our model can translate both images requiring holistic changes and images requiring large shape changes. Moreover, our new AdaLIN (Adaptive Layer-Instance Normalization) function helps our attention-guided model to flexibly control the amount of change in shape and texture by learned parameters depending on datasets. Experimental results show the superiority of the proposed method compared to the existing state-of-the-art models with a fixed network architecture and hyper-parameters.*

## Usage
```
├── dataset
   └── YOUR_DATASET_NAME
       ├── trainA
           ├── xxx.jpg (name, format doesn't matter)
           ├── yyy.png
           └── ...
       ├── trainB
           ├── zzz.jpg
           ├── www.png
           └── ...
       ├── testA
           ├── aaa.jpg 
           ├── bbb.png
           └── ...
       └── testB
           ├── ccc.jpg 
           ├── ddd.png
           └── ...
```

### Train
```
> python main.py --dataset selfie2anime
```
* If the memory of gpu is **not sufficient**, set `--light` to True

### Test
```
> python main.py --dataset selfie2anime --phase test
```

## Architecture
<div align="center">
  <img src = './assets/generator.png' width = '785px' height = '500px'>
</div>

---

<div align="center">
  <img src = './assets/discriminator.png' width = '785px' height = '450px'>
</div>

## Results
### Ablation study
<div align="center">
  <img src = './assets/ablation.png' width = '438px' height = '346px'>
</div>

### User study
<div align="center">
  <img src = './assets/user_study.png' width = '738px' height = '187px'>
</div>

### Comparison
<div align="center">
  <img src = './assets/kid.png' width = '787px' height = '344px'>
</div>

## Citation
If you find this code useful for your research, please cite our paper:

```
@misc{kim2019ugatit,
    title={U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation},
    author={Junho Kim and Minjae Kim and Hyeonwoo Kang and Kwanghee Lee},
    year={2019},
    eprint={1907.10830},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## Author
[Junho Kim](http://bit.ly/jhkim_ai), Minjae Kim, Hyeonwoo Kang, Kwanghee Lee


GitHub热榜第一：小姐姐自拍，变成二次元萌妹，效果远胜CycleGAN
原创 量子位 2019-08-05 12:49:13
栗子 发自 凹非寺

量子位 报道 | 公众号 QbitAI

如何能让一个小姐姐属于你？

把她变成二次元的人类，就可以解锁一个老婆了。

韩国游戏公司NCSOFT，最近开源了一只技艺精湛的AI。

只要任意输入小姐姐的自拍，就能得到她在二次元的样子了：

GitHub热榜第一：小姐姐自拍，变成二次元萌妹，效果远胜CycleGAN
对比原图，感觉小姐姐还是那个小姐姐。

一个眼神，一个围笑，都是三次元时的样子没变。

当然，如果你有喜欢的二次元老婆，想看她穿越到现实会是什么样子，也没有问题。只要输入一张她的头像：



GitHub热榜第一：小姐姐自拍，变成二次元萌妹，效果远胜CycleGAN


就生成了逼真的小姐姐。

这个算法叫U-GAT-IT，名字也令人神往。重要的是，它是用无监督方法训练的，连成对的数据都不需要。

现在，团队已经把TensorFlow实现和PyTorch实现，都放上了GitHub。两个项目一起登上了趋势榜，且TF项目一度冲到第一。

在食用之前，不妨来看看究竟是怎样的AI，能给你这般丰盛的福利：

这只GAN的注意力，与众不同
U-GAT-IT，是一个图到图翻译算法，由两只GAN组成的。

一只GAN，要把妹子的自拍，变成二次元小姐姐。这是从源领域到目标领域的翻译。

另一只GAN，要把二次元小姐姐，再变回三次元自拍。这是从目标领域到源领域的翻译。

这样，就有两套生成器&判别器的组合。

生成器负责生成逼真的假图，欺骗判别器；而判别器负责识破假图。相生相长。



GitHub热榜第一：小姐姐自拍，变成二次元萌妹，效果远胜CycleGAN


而为了生成更加逼真的图像，团队给这两只四个部件，加入了不同的注意力。

具体的方法，受到了周博磊团队2016年CAM研究的启发。

CAM，是类激活图的简称。它能找出对于判断一张图的真假，最重要的区域，然后AI就能把注意力集中在那里。

只不过在上采样部分，CAM用的是全局平均池化。而U-GAT-IT为了更好的效果，结合了全局的平均池化和最大池化。

这里，用第一只GAN，就是生成二次元小姐姐的GAN来举例。先看判别器：



GitHub热榜第一：小姐姐自拍，变成二次元萌妹，效果远胜CycleGAN


它要判断一张图片，是不是和数据集里的二次元妹子属于一个类别。假如不是同类，那生成器的作品。

判别器有一个辅助分类器 (CAM) ，会找出对类别判断更重要的区域。

这也会引导生成器，把注意力集中在重要的区域。

再看生成器：



GitHub热榜第一：小姐姐自拍，变成二次元萌妹，效果远胜CycleGAN


它的辅助分类器，会找出属于三次元妹子的重要区域。然后，把两个领域的重要区域作比对，注意力模块就知道，生成器的注意力该集中在哪了。

第二只GAN，只是生成方向相反，道理也是一样的。

而要把两只GAN结合在一起，损失函数也是精心设计过的：

损失函数有四部分
一是对抗损失，不多解释，每只GAN都有。

二是循环损失，这是为了避免生成器和判别器找到某种平衡之后相互和解、停滞不前 (Mode Collapse) 。

要保证为目标领域生成的图像，还要能回到源领域被认可，就给生成器用了个循环一致性 (Cycle Consistency) 的约束。



GitHub热榜第一：小姐姐自拍，变成二次元萌妹，效果远胜CycleGAN


三是身份损失，为了保证输入图像和输出图像的色彩分布类似，给生成器用了个身份一致性的约束。

具体说来，从目标领域挑一张图，如果给它做一个从源领域到目标领域的翻译，应该不发生任何变化才对。

四是CAM损失，给出一张图激活图，生成器和判别器会知道它们需要提升哪里。或者说，知道两个领域之间，当前最大的区别在哪里。

除此之外，U-GAT-IT还有一个重要的贡献：

AdaLIN可选归一化方法
通常来说，Instance Normalization (IN) 是比较常用的方法，把图像的特征统计直接归一化，就能消除风格变化 (Style Variation) 。

相比之下，批量归一化 (BN) 和层归一化 (LN**) 没有那么常用。

而给图片做归一化的时候，更多见的是自适应的IN，简称AdaIN。

但在这里，团队提出了AdaLIN，它可以在IN和LN之间动态选择。



GitHub热榜第一：小姐姐自拍，变成二次元萌妹，效果远胜CycleGAN


有了它，AI就能灵活控制形状与质地的变化多大了。

从前基于注意力的模型，没办法解决不同领域之间的几何变化；

但U-GAT-IT，既可以做需要整体变化 (Holistic Changes) 的翻译，也可以做需要大幅形状变化 (Large Shape Changes. ) 的翻译。

最后再来讲一下数据集。

无监督，不成对

selfie2anime，有两个数据集。

一个是自拍数据集，一个是二次元数据集，都是只选了妹子。



GitHub热榜第一：小姐姐自拍，变成二次元萌妹，效果远胜CycleGAN


各自是训练集里3400张，测试集里100张。没有配对。

其实也不止这些，还有马变斑马，猫变狗，照片变梵高画风等等，训练了各种功能。

来看看成果如何：

效果远胜前辈
U-GAT-IT (b) 和许多厉害的前辈比了一场，它们是：

CycleGAN (c) 、UNIT (d) 、MUNIT (e) 、DRIT (f) 。



GitHub热榜第一：小姐姐自拍，变成二次元萌妹，效果远胜CycleGAN
△ 第四行，是照片变成肖像画作；第五行是变梵高画风



反向生成，比如二次元变三次，斑马变马之类，也都可以：



GitHub热榜第一：小姐姐自拍，变成二次元萌妹，效果远胜CycleGAN


(b) 列是本文主角，在穿越次元的任务上，表现明显优于各路前辈。在其他任务上，生成效果也总体超过前辈。

然后，再来看一下注意力模块 (CAM) 到底有没有作用。

右边两列，差别尽显。(e)是有注意力，(f)是没有注意力：



GitHub热榜第一：小姐姐自拍，变成二次元萌妹，效果远胜CycleGAN


最后，观察可以动态选择归一化方式的AdaLIN，比起无法选择，效果如何。

(b)是AdaLIN，右边四列是陪跑的归一化方法 (以及各种归一化的结合) ：



GitHub热榜第一：小姐姐自拍，变成二次元萌妹，效果远胜CycleGAN


AdaLIN的生成结果，更加完整，瑕疵比较少见。

如此一来，U-GAT-IT从各个角度看，都成功了。

令人内心一阵狂喜，快来领取开源代码吧。

这是TensorFlow版本，曾经登上趋势榜第一 (现在变成了第三)：

https://github.com/taki0112/UGATIT

这是PyTorch版本：

https://github.com/znxlwm/UGATIT-pytorch

这是论文：

https://arxiv.org/abs/1907.10830

— 完 —

诚挚招聘

量子位正在招募编辑/记者，工作地点在北京中关村。期待有才气、有热情的同学加入我们！相关细节，请在量子位公众号(QbitAI)对话界面，回复“招聘”两个字。

量子位 QbitAI · 头条号签约作者

վ'ᴗ' ի 追踪AI技术和产品新动态
