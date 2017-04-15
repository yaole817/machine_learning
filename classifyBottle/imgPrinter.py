#!/usr/bin/python
#-*-coding:utf-8 -*-

from PIL import Image
import sys
#计算指纹
def avhash(img):
    if not isinstance(img, Image.Image):
        img = Image.open(img)
    img = img.resize((8, 8), Image.ANTIALIAS).convert('L') #将image压缩为8*8,转化为灰度图
    avg = reduce(lambda x, y: x + y, img.getdata()) / 64. #对每个像素点的灰度累和,最后除以64,得到灰度的平均值

    #这一句代码很pythonic,需要仔细消化
    #map对每个像素做判断,大于平均值为1,否则为0
    #enumerate函数返回一个列表的下标及该下标对应的元素,用tuple装起来: (index, element)
    #reduce,对每个元素右移对应的下标位,并且与前一个元素做或运算,最终得到的结果为一个
    # 64位的二进制数,每一位的0,1代表该位的像素灰度与平均像素灰度的比较结果
    return reduce(lambda x, (y, z): x | (z << y), enumerate(map(lambda i: 0 if i < avg else 1, img.getdata())), 0)

#计算汉明距离
def hamming(h1, h2):
    #直接对两个数按位做异或操作,这样得到一个64位的二进制数,该二进制数包含的1的个数,即为汉明距离
    h, d = 0, h1 ^ h2
    #求d中包含的1的个数
    while d:
        h += 1
        d &= d - 1
    return h

if __name__ == '__main__':
    img1 = str(sys.argv[1])+'.jpg'
    img2 = str(sys.argv[2])+'.jpg'
    h1 = avhash(img1)
    h2 = avhash(img2)
    print u"两张图片的指纹汉明距离为:%s" % hamming(h1, h2)