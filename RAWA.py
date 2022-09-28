from PIL import Image
import os
import numpy as np
import random
from skimage.metrics import mean_squared_error,normalized_root_mse,peak_signal_noise_ratio
import time

def alphaBlending(image,watermark,xs):
    """
    image:原图像
    watermark:水印
    xs代表[x,y:插入的像素位置，
            alpha:透明度，
            scale:缩放系数>>>>>beta:缩放比例，
            rotation:旋转角度]
    测试：alphaBlending("./project/3.png","./project/Berkely.png",[100,100,0.3,0.3,45])
    """
    # 原图像读取
    rgba_image = Image.open(image)
    rgba_image = rgba_image.convert("RGBA") #转换成4通道，A代表透明度通道
    image_x, image_y = rgba_image.size
    #print((np.array(rgba_image).T[3]==255).all())

    # 水印读取
    rgba_watermark = Image.open(watermark)
    rgba_watermark = rgba_watermark.convert('RGBA')

    # 旋转水印
    rgba_r_watermark = rgba_watermark.rotate(xs[4], expand=1)

    # 缩放水印
    watermark_x, watermark_y = rgba_r_watermark.size
    watermark_scale = min(image_x * xs[3] / watermark_x, image_y * xs[3] / watermark_y)
    new_size = (int(watermark_x * watermark_scale), int(watermark_y * watermark_scale))
    #rgba_rs_watermark = rgba_r_watermark.resize(new_size, Image.ANTIALIAS)
    rgba_rs_watermark = rgba_r_watermark.resize(new_size, resample=Image.Resampling.LANCZOS)

    # 水印插入
    x1,y1 = rgba_rs_watermark.size
    scr_channels = np.array(rgba_rs_watermark).T[0:3]
    dstt_channels = np.array(rgba_image).T[0:3]
    a = np.array(rgba_rs_watermark).T[3]

    # 限制大小
    x = np.clip(xs[0], 0, image_x - x1)
    y = np.clip(xs[1], 0, image_y - y1)
    alpha=xs[2]

    for i in range(3):
        dstt_channels[i][int(y):int(y)+y1, int(x):int(x)+x1] = \
        (dstt_channels[i][int(y):int(y)+y1, int(x):int(x)+x1] * (255.0-a*alpha)/ 255 +\
        np.array(scr_channels[i] * (a*alpha) / 255, dtype=np.uint8))

    result = Image.fromarray(dstt_channels.T).convert('RGB')
    return result

import torch
from torch.autograd import Variable
from torchvision import models, transforms

backbone_name = 'resnet101'
#backbone_name = 'alexnet'
#backbone_name = 'vgg16'
#老版本使用model = models.__dict__[backbone_name](pretrained=True) # N x 2048
model = models.__dict__[backbone_name](weights=True) # N x 2048

model.eval()
if torch.cuda.is_available():
    model.cuda()

#输出模型的输出
def label_model(input):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor()]
    )
    input = transform(input)
    input = Variable(torch.unsqueeze(input, dim=0).float(), requires_grad=False)
    return model(input.cuda())

def obj_func(xs):
    imgs_perturbed = alphaBlending(imagepath,watermarkpath, xs)
    prediction = label_model(imgs_perturbed).cpu().detach().numpy()
    prediction=prediction[0][target_class]
    Loss = 1/peak_signal_noise_ratio(np.array(Image.open(imagepath).convert('RGB')),np.array(imgs_perturbed))
    return [prediction,Loss]

# 判断是否攻击成功
def attack_success(xs,targeted_attack=False):
    """
    image:原图像OPEN
    watermark:水印OPEN
    xs代表[x,y:插入的像素位置，
            alpha:透明度，
            scale:缩放系数>>>>>beta:缩放比例，
            rotation:旋转角度]
    targeted_attack是否为有目标的攻击
    """
    attack_image = alphaBlending(imagepath,watermarkpath,xs) # 添加水印

    predict = label_model(attack_image).cpu().detach().cpu().numpy()                # 把模型结果传到cpu解析成numpy数组
    predicted_class = np.argmax(predict)                                            # 模型认为新图像的类别
    #print('Confidence:', predict[0][target_class])

    if ((targeted_attack and predicted_class == target_class) or (not targeted_attack and predicted_class != target_class)):
        return True

class Population:
    def __init__(self, min_range, max_range, dim=5, g=5, p_size=50, object_func=obj_func, co=0.9):
        self.min_range = min_range  # 向量下限
        self.max_range = max_range  # 向量上限
        self.dimension = dim        # 维度
        self.generation = g         # 代数
        self.size = p_size          # 种群数量
        self.cur_round = 0
        self.CO = co                # 交叉系数
        self.get_object_function_value = object_func  # 预测目标的置信度
        self.mutant = None
        self.r = 1
        self.phi = random.randint(0, 4)
        self.epsilon = 0.025
        self.Best_individuality = []

    def initialtion(self):
        #print("种群初始化")
        #self.individuality = [np.array([random.uniform(self.min_range[s], self.max_range[s]) for s in range(self.dimension)]) for i in range(self.size)]
        self.individuality = []
        sum = 0
        while sum <self.size:
            x = []  # 个体，基因
            for j in range(self.dimension):
                x.append((self.min_range[j] + random.random() * (self.max_range[j] - self.min_range[j])))
            if self.get_object_function_value(x)[1]<self.epsilon:
                self.individuality.append(x)
                sum = sum + 1
        self.object_function_values = [self.get_object_function_value(v)[0] for v in self.individuality]

    def mutate(self):
        #print("变异")
        self.mutant = []

        for i in range(self.size):
            # 自定义流域跳跃
            x = np.zeros(5)
            x[0] += np.random.uniform(-4, 4)
            x[1] += np.random.uniform(-4, 4)
            x[2] += np.random.uniform(-0.02, 0.02)
            x[3] += np.random.uniform(-0.02, 0.02)
            x[4] += np.random.uniform(-0.4, 0.4)
            V = self.individuality[i] + x * (self.r)

            # accept()
            for j in range(self.dimension):
                if V[j] > self.max_range[j] or V[j] < self.min_range[j]:
                    V[j] = self.individuality[i][j]
            self.mutant.append(V)

    def crossover(self):
        #print("交叉")
        for i in range(self.size):
            Jrand = random.randint(0, self.dimension)
            for j in range(self.dimension):
                if random.random() <= (self.CO * np.cos(np.pi * self.phi * self.cur_round / 2)) and j != Jrand:
                    self.mutant[i][j] = self.individuality[i][j]
                else:
                    self.mutant[i][j] = self.mutant[i][j]

    def select(self):
        #print("选择")
        for i in range(self.size):
            tmp = self.get_object_function_value(self.mutant[i])
            if tmp[0] <= self.object_function_values[i] and tmp[1]<self.epsilon:
                self.individuality[i] = self.mutant[i]
                self.object_function_values[i] = tmp[0]
            else:
                self.individuality[i] = self.individuality[i]

    def calculate_best(self):
        m = min(self.object_function_values)
        i = self.object_function_values.index(m)
        self.Best_individuality = self.individuality[i]
        # print("Best individuality：", self.individuality[i])
        # print("ObjFuncValue：", m)


    def evolution(self):
        self.initialtion()
        while self.cur_round < self.generation:
            self.mutate()
            self.crossover()
            self.select()
            self.cur_round = self.cur_round + 1
        self.calculate_best()
        return self.Best_individuality

def attack():
    p = Population(min_range=[0,0,0,0.1,-180], max_range=[224, 224,1,1,180], dim=5, g=8, p_size=50, object_func=obj_func, co=0.8)
    best_x = p.evolution()
    best_attack=alphaBlending(imagepath, watermarkpath, best_x)
    return best_x,best_attack


f = open("/media/storage/jijunhao/classes.txt","r")
labels = f.read().splitlines()
f.close()

count=0
attackcount = 0
logo = 'Berkely'
if not os.path.exists('/media/storage/jijunhao/attack/'+backbone_name+"_"+logo):
    os.mkdir('/media/storage/jijunhao/attack/'+backbone_name+"_"+logo)

for i in range(1000):
    imagepath = '/media/storage/jijunhao/imagenet/'+str(i)+'.png'
    watermarkpath = '/media/storage/jijunhao/logo/'+logo+'.png'
    target_class = int(labels[i])
    predict=label_model(Image.open(imagepath)).cpu().detach().numpy()
    if np.argmax(predict)==int(target_class):
        if count<100:
            count=count+1
            print('Attacking No.',i,' images...')
            if os.path.exists('/media/storage/jijunhao/attack/'+backbone_name+"_"+logo+'/'+str(i) +'.png'):    # 断点续算
                attackcount = attackcount + 1
            else:
                time_start = time.time()  # 记录开始时间
                bestx,best_attack = attack()
                time_end = time.time()    # 记录结束时间
                time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
                print("程序耗时:",time_sum)

                # Save
                if attack_success(bestx):
                    attackcount = attackcount+1
                    with open('/media/storage/jijunhao/attack/'+backbone_name+"_"+logo+'/'+'data.txt', 'a') as f:
                        f.write(str(i)+str(bestx))
                    best_attack.save('/media/storage/jijunhao/attack/'+backbone_name+"_"+logo+'/'+str(i)+'.png')
        else:
            break
print('攻击成功率：%.2f%%' % (attackcount/100))