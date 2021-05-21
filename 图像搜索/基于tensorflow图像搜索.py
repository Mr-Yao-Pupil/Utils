import time

import h5py
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input as preprocess_input_densenet


class VGGNet:
    def __init__(self):
        # weights: 'imagenet'
        # pooling: 'max' or 'avg'
        # input_shape: (width, height, 3), width and height should >= 48
        self.input_shape = (224, 224, 3)
        self.weight = "imagenet"
        self.pooling = 'avg'  # max
        # include_top：是否保留顶层的3个全连接网络
        # weights：None代表随机初始化，即不加载预训练权重。'imagenet'代表加载预训练权重
        # input_tensor：可填入Keras tensor作为模型的图像输出tensor
        # input_shape：可选，仅当include_top=False有效，应为长为3的tuple，指明输入图片的shape，图片的宽高必须大于48，如(200,200,3)
        # pooling：当include_top = False时，该参数指定了池化方式。None代表不池化，最后一个卷积层的输出为4D张量。‘avg’代表全局平均池化，‘max’代表全局最大值池化。
        # classes：可选，图片分类的类别数，仅当include_top = True并且不加载预训练权重时可用。

        # self.model_vgg = VGG16(weights=self.weight,
        #                        input_shape=(self.input_shape[0], self.input_shape[1], self.input_shape[2]),
        #                        pooling=self.pooling, include_top=False)

        # self.model_resnet = ResNet50(weights=self.weight,
        #                              input_shape=(self.input_shape[0], self.input_shape[1], self.input_shape[2]),
        #                              pooling=self.pooling, include_top=False)

        self.model_densenet = DenseNet121(weights=self.weight,
                                          input_shape=(self.input_shape[0], self.input_shape[1], self.input_shape[2]),
                                          pooling=self.pooling, include_top=False)

        # self.model_mobilenet_v2 = MobileNetV2(weights=self.weight,
        #                                       input_shape=(
        #                                           self.input_shape[0], self.input_shape[1], self.input_shape[2]),
        #                                       pooling=self.pooling, include_top=False)

        # self.model_nasnet_mobile = NASNetMobile(weights=self.weight,
        #                                         input_shape=(
        #                                             self.input_shape[0], self.input_shape[1], self.input_shape[2]),
        #                                         pooling=self.pooling, include_top=False)
        #
        # self.model_inception_v3 = InceptionV3(weights=self.weight,
        #                                       input_shape=(
        #                                           self.input_shape[0], self.input_shape[1], self.input_shape[2]),
        #                                       pooling=self.pooling, include_top=False)
        #
        # self.model_xception = Xception(weights=self.weight,
        #                                input_shape=(
        #                                    self.input_shape[0], self.input_shape[1], self.input_shape[2]),
        #                                pooling=self.pooling, include_top=False)
        # self.model_vgg.predict(np.zeros((1, 224, 224, 3)))
        # self.model_resnet.predict(np.zeros((1, 224, 224, 3)))
        self.model_densenet.predict(np.zeros((1, 224, 224, 3)))
        # self.model_mobilenet_v2.predict(np.zeros((1, 224, 224, 3)))
        # self.model_nasnet_mobile.predict(np.zeros((1, 224, 224, 3)))
        # self.model_inception_v3.predict(np.zeros((1, 224, 224, 3)))
        # self.model_xception.predict(np.zeros((1, 224, 224, 3)))

    '''
    Use vgg16/Resnet model to extract features
    Output normalized feature vector
    '''

    # 提取vgg16最后一层卷积特征
    # def vgg_extract_feat(self, img_path):
    #     img = image.load_img(img_path, target_size=(self.input_shape[0], self.input_shape[1]))
    #     img = image.img_to_array(img)
    #     img = np.expand_dims(img, axis=0)
    #     img = preprocess_input_vgg(img)
    #     feat = self.model_vgg.predict(img)
    #     norm_feat = feat[0] / LA.norm(feat[0])
    #     return norm_feat

    # 提取resnet50最后一层卷积特征
    # def resnet_extract_feat(self, img_path):
    #     img = image.load_img(img_path, target_size=(self.input_shape[0], self.input_shape[1]))
    #     img = image.img_to_array(img)
    #     img = np.expand_dims(img, axis=0)
    #     img = preprocess_input_resnet(img)
    #     feat = self.model_resnet.predict(img)
    #     norm_feat = feat[0] / LA.norm(feat[0])
    #     return norm_feat

    # 提取densenet121最后一层卷积特征
    def densenet_extract_feat(self, img_path):
        img = image.load_img(img_path, target_size=(self.input_shape[0], self.input_shape[1]))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input_densenet(img)
        feat = self.model_densenet.predict(img)
        # print(feat.shape)
        norm_feat = feat[0] / LA.norm(feat[0])
        return norm_feat

    # def mobilenet_v2_extract_feat(self, img_path):
    #     img = image.load_img(img_path, target_size=(self.input_shape[0], self.input_shape[1]))
    #     img = image.img_to_array(img)
    #     img = np.expand_dims(img, axis=0)
    #     img = preprocess_input_mobilenet_v2(img)
    #     feat = self.model_mobilenet_v2.predict(img)
    #     norm_feat = feat[0] / LA.norm(feat[0])
    #     return norm_feat
    #
    # def nasnet_mobile_extract_feat(self, img_path):
    #     img = image.load_img(img_path, target_size=(self.input_shape[0], self.input_shape[1]))
    #     img = image.img_to_array(img)
    #     img = np.expand_dims(img, axis=0)
    #     img = preprocess_input_nasnet(img)
    #     feat = self.model_nasnet_mobile.predict(img)
    #     norm_feat = feat[0] / LA.norm(feat[0])
    #     return norm_feat

    # def inception_v3_extract_feat(self, img_path):
    #     img = image.load_img(img_path, target_size=(self.input_shape[0], self.input_shape[1]))
    #     img = image.img_to_array(img)
    #     img = np.expand_dims(img, axis=0)
    #     img = preprocess_input_inception_v3(img)
    #     feat = self.model_inception_v3.predict(img)
    #     norm_feat = feat[0] / LA.norm(feat[0])
    #     return norm_feat

    # def xception_extract_feat(self, img_path):
    #     img = image.load_img(img_path, target_size=(self.input_shape[0], self.input_shape[1]))
    #     img = image.img_to_array(img)
    #     img = np.expand_dims(img, axis=0)
    #     img = preprocess_input_xception(img)
    #     feat = self.model_xception.predict(img)
    #     norm_feat = feat[0] / LA.norm(feat[0])
    #     return norm_feat


query = r'C:\Users\duizhuang\Desktop\test_data\52.png'
index = 'black_ground_features_densenet121_avg_pool.h5'
result = r'D:\IDE\sources\1\black_ground_classes_data'
# read in indexed images' feature vectors and corresponding image names
st = time.time()
h5f = h5py.File(index, 'r')
# feats = h5f['dataset_1'][:]
feats = h5f['dataset_1'][:]
print(feats)
imgNames = h5f['dataset_2'][:]
print(imgNames)
h5f.close()
print("使用时间：", time.time() - st)

print("--------------------------------------------------")
print("               searching starts")
print("--------------------------------------------------")

# read and show query image
# queryDir = args["query"]

# 展示原图
queryImg = mpimg.imread(query)
plt.title("Query Image")
plt.imshow(queryImg)
plt.show()
st = time.time()

# init VGGNet16 model
model = VGGNet()

print("读取模型时间：", time.time() - st)

st = time.time()
# extract query image's feature, compute simlarity score and sort
queryVec = model.densenet_extract_feat(query)  # 修改此处改变提取特征的网络
print("模型提取特征时间：", time.time() - st)

print(queryVec.shape)
print(feats.shape)

st = time.time()
scores = np.dot(queryVec, feats.T)
rank_ID = np.argsort(scores)[::-1]
rank_score = scores[rank_ID]
# print (rank_ID)
print(rank_score)
print("排序时间：", time.time() - st)

st = time.time()
# number of top retrieved images to show
maxres = 10  # 检索出三张相似度最高的图片
imlist = []
for i, index in enumerate(rank_ID[0:maxres]):
    imlist.append(imgNames[index])
    # print(type(imgNames[index]))
    print("image names: " + str(imgNames[index]) + " scores: %f" % rank_score[i])
print("top %d images in order are: " % maxres, imlist)
print("排序时间：", time.time() - st)

# 展示结果
# show top #maxres retrieved result one by one
for i, im in enumerate(imlist):
    image = mpimg.imread(result + "/" + str(im, 'utf-8'))
    plt.title("search output %d" % (i + 1))
    plt.imshow(image)
    plt.show()
