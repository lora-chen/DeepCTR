#coding=utf-8
import numpy as np
import pandas as pd
import collections
from  sklearn.metrics import log_loss, roc_auc_score
from  sklearn .model_selection import train_test_split
from  sklearn.preprocessing import LabelEncoder, MinMaxScaler


"""
预处理文档

"""
df = pd.read_csv('./selected_data.csv',nrows=900)
#df = pd.read_csv('./selected_data.csv')
df0 = df.fillna(0)

print("loading....")

# 地理数据删除三行
df0.drop(['prefname','cntyname','townname', 'villname'],axis=1,inplace=True)
# df0.replace([ '(' , ')' , '[' , ']' ,'-'], '', inplace=True)

# create dict
dict_replace = {'??':0 ,-1:0}
"""
print (df['education'].unique())
['小学及以下' '初中' '高中/中专/大学' '大专/高职' '本科']
对应为1,2,3,4,5
"""
arrary_education = df['education'].unique()
ii=1
for element in arrary_education:
    dict_replace[element] = ii
    ii=ii+1
df0.replace(dict_replace,inplace=True)

"""
将中文向量列名转成可迭代输入的形式
"""
dict_columns = collections.OrderedDict()
arrary_columns = ['行政地标','村庄','房地产','政府机构','住宅区','购物','教育培训','公司企业','行政单位','幼儿园','各级政府','公司','医疗','金融','汽车服务','其他','小学','酒店','超市','写字楼','内部楼栋','商铺','银行','家居建材','美食','交通设施','公检法机构','厂矿','中餐厅','旅游景点','中学','出入口','门','园区','汽车维修','诊所','乡镇','桥','家电数码','休闲娱乐','购物中心','汽车销售','综合医院','信用社','培训机构','快捷酒店','市场','文物古迹','汽车美容','自然地物','山峰','公园','四星级','洗浴按摩','专科医院','长途汽车站','休闲广场','高等院校','农林园艺','宿舍','运动健身','生活服务','教堂','居民委员会','路侧停车位','福利机构','风景区','景点','疗养院','体育场馆','电影院','港口','投资理财','火车站','文化传媒','三星级','农家院','科研机构','社会团体','汽车配件','游乐园','充电站','星级酒店','图书馆','剧院','博物馆','展览馆','公寓式酒店','ktv','疾控中心','邮局','外国餐厅','党派团体','水系','文化宫','度假村','汽车检测场','咖啡厅','物流公司','房产中介机构','健身中心','政治教育机构','汽车租赁','内部楼号','急救中心','服务区','美术馆','涉外机构','殡葬服务','五星级','亲子教育','行政区划','典当行','植物园','公用事业','医疗保健','p_行政地标','p_村庄','p_房地产','p_政府机构','p_住宅区','p_购物','p_教育培训','p_公司企业','p_行政单位','p_幼儿园','p_各级政府','p_公司','p_医疗','p_金融','p_汽车服务','p_其他','p_小学','p_酒店','p_超市','p_写字楼','p_内部楼栋','p_商铺','p_银行','p_家居建材','p_美食','p_交通设施','p_公检法机构','p_厂矿','p_中餐厅','p_旅游景点','p_中学','p_出入口','p_门','p_园区','p_汽车维修','p_诊所','p_乡镇','p_桥','p_家电数码','p_休闲娱乐','p_购物中心','p_汽车销售','p_综合医院','p_信用社','p_培训机构','p_快捷酒店','p_市场','p_文物古迹','p_汽车美容','p_自然地物','p_山峰','p_公园','p_四星级','p_洗浴按摩','p_专科医院','p_长途汽车站','p_休闲广场','p_高等院校','p_农林园艺','p_宿舍','p_运动健身','p_生活服务','p_教堂','p_居民委员会','p_路侧停车位','p_福利机构','p_风景区','p_景点','p_疗养院','p_体育场馆','p_电影院','p_港口','p_投资理财','p_火车站','p_文化传媒','p_三星级','p_农家院','p_科研机构','p_社会团体','p_汽车配件','p_游乐园','p_充电站','p_星级酒店','p_图书馆','p_剧院','p_博物馆','p_展览馆','p_公寓式酒店','p_ktv','p_疾控中心','p_邮局','p_外国餐厅','p_党派团体','p_水系','p_文化宫','p_度假村','p_汽车检测场','p_咖啡厅','p_物流公司','p_房产中介机构','p_健身中心','p_政治教育机构','p_汽车租赁','p_内部楼号','p_急救中心','p_服务区','p_美术馆','p_涉外机构','p_殡葬服务','p_五星级','p_亲子教育','p_行政区划','p_典当行','p_植物园','p_公用事业','p_医疗保健','i_行政地标','i_村庄','i_房地产','i_政府机构','i_住宅区','i_购物','i_教育培训','i_公司企业','i_行政单位','i_幼儿园','i_各级政府','i_公司','i_医疗','i_金融','i_汽车服务','i_其他','i_小学','i_酒店','i_超市','i_写字楼','i_内部楼栋','i_商铺','i_银行','i_家居建材','i_美食','i_交通设施','i_公检法机构','i_厂矿','i_中餐厅','i_旅游景点','i_中学','i_出入口','i_门','i_园区','i_汽车维修','i_诊所','i_乡镇','i_桥','i_家电数码','i_休闲娱乐','i_购物中心','i_汽车销售','i_综合医院','i_信用社','i_培训机构','i_快捷酒店','i_市场','i_文物古迹','i_汽车美容','i_自然地物','i_山峰','i_公园','i_四星级','i_洗浴按摩','i_专科医院','i_长途汽车站','i_休闲广场','i_高等院校','i_农林园艺','i_宿舍','i_运动健身','i_生活服务','i_教堂','i_居民委员会','i_路侧停车位','i_福利机构','i_风景区','i_景点','i_疗养院','i_体育场馆','i_电影院','i_港口','i_投资理财','i_火车站','i_文化传媒','i_三星级','i_农家院','i_科研机构','i_社会团体','i_汽车配件','i_游乐园','i_充电站','i_星级酒店','i_图书馆','i_剧院','i_博物馆','i_展览馆','i_公寓式酒店','i_ktv','i_疾控中心','i_邮局','i_外国餐厅','i_党派团体','i_水系','i_文化宫','i_度假村','i_汽车检测场','i_咖啡厅','i_物流公司','i_房产中介机构','i_健身中心','i_政治教育机构','i_汽车租赁','i_内部楼号','i_急救中心','i_服务区','i_美术馆','i_涉外机构','i_殡葬服务','i_五星级','i_亲子教育','i_行政区划','i_典当行','i_植物园','i_公用事业','i_医疗保健'
                    ]

ii= 1
for yy in range(len(arrary_columns)):
    dict_columns[arrary_columns[yy]] =  'Sparse' + str(ii)
    ii=ii+1
df0.rename(columns=(dict_columns),inplace=True)

# 1.Label Encoding for sparse features,and do simple Transformation for dense features

sparse_features = ['Sparse' + str(i) for i in range(1, 348)]
# chinese_features = [ 'year',	'provname','urbcode',	'birthyear','marriage']
chinese_features = [ 'year',	'provname','urbcode',	'birthyear',	'marriage',	'houseowner',	'bathroom'	,'familyincm',	'expenditure',	'worktype'	,'industry']
sparse_features  = sparse_features + chinese_features

for feat in sparse_features:
        lbe = LabelEncoder()
        df0[feat]=str(df0[feat])
        df0[feat] = lbe.fit_transform(df0[feat])

# 归一化
norm_feature = ['income',	'birthyear_1',	'dispincm'	,'villmean' ,'education'      ]
norm_feature = ['income',	'birthyear_1',	'dispincm'	,'villmean'       ]
mms = MinMaxScaler(feature_range=(0, 1))
df0[norm_feature ] = mms.fit_transform(df0[norm_feature ])
df0.to_csv("test.csv",index=False,sep=',',encoding = 'utf_8_sig')

print("Over")


















