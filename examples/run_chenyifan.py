#coding=utf-8
import pandas as pd
from  sklearn.metrics import log_loss, roc_auc_score
from  sklearn .model_selection import train_test_split
from  sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
import warnings

from deepctr.models import DeepFM
from deepctr.inputs import  SparseFeat, DenseFeat, get_feature_names

if __name__ == "__main__":

    use_plot = True

    warnings.filterwarnings("ignore")
    data = pd.read_csv('./test.csv')

    sparse_features = ['Sparse' + str(i) for i in range(1, 348)]
    chinese_features = ['year', 'provname', 'urbcode', 'birthyear', 'marriage', 'houseowner', 'bathroom', 'familyincm',
                        'expenditure', 'worktype', 'industry']
    sparse_features = sparse_features + chinese_features

    dense_features1 = ['sex',	'eduyr','income',	'earnlvl',	'hauslvl',	'faminlvl',	'birthyear_1',	'urbcode_1',	'sex_1',	'marriage_1',	'houseowner_1',	'bathroom_1',	'education_1'	,'familyincm_1'	,'expenditure_1',	'worktype_1',	'industry_1',	'townincm',	'dispincm',	'workavgincm',	'villmean'
                       ]
    mms = MinMaxScaler(feature_range=(0, 1))

    dense_features = dense_features1
    data[dense_features] = mms.fit_transform(data[dense_features])
    target = ['education']
    data[target] = mms.fit_transform(data[target])

    # 2.count #unique features for each sparse field,and record dense feature field name

    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())
                           for feat in sparse_features] + [DenseFeat(feat, 1,)
                          for feat in dense_features]


    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model

    train, test = train_test_split(data, test_size=0.2)
    train_model_input = {name:train[name] for name in feature_names}
    test_model_input = {name:test[name] for name in feature_names}
    print(type(train_model_input))

    # 4.Define Model,train,predict and evaluate
    model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary')
    # compile(self, optimizer, loss, metrics=None, loss_weights=None, sample_weight_mode=None, weighted_metrics=None,
    #         target_tensors=None)
    model.compile("adam", "binary_crossentropy",
                  metrics=['binary_crossentropy'], )

    history = model.fit(train_model_input, train[target].values,
                        batch_size=256, epochs=30, verbose=2, validation_split=0.2, )
    pred_ans = model.predict(test_model_input, batch_size=256)



    # test[target].values = one_hot.fit_transform(test[target].values)
    # pred_ans = one_hot.fit_transform(pred_ans)

    Logloss = []
    # print(type(log_loss(test[target].values, pred_ans)))
    print("test LogLoss", int(round(log_loss(test[target].values, pred_ans), 10)))
    # print("test AUC", int(round(roc_auc_score(test[target].values, pred_ans), 10)))

# Epoch 8/30
# 20473/20473 - 9s - loss: 0.3565 - binary_crossentropy: 0.3565 - val_loss: 0.3555 - val_binary_crossentropy: 0.3555