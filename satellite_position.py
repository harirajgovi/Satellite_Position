#importing libraries
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings

warnings.filterwarnings("ignore")

#importing datas
df_train = pd.read_csv("train.csv", index_col="epoch", parse_dates=True,
                       infer_datetime_format=True)
df_test = pd.read_csv("test.csv", index_col="epoch", parse_dates=True, 
                      infer_datetime_format=True)

#cleaning datas
df_train.drop("id", axis=1, inplace=True)
test_id = pd.DataFrame(df_test[["id"]].iloc[:].values, columns=["id"])
df_test.drop("id", axis=1, inplace=True)

#grouping datas based on satellites
sat = {"{}".format(i) : df_train[df_train["sat_id"] == i] 
       for i in list(df_train["sat_id"].unique())}

sat_test = {"{}".format(i) : df_test[df_test["sat_id"] == i] 
            for i in list(df_test["sat_id"].unique())}

#splitting datetime evenly
for i in sat.keys():
    idx = pd.date_range(start=sat[i].index[0], end=sat[i].index[-1], periods=len(sat[i]))
    freq = str(idx.hour[1]) + "H" + str(idx.minute[1])+ "T" + str(idx.second[1]) + "S" + str(idx.microsecond[1]) + "U" + str(idx.nanosecond[1]) + "N"
    new_index = pd.date_range(start=sat[i].index[0], end=sat[i].index[-1],
                              freq=freq)
    sat[i] = pd.DataFrame(np.array(sat[i]), index=new_index, columns=list(sat[i].columns))

#training datas
train_data_x = {i : sat[i][["x"]] for i in sat.keys()}
train_data_y = {i : sat[i][["y"]] for i in sat.keys()}
train_data_z = {i : sat[i][["z"]] for i in sat.keys()}
train_data_Vx = {i : sat[i][["Vx"]] for i in sat.keys()}
train_data_Vy = {i : sat[i][["Vy"]] for i in sat.keys()}
train_data_Vz = {i : sat[i][["Vz"]] for i in sat.keys()}

    
#TripleExponentialSmoothing model
model_x = {i : ExponentialSmoothing(train_data_x[i], trend="add", seasonal='add',
                                    seasonal_periods=24).fit(use_brute=False, use_basinhopping=True) for i in sat.keys()}

model_y = {i : ExponentialSmoothing(train_data_y[i], trend="add", seasonal='add', 
                                    seasonal_periods=24).fit(use_brute=False, use_basinhopping=True) for i in sat.keys()}

model_z = {i : ExponentialSmoothing(train_data_z[i], trend="add", seasonal='add', 
                                    seasonal_periods=24).fit(use_brute=False, use_basinhopping=True) for i in sat.keys()}

model_Vx = {i : ExponentialSmoothing(train_data_Vx[i], trend="add", seasonal='add', 
                                     seasonal_periods=24).fit(use_brute=False, use_basinhopping=True) for i in sat.keys()}

model_Vy = {i : ExponentialSmoothing(train_data_Vy[i], trend="add", seasonal='add', 
                                     seasonal_periods=24).fit(use_brute=False, use_basinhopping=True) for i in sat.keys()}

model_Vz = {i : ExponentialSmoothing(train_data_Vz[i], trend="add", seasonal='add', 
                                     seasonal_periods=24).fit(use_brute=False, use_basinhopping=True) for i in sat.keys()}
    
#predicting future datas
predictions_x = {i : model_x[i].forecast(len(sat_test[i])) for i in sat_test.keys()}
predictions_y = {i : model_y[i].forecast(len(sat_test[i])) for i in sat_test.keys()}
predictions_z = {i : model_z[i].forecast(len(sat_test[i])) for i in sat_test.keys()}
predictions_Vx = {i : model_Vx[i].forecast(len(sat_test[i])) for i in sat_test.keys()}
predictions_Vy = {i : model_Vy[i].forecast(len(sat_test[i])) for i in sat_test.keys()}
predictions_Vz = {i : model_Vz[i].forecast(len(sat_test[i])) for i in sat_test.keys()}

x_predicted = pd.DataFrame(pd.concat([predictions_x[i] for i in sat_test.keys()], axis=0).iloc[:].values, columns=["x_fcst"])
y_predicted = pd.DataFrame(pd.concat([predictions_y[i] for i in sat_test.keys()], axis=0).iloc[:].values, columns=["y_fcst"])
z_predicted = pd.DataFrame(pd.concat([predictions_z[i] for i in sat_test.keys()], axis=0).iloc[:].values, columns=["z_fcst"])
Vx_predicted = pd.DataFrame(pd.concat([predictions_Vx[i] for i in sat_test.keys()], axis=0).iloc[:].values, columns=["Vx_fcst"])
Vy_predicted = pd.DataFrame(pd.concat([predictions_Vy[i] for i in sat_test.keys()], axis=0).iloc[:].values, columns=["Vy_fcst"])
Vz_predicted = pd.DataFrame(pd.concat([predictions_Vz[i] for i in sat_test.keys()], axis=0).iloc[:].values, columns=["Vz_fcst"])

test_predictions = pd.concat([x_predicted, y_predicted, z_predicted, 
                              Vx_predicted, Vy_predicted, Vz_predicted], axis=1)

#saving predicted datas to local drive
test_predictions.to_csv("submission.csv", encoding="utf-8", index=False)


    