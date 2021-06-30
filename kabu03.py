import sys
from yahoo_finance_api2 import share
from yahoo_finance_api2.exceptions import YahooFinanceError
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import math
from torch.optim import SGD
import numpy as np
from matplotlib import animation
import statistics
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import random
import datetime


codelist = pd.read_excel("data_j.xls")

stock_list = []       
filename = ""  

filename = "ETF・ETN"
codelist_selected = codelist[codelist["市場・商品区分"]=="ETF・ETN"]["コード"].values.tolist()
for i in range(10):
    stock_list.extend([codelist_selected[np.random.randint(0, len(codelist_selected) - 1)]])


filename = "市場第一部（内国株）"
codelist_selected = codelist[codelist["市場・商品区分"]=="市場第一部（内国株）"]["コード"].values.tolist()
for i in range(10):
    stock_list.extend([codelist_selected[np.random.randint(0, len(codelist_selected) - 1)]])


# stock_list.extend(range(1617, 1634))                                                          # ETF17日本株(業種別)
                                                                
# stock_list.extend([1407,1419,1448,1628,1671,1699,4825,5017,6905,7013,7294,9793,9908])
# stock_list.extend([1407,1419,1448,1628])
# stock_list.extend([1628,1671,1699,2148,3992,4825,6703,6752,6770,6905,7013,7294,7296,9684])    # 所有証券
# stock_list.extend(range(1300, 1400))                                                          # ETF探訪1300-1400
# stock_list.extend(range(1400, 1500))                                                          # ETF探訪1400-1500
# stock_list.extend(range(1500, 1600))                                                          # ETF探訪1500-1600
# stock_list.extend(range(1600, 1700))                                                          # ETF探訪1600-1700
# stock_list.extend(range(5050, 5100))                                                          # 個別株探訪5000-5100
# stock_list.extend(range(9600, 9700))                                                          # 個別株探訪9600-9700
# stock_list.extend(range(9700, 9800))                                                          # 個別株探訪9700-9800
# stock_list.extend(range(9800, 9900))                                                          # 個別株探訪9800-9900
# stock_list.extend(range(9900, 10000))                                                          # 個別株探訪9900-10000

# stock_list.extend([6532,9613,9739,4307,2352,3626,2492,4776,6702,3923]) # DX
# stock_list.extend([1407,9517,9519,5074,6361,3150,1945,4237,2151,4169]) # 再生可能エネルギー
# stock_list.extend(range(2552, 2573))                                                          # ETF・ETN
# stock_list.extend(range(2620, 2637))                                                          # ETF・ETN
# stock_list.extend(range(1385, 1400))                                                          # ETF・ETN


# stock_list.extend([4927,5017,9603,9793,9878,9908,7296,6770,3992,2148,1628,1407,1419,1430,1448,1570,1680])                                                      
# stock_list.extend([4927,5017,9603,1407,1430,1570,1680])                                                      
# stock_list.extend([1671,6440,4927,5021,9621,9603,1344,1365])                                                
# stock_list.extend([6905,6806,5491,4463,8963,2408,4380,7294,3267])              
#                              
# stock_list.extend([6806,5491,4463,4380])                         # 売買寄成                                                                          
# stock_list.extend(range(1300+17, 10000, 150))                          
# stock_list.extend([6905,6806,5491,4463,8963,2408,4380,7294,3267])            
# stock_list.extend([9793])                 
# stock_list.extend([6951,7944])            


# filename = "所有証券"
# stock_list.extend([1323,1407,1419,1448,1628,1671,1699,2038,4825,5017,6806,6905,7013,7687,9793,9908])                         # 所有証券      


# stock_list.extend([2515,1322,2529,3071,9115,5216,8789])                         # 買引成  
sampling = 3


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def mkRandomBatch(train_x, train_t, batch_size=10):
    """
    train_x, train_tを受け取ってbatch_x, batch_tを返す。
    """
    batch_x = []
    batch_t = []

    for _ in range(batch_size):
        idx = np.random.randint(0, len(train_x) - 1)
        batch_x.append(train_x[idx])
        batch_t.append(train_t[idx])
    
    return torch.tensor(batch_x).to(device), torch.tensor(batch_t).to(device)

class Predictor(nn.Module):
    def __init__(self, inputDim, hiddenDim, outputDim):
        super(Predictor, self).__init__()

        self.rnn = nn.LSTM(input_size = inputDim,
                            hidden_size = hiddenDim,
                            batch_first = True).to(device)
        self.output_layer = nn.Linear(hiddenDim, outputDim).to(device)
    
    def forward(self, inputs, hidden0=None):
        inputs = inputs.to(device)
        # output, (hidden, cell) = self.rnn(inputs, None) #LSTM層
        output, hidden = self.rnn(inputs, None) #LSTM層
        output = self.output_layer(output[:, -1, :]) #全結合層

        return output.to(device)



index = 0
pd_record = pd.DataFrame(columns=['コード', '銘柄名', '予想初値（正答率）', '予想終値（正答率）', '予想初値', '予想終値', '予想初値（明後日）', '予想終値（明後日）', '行動'])

for stock in stock_list:
    print(stock)

    ############## 株価インポート #############################
    my_share = share.Share(str(stock)+'.T')
    symbol_data = None
    try:
        symbol_data = my_share.get_historical(
            share.PERIOD_TYPE_WEEK, 200,
            share.FREQUENCY_TYPE_DAY, 1)
    except YahooFinanceError as e:
        print(e.message)
        continue
    ###########################################################

    df = pd.DataFrame(symbol_data)
    # df[:-1]                                                         # 取引時間中
    data_size = len(df.index)

    if data_size<100:
        continue
    if np.isnan(df.close[data_size-1]):
        continue

    df["datetime"]          = pd.to_datetime(df.timestamp, unit="ms") + datetime.timedelta(hours=9)
    df["countdown_days"]    = (df.timestamp - df.timestamp[len(df.timestamp)-1])/1000/60/60/24

    df["varate_yes"]    = None
    df["varrate_day"]   = None
    df["gap"]           = None
    for idx in range(1, data_size):
        df["varate_yes"][idx]   = (df.close[idx] - df.close[idx-1])/df.close[idx-1]
        df["varrate_day"][idx]  = (df.close[idx] - df.open[idx])/df.close[idx-1]
        df["gap"][idx]          = (df.open[idx] - df.close[idx-1])/df.close[idx-1]
    # print(df.head)


    # fig = plt.figure()
    # plt.scatter(df["timestamp"], df["open"])
    # plt.scatter(df["timestamp"], df["close"])
    # fig.savefig("img.png")

    # fig = plt.figure()
    # plt.scatter(df["countdown_days"], df["open"])
    # plt.scatter(df["countdown_days"], df["close"])
    # fig.savefig("countdown_days.png")

    # fig = plt.figure()
    # plt.scatter(df["countdown_days"], df["varate_yes"])
    # fig.savefig("前日比.png")

    fig = plt.figure()
    plt.scatter(df["countdown_days"], df["gap"])
    fig.savefig("gap.png")


    # fig = plt.figure()
    # plt.scatter(df["varate_yes"][1:277], df["varrate_day"][0:276])
    # fig.savefig("varrate_rel.png")

    # print(df.head)

    scaler = RobustScaler()
    scaler2 = RobustScaler()
    scaler3 = MinMaxScaler(feature_range=(0, 1))
    scaler4 = MinMaxScaler(feature_range=(0, 1))
    df[["varate_yes_scl"]] = scaler3.fit_transform(scaler.fit_transform(df[["varate_yes"]]).astype('float64'))
    df[["gap_scl"]] = scaler4.fit_transform(scaler2.fit_transform(df[["gap"]]).astype('float64'))




    fig = plt.figure()
    plt.scatter(df["countdown_days"], df["varate_yes_scl"])
    fig.savefig("scaler.png")

    fig = plt.figure()
    plt.scatter(df["countdown_days"], df["gap_scl"])
    fig.savefig("scaler2.png")


    ############################# < 点検済み > ##########################

    print(df.head)
    train_x = []
    train_t = []

    period = 39
    for offset in range(1, data_size-period):
        train_x.append(df[["varate_yes_scl","gap_scl"]][offset:offset+period].values.reshape([-1,2]).tolist())
        train_t.append([[df[["varate_yes_scl","gap_scl"]].iloc[offset+period,:].values.reshape([-1,2]).tolist()]])
        # train_x.append(df["varate_yes_scl"][offset:offset+period].values.reshape([-1,1]).tolist())
        # train_t.append([[df["varate_yes_scl"].iloc[offset+period]]])
        # train_t.append([[df["varate_yes_scl"].iloc[offset+period,:].values.reshape([-1,2]).tolist()]])



        # print(df[["varate_yes_scl","gap_scl"]].iloc[offset+period,:])
        # train_t.append([[df[["varate_yes_scl","gap_scl"]][offset+period]]].values.reshape([-1,2]).tolist())

    train_t = np.array(train_t).reshape(-1, 2).tolist()

    x_train, x_test, t_train, t_test = train_test_split(train_x, train_t, test_size=0.2)
    training_size   = len(t_train)
    test_size       = len(t_test)


    train_tomorrow = df[["varate_yes_scl","gap_scl"]][data_size-period:data_size].values.reshape([-1,2]).tolist()

    # train_tomorrow2 = df[["varate_yes_scl","gap_scl"]][data_size-period:data_size].values.reshape([-1,2]).tolist()
    # train_tomorrow = df["varate_yes_scl"][data_size-period:data_size].values.reshape([-1,1]).tolist()

    print(train_x[0])
    print(train_t[0])

    
    epochs_num = 47 #traningのepoch回数
    hidden_size = 100 #LSTMの隠れ層の次元数

    hatune_log = []
    owarine_log = []
    sofarbestmodel = None
    sofarbestmodel_acc = 0

    newscl = 0.005
    owarine_ans = scaler3.transform(scaler.transform([[newscl]]))[0,0]
    owarine_ansm = scaler3.transform(scaler.transform([[-newscl]]))[0,0]
    hatune_ans = scaler4.transform(scaler2.transform([[newscl]]))[0,0]
    hatune_ansm = scaler4.transform(scaler2.transform([[-newscl]]))[0,0]

    owarine_ans = (owarine_ans-owarine_ansm)/2.0
    hatune_ans = (hatune_ans-hatune_ansm)/2.0
    print(owarine_ans)
    print(hatune_ans)
    
    for rept in range(sampling):

        model = Predictor(2, hidden_size, 2) #modelの宣言

        criterion = nn.MSELoss() #評価関数の宣言
        optimizer = SGD(model.parameters(), lr=0.01) #最適化関数の宣言




        training_accuracy_log = []
        training_accuracy_log2 = []
        batch_size = 33
        
        for epoch in range(epochs_num):
            # training
            running_loss = 0.0
            training_accuracy = 0.0
            training_accuracy2 = 0.0
            for i in range(int(training_size/batch_size)):
                optimizer.zero_grad()
                offset = i * batch_size
                # data = torch.tensor([train_x[i]]).to(device)
                # label = torch.tensor([train_t[i]]).to(device)
                # data, label = mkRandomBatch(x_train, t_train, batch_size)
                # print(x_train[offset:offset+batch_size])

                data    = torch.tensor(x_train[offset:offset+batch_size]).to(device)
                label   = torch.tensor(t_train[offset:offset+batch_size]).to(device)

                output = model(data)

                loss = criterion(output.float(), label.float())
                loss.backward()
                optimizer.step()

                running_loss += loss.data
                output_cpu = output.to('cpu').detach()
                label_cpu = label.to('cpu').detach()
                
                training_accuracy += np.sum(np.abs(torch.reshape(output_cpu.data[:,0], (-1,1)) - torch.reshape(torch.reshape(label_cpu.data, (-1,2))[:,0], (-1,1,1))).numpy() < owarine_ans) #outputとlabelの誤差が0.1以内なら正しいとみなす。
                training_accuracy2 += np.sum(np.abs(torch.reshape(output_cpu.data[:,1], (-1,1)) - torch.reshape(torch.reshape(label_cpu.data, (-1,2))[:,1], (-1,1,1))).numpy() < hatune_ans) #outputとlabelの誤差が0.1以内なら正しいとみなす。
                
            
            # テスト
            test_accuracy = 0.0
            test_accuracy2 = 0.0
            for i in range(int(test_size / batch_size)):
                offset = i * batch_size
                # data, label = torch.tensor(x_test[offset:offset+batch_size]), torch.tensor(t_test[offset:offset+batch_size])
                data    = torch.tensor(x_test[offset:offset+batch_size]).to(device)
                label   = torch.tensor(t_test[offset:offset+batch_size]).to(device)
                output = model(data, None)

                
                output_cpu = output.to('cpu').detach()
                label_cpu = label.to('cpu').detach()
                test_accuracy += np.sum(np.abs(torch.reshape(output_cpu.data[:,0], (-1,1)) - torch.reshape(torch.reshape(label_cpu.data, (-1,2))[:,0], (-1,1,1))).numpy() < owarine_ans) #outputとlabelの誤差が0.1以内なら正しいとみなす。
                test_accuracy2 += np.sum(np.abs(torch.reshape(output_cpu.data[:,1], (-1,1)) - torch.reshape(torch.reshape(label_cpu.data, (-1,2))[:,1], (-1,1,1))).numpy() < hatune_ans) #outputとlabelの誤差が0.1以内なら正しいとみなす。


            training_accuracy /= training_size * batch_size
            training_accuracy2 /= training_size * batch_size

            test_accuracy /= test_size * batch_size
            test_accuracy2 /= test_size * batch_size

            print('%d loss: %.3f, training_accuracy : %.5f, test_accuracy : %.5f' % (epoch + 1, running_loss, training_accuracy, test_accuracy))
            print('%d loss: %.3f, training_accuracy2: %.5f, test_accuracy2: %.5f' % (epoch + 1, running_loss, training_accuracy2, test_accuracy2))
        
        training_accuracy_log.append(test_accuracy)
        training_accuracy_log2.append(test_accuracy2)

        if sofarbestmodel_acc < test_accuracy + test_accuracy2:
            sofarbestmodel_acc = test_accuracy + test_accuracy2
            sofarbestmodel = model

        data_tomrrow = torch.tensor([train_tomorrow]).to(device)
        output_tomrrow = model(data_tomrrow)
        print(output.data[0])
        output_np = output.to('cpu').detach().numpy().copy()

        print(output_np)

        output_np_inv = scaler.inverse_transform(scaler3.inverse_transform(output_np[:,0].reshape([-1, 1])))
        output_np_inv2 = scaler2.inverse_transform(scaler4.inverse_transform(output_np[:,1].reshape([-1, 1])))
        print(output_np_inv[len(output_np_inv)-1])
        print('初値：{:.2%}'.format(output_np_inv2[len(output_np_inv2)-1][0]))
        print('終値：{:.2%}'.format(output_np_inv[len(output_np_inv)-1][0]))

        hatune_log.append(output_np_inv2[len(output_np_inv2)-1][0])
        owarine_log.append(output_np_inv[len(output_np_inv)-1][0])

    # hatune_log.remove(min(hatune_log))
    # hatune_log.remove(max(hatune_log))
    # owarine_log.remove(min(owarine_log))
    # owarine_log.remove(max(owarine_log))

    owarine_acc = statistics.mean(training_accuracy_log)
    hatune_acc = statistics.mean(training_accuracy_log2)

    hatune = statistics.mean(hatune_log)
    owarine = statistics.mean(owarine_log)
    hatune_v = np.std(np.array(hatune_log)).tolist()
    owarine_v = np.std(np.array(owarine_log)).tolist()


            
    # output_np_inv2 = scaler2.inverse_transform([output_np[1]])
    # # print(output_np_inv[0,0])　
    # print('{:.2%}'.format(output_np_inv2[0]))
    print(stock, codelist[codelist['コード'] == stock]["銘柄名"])

    # hatune_out = '{:.2%}'.format(hatune) + ' ± ' + '{:.2%}'.format(hatune_v)
    # owarine_out = '{:.2%}'.format(owarine) + ' ± ' + '{:.2%}'.format(owarine_v)
    hatune_out = '{:.2%}'.format(hatune) 
    owarine_out = '{:.2%}'.format(owarine)
    hatune_acc_out = '{:.2%}'.format(hatune_acc)
    owarine_acc_out = '{:.2%}'.format(owarine_acc)
    # pd_record.append(pd.Series([str(stock), codelist[codelist['コード'] == stock]["銘柄名"], '{:.2%}'.format(output_np_inv2[len(output_np_inv2)-1][0]), '{:.2%}'.format(output_np_inv[len(output_np_inv)-1][0])]), ignore_index=True)

    ################## 翌々日 ########################
    train_tomorrow = np.array(train_tomorrow).reshape([-1, 2])

    train_tomorrow = np.delete(train_tomorrow, 0, 0)
    print(scaler3.transform(scaler.transform([[owarine]])))
    print(scaler4.transform(scaler2.transform([[hatune]])))
    train_tomorrow = np.append(train_tomorrow, [[scaler3.transform(scaler.transform([[owarine]]))[0,0], scaler4.transform(scaler2.transform([[hatune]]))[0,0]]], 0)        # ここ修正

    print(train_tomorrow[0,:])
    print(train_tomorrow[-1,:])

    # df_tomorrow0 = scaler.transform(train_tomorrow[-1, 0]).astype('float64')
    # df_tomorrow1 = scaler2.transform(train_tomorrow[-1, 1]).astype('float64')


    # df_tomorrow0 = scaler.transform(pd.DataFrame(train_tomorrow[:][0])).astype('float64')
    # df_tomorrow1 = scaler2.transform(pd.DataFrame(train_tomorrow[:][1])).astype('float64')
    # df_tomorrow = np.concatenate([df_tomorrow0, df_tomorrow1],1)
    # train_tomorrow = df[["varate_yes_scl","gap_scl"]].iloc[100].values.reshape([-1,2]).tolist()

    data_tomrrow = torch.tensor([train_tomorrow.tolist()]).to(device)
    output_tomrrow = sofarbestmodel(data_tomrrow)
    # print(output_tomrrow.data[0])
    output_np = output_tomrrow.to('cpu').detach().numpy().copy()

    # print(output_np)

    output_np_inv = scaler.inverse_transform(scaler3.inverse_transform(output_np[:,0].reshape([-1, 1])))
    output_np_inv2 = scaler2.inverse_transform(scaler4.inverse_transform(output_np[:,1].reshape([-1, 1])))
    print(output_np_inv[len(output_np_inv)-1])
    print('初値：{:.2%}'.format(output_np_inv2[len(output_np_inv2)-1][0]))
    print('終値：{:.2%}'.format(output_np_inv[len(output_np_inv)-1][0]))

    hatune2 = output_np_inv2[len(output_np_inv2)-1][0]
    owarine2 = output_np_inv[len(output_np_inv)-1][0]
    hatune_tdat = '{:.2%}'.format(hatune2 + owarine)
    owarine_tdat = '{:.2%}'.format(owarine2 + owarine)


    ######################### 判断 ###########################
    comment = ""

    cap_odd = 1.0
    # cap_odd = 2.0     #　S株、キャンペーン時はコメントアウト
    dev_line = 0.0033 * cap_odd
    buy_line = dev_line * 1.0
    cut_line = -0.001
    day_line = 0.002

    price = np.array([hatune, owarine, hatune2+owarine, owarine2+owarine])
    price_indv = np.array([hatune, owarine, hatune2, owarine2])
    place = ["寄成", "引成", "寄成+1D", "引成+1D"]

    pmax = np.argmax(price)
    pmin = np.argmin(price)
    indmax = np.argmax(price_indv)
    indmin = np.argmin(price_indv)

    if price[pmax] - price[pmin] > day_line:
        if pmin < pmax:
            comment = "買" + place[pmin]
        else:
            comment = "売" + place[pmax]
    # else:
    #     if price_indv[indmin]>

    # if owarine<=cut_line:
    #     if hatune>=owarine:
    #         comment = "寄成売"
    #     else:
    #         comment += "引成売"
    #         if owarine-hatune>day_line:
    #             comment += "& 寄成買"
    # elif owarine>=dev_line:
    #     if hatune<=owarine:
    #         comment = "寄成買"
    #         if owarine-hatune>buy_line:
    #             comment += "（必）"
    #     else:
    #         comment += "引成買"
    #         # if hatune-owarine>buy_line:
    #         #     comment += "引成買"
    #         if hatune-owarine>day_line:
    #             comment += " & 寄成売"
    # else:
    #     if hatune-owarine>day_line:
    #         comment += "寄成売"
        
    #     if owarine-hatune>day_line:
    #         comment += "寄成買 & 引成売"


    index += 1
    pd_record.loc[str(index)] = [str(stock), codelist[codelist['コード'] == stock]["銘柄名"].values[0], hatune_acc_out, owarine_acc_out, hatune_out, owarine_out, hatune_tdat, owarine_tdat, comment]
    # output_np_inv2 = scaler2.inverse_transform([output_np[1]])
    # # print(output_np_inv[0,0])　
    # print('{:.2%}'.format(output_np_inv2[0]))
    print(stock, codelist[codelist['コード'] == stock]["銘柄名"])

    # fig = plt.figure()
    # plt.plot(training_accuracy_log)
    # plt.plot(training_accuracy_log2)
    # plt.show()

now = datetime.datetime.now()
pd_record.to_excel("pd_record_"+'{:02d}{:02d}_{:02d}{:02d}_'.format(now.month, now.day, now.hour, now.minute)+filename+".xlsx")
