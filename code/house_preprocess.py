import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm , skew
from scipy.special import boxcox
import os
from scipy import stats
import warnings  ### 煩 XD
warnings.filterwarnings("ignore")


def fill_address_na(df, target):
    addr_na_index = df[df[target].isna() == True].index.tolist()
    addr_na_latlon = df[['LATITUDE', 'LONGITUDE']].filter(items=addr_na_index, axis=0)
    addr_na_latlon_index = addr_na_latlon.index.tolist()
    addr_data = df[['LATITUDE', 'LONGITUDE']].drop(index=addr_na_index)

    min_index_dis = {}

    for i in range(len(addr_na_latlon)):
        dis_list = []

        dis_list.append((addr_data.iloc[:, 0] - addr_na_latlon.iloc[i, 0]) ** 2 + (
                    addr_data.iloc[:, 1] - addr_na_latlon.iloc[i, 1]) ** 2)

        ### 轉成dataframe 方便找出index
        dis_list = pd.DataFrame(dis_list[0])

        min_index_dis[addr_na_latlon_index[i]] = dis_list[0].idxmin()
        #### dis_list[0] 0為col名稱

    for na, fill in zip(min_index_dis, min_index_dis.values()):
        df[target][na] = df[target][fill]

    return df


def fill_dis_neg(df, target):  ## 越遠越不好  corr < 0
    df[target] = df[target].fillna(df[target].max() + df[target].std())

    return df


def fill_dis_pos(df, target):  ## 越遠越好  corr > 0

    df[target] = df[target].fillna(0)

    return df


def fill_all_disance(df, target, data_all_na_col):
    for i in data_all_na_col:
        if 'DISTANCE' in i:
            if df[target].corr(df[i]) >= 0:
                fill_dis_pos(df, i)
            else:
                fill_dis_neg(df, i)

    return df


def new_address(df, target, new_col):  ###########################   可以用OTROAD 取代

    new_add = []
    numlist = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    for address_name in df[target]:
        list_count = 0  ### for find the list index

        for word in address_name:
            if word in numlist:
                new_add.append(address_name[:list_count])
                break
            else:
                pass

            list_count += 1

            if len(address_name) == list_count:
                new_add.append(address_name[:list_count])
    df[new_col] = new_add

    return df


def com_ornot(df, target, new_col):
    com_ = []
    for com in df[target].isna():

        if com == True:

            com_.append(0)

        else:
            com_.append(1)

    df[new_col] = com_
    return df


def get_years_build(df, target, new_col):
    years_list = []

    for years in df[target]:
        years = str(years)
        years_list.append(years[:-6])

    df[new_col] = years_list

    return df


def get_years_trad(df, target, new_col):
    years_list = []

    for years in df[target]:
        years = str(years)
        years_list.append(years[:-4])

    df[new_col] = years_list

    return df


def get_month_build(df, target, new_col):
    month_list = []

    for month in df[target]:
        month = str(month)
        month_list.append(month[-6:-4])

    df[new_col] = month_list

    return df


def get_month_trad(df, target, new_col):
    month_list = []
    for month in df[target]:
        month = str(month)
        month_list.append(month[-2:])
    df[new_col] = month_list
    return df


def top_floor(df, floor, total_floor, new_col):
    top = []
    for com in df[floor] == df[total_floor]:
        if com == True:
            top.append(1)
        else:
            top.append(0)
    df[new_col] = top

    return df


def cov_all_distance_log1p(df, data_all_col):
    for i in data_all_col:
        if 'DISTANCE' in i:
            df[i] = np.log1p(df[i])
        else:
            pass
    return df

def cut_distance_into_bins(df,data_all_col) :
    for i in data_all_col :
        if 'DISTANCE' in i :
            df[i+'_category'] = pd.cut((df[i]),[0,1000,100000]) ##### Can change
        else :
            pass
    return df


def cut_lat(df, target, new_col):
    df[new_col] = pd.cut(df[target], 10).astype('str')

    return df


def cut_lon(df, target, new_col):
    df[new_col] = pd.cut(df[target], 5).astype('str')

    return df


def cross_col(df, target1, target2, new_col):
    df[new_col] = df[target1].astype('str') + df[target2].astype('str')

    return df

def prosperity(df,target) :
    try :
        df[target+'_pros'] = (df[target+'_CNT'] / df[target+'_DISTANCE'])
    except :
        df[target+'_pros'] = (df[target+'_CNT'] / df[target+'_DISTANCE_SQL'])
    return df

def get_one_hot (df,nd_onehot_list) :
    df = pd.concat([df] + [pd.get_dummies(df[col], prefix=col) for col in nd_onehot_list], axis=1)
    df = df.drop(nd_onehot_list,axis=1)
    return df


def get_target_encoding_cate (df, test, target, nd_target_list):
    tar = TargetEncoder(smoothing=0.987).fit(df[nd_target_list], df[target])
    df[nd_target_list] = tar.transform(df[nd_target_list])
    test[nd_target_list] = tar.transform(test[nd_target_list])

    return df, test


def get_target_encoding_int(df, test, target, nd_target_list):
    tar = TargetEncoder(smoothing=0.987).fit(df[nd_target_list].astype('str'), df[target])
    df[nd_target_list] = tar.transform(df[nd_target_list])
    test[nd_target_list] = tar.transform(test[nd_target_list])

    return df, test


def get_leaveoneout_encoding_cate(df, test, target, nd_target_list):
    leav = LeaveOneOutEncoder().fit(df[nd_target_list], df[target])
    df[nd_target_list] = leav.transform(df[nd_target_list])
    test[nd_target_list] = leav.transform(test[nd_target_list])

    return df, test


def get_leaveoneout_encoding_int(df, test, target, nd_target_list):
    leav = LeaveOneOutEncoder().fit(df[nd_target_list].astype('str'), df[target])
    df[nd_target_list] = leav.transform(df[nd_target_list])
    test[nd_target_list] = leav.transform(test[nd_target_list])

    return df, test


