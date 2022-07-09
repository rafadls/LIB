import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re

import os
import warnings
import math

def get_df_to_plot(df,var1,var2):
    col_inputs = ['Current', 'K', 'Flujo', 't_viento', 'Diametro']
    df_aux = pd.concat([df[col_inputs],df.filter(regex=(var2 + "+\d")) ],axis=1)
    col_inputs.remove(var1)
    dir_base_values = dict(df.groupby(by=col_inputs).size().reset_index().rename(columns={0:'records'}).sort_values(by=['records'],ascending=False).reset_index(drop=True).iloc[0,:])
    for col in col_inputs:
        df_aux = df_aux[ df_aux[col]==dir_base_values[col]]
    df_aux.drop(columns=col_inputs, inplace=True)
    df_aux.reset_index(drop=True,inplace=True)
    return df_aux

def get_dataFrame(path_to_file):
    data = pd.read_csv(path_to_file, header=6)
    colCoded = data.columns.tolist()
    nameCode = pd.read_csv(path_to_file, header=3,nrows=1)
    colName = [[col.replace(d+' - ','') for i,col in enumerate(nameCode) if re.search(d, col)] for d in colCoded]
    colName = [temp[0] if temp else 'Name' for temp in colName]
    colName = [temp if (' ' not in temp) else temp[:temp.index(' ')] for temp in colName]
    data.rename(columns=dict(zip(colCoded, colName)), inplace=True)
    data.dropna(inplace=True)
    data.drop_duplicates(inplace=True)
    return data

def get_data_simple(df):
    df_output = df[['Current', 'K', 'Flujo', 't_viento', 'Diametro']]
    for string in ['V','P','TF','TC']:
        if string[0]=='T':
            df_aux = df.filter(regex=(string + "+\d")) - 273.15
        else:
            df_aux = df.filter(regex=(string + "+\d"))
        df_output = pd.concat([df_output,df_aux], axis=1)
    df_output = df_output[ df_output['Flujo'] > 10]
    df_output.reset_index(drop=True,inplace=True)
    return df_output

def compare(input, dataset_array, names):
    dict_var2spanish = {
        'Current': 'la corriente que pasa a través de las celdas (I)', 'K': 'el factor de separación de las celdas (S)', 'Flujo': r'el flujo inicial de fluido ($F_{in}$)', 't_viento': r'la temperatura inicial de fluido ($T_{in}$)', 'Diametro': 'el diametro de celda (D)'
    }
    dict_output2label = {
        'V': 'Velocidad [m/s]', 'P':'Presión [Pa]', 'TF': 'Temperatura [°C]','TC': 'Temperatura [°C]'
    }
    dict_output2title = {
        'V': 'Velocidad de fluido', 'P':'Presión de fluido', 'TF': 'Temperatura de fluido','TC': 'Temperatura de celda'
    }

    output_array = ['V', 'P', 'TF','TC']

    fig = plt.figure(constrained_layout=True, figsize=(len(dataset_array)*6, len(output_array)*4))
    fig.suptitle('Análisis de curvas físicas en función de ' + dict_var2spanish[input], fontsize='xx-large')
    subfigs = fig.subfigures(len(output_array), 1)

    for i in range(len(output_array)):
        subfigs[i].suptitle(dict_output2title[output_array[i]] + ' en función de ' + dict_var2spanish[input], fontsize='x-large')
        ax = subfigs[i].subplots(1, len(dataset_array), sharey=True)
        for j in range(len(dataset_array)):
            df_i_vs_o = get_df_to_plot(dataset_array[j],input,output_array[i])
            columns_to_plot = list(df_i_vs_o.columns)
            columns_to_plot.remove(input)
            df_i_vs_o.plot(x=input, y=columns_to_plot, ax=ax[j])
            ax[j].set_title(names[j])
            ax[j].set_xlabel(input)
            ax[j].set_ylabel(dict_output2label[output_array[i]])
    plt.savefig('Compare/' + input + '.png')


