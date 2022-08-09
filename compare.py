import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re

import os
import warnings
import math

warnings.filterwarnings('ignore')

from model import Model
from Core.compare import *

# Parámetros
n_celdas = 74
col_fluido=int((n_celdas+3)/7)
col_celda=int(col_fluido-1)

# Individuo
'''
:param cdrag_tree:     inputs(5) (reynolds, separation, index, normalizedArea, normalizedDensity) 
:param ffriction_tree: inputs(5) (reynolds, separation, index, normalizedVelocity, normalizedDensity)
:param nnusselt_tree:  inputs(4) (reynolds, prandtl, separation, index)
'''
# Individuo
ind = [ lambda Re, S: 3.901/(S**0.365 * Re**0.096), \
        lambda Re, S: 25.404/((S**1.154)*(Re**0.254)), \
        lambda Re, Pr, S: (0.383 * S + 0.078)/S * (Re**0.64) * Pr]

        
# Data ANSYS 
df_ansys = get_dataFrame('DATA/ANSYS/entradas_1_'+str(n_celdas)+'_celdas.csv')
df_ansys = get_data_simple(df_ansys)
print(df_ansys.head())

# Data Modelo Fenomenológico
df_mf = df_ansys[['Current'	,'K','Flujo','t_viento','Diametro']]
for i in range(col_fluido):
  num_string = str(i+1)
  if len(num_string)<2:
    num_string = '0' + num_string
  df_mf['V' + num_string] = None
  df_mf['P' + num_string] = None
  df_mf['TF' + num_string] = None
  if i != (col_fluido-1):
    df_mf['TC' + num_string] = None

for index, row in df_mf.iterrows():
  mdl = Model(current=row['Current'], cellDiamater=row['Diametro'], separation=row['K'], initFlow=row['Flujo'], initTemperature=row['t_viento'], col_fluido=col_fluido, col_celda=col_celda, n_fluido=4, n_celda=3, nmax=10)
  mdl.load_individual(*ind)
  mdl.start()
  results_dict = mdl.evolve()
  V_array = results_dict['vf']
  P_array = results_dict['pf']
  T_array = results_dict['tf']
  TC_array = results_dict['tc']
  for i in range(col_fluido):
    num_string = str(i+1)
    if len(num_string)<2:
      num_string = '0' + num_string
    df_mf.at[index,'V' + num_string] = V_array[i]
    df_mf.at[index,'P' + num_string] = P_array[i]
    df_mf.at[index,'TF' + num_string] = T_array[i]
    if i != (col_fluido-1):
      df_mf.at[index,'TC' + num_string] = TC_array[i]

print(df_mf.head())

# Get curves
dataset_array = [df_ansys, df_mf]
names = ['ANSYS', 'Modelo fenomenológio']
for input in ['Current'	,'K',	'Flujo',	't_viento',	'Diametro']:
    compare(input, dataset_array, names)