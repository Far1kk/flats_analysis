# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 18:18:58 2023

@author: bogdan
"""

import os
os.chdir("C:/Users/bogdan/work")
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# Читаем эксель и создаем датафрейм
pth_a = './data/flats_moscow.xlsx' # Путь относительно рабочего каталога
FLATSA = pd.read_excel(pth_a)
FA = FLATSA.copy() # Копируем датафрейм для работы

# Перекодируем значения переменных
FA['walk'].replace({1:"пешком", 0:"на транспорте"}, inplace=True)
FA['brick'].replace({1:"кирпич или монолит", 0:"другой"}, inplace=True)
FA['floor'].replace({1:"другой", 0:"1 или последний"}, inplace=True)

# Преобразуем значения в определенные шкалы
FA = FA.astype({'price':np.float64, 'totsp':np.float64, 
                'livesp':np.float64, 'kitsp':np.float64,
                'dist':np.float64, 'metrdist':np.float64,
                'walk':'category', 'brick':'category',
                'floor':'category'})

# Выбираем количественные значения для формул Пирсона и Спирмана 
FAF = FA.select_dtypes(include='float')
# Создаем краткую стаистику
FA_STATS = FAF.describe()
FA_iqr = FAF.quantile(q=0.75) - FAF.quantile(q=0.25) # Межквартильный размах
FA_sk = FAF.skew() # Коэффицент ассиметрии
FA_kur = FAF.kurtosis() # Эксцесс
# Создаем pandas.DataFrame из новых статистик
W = pd.DataFrame([FA_sk, FA_kur, FA_iqr], index=['skewness', 'kurtosis', 'IQR'])
# Объединяем FA_STAT и W
FA_STATS = pd.concat([FA_STATS, W])

# Создаем эксель и записываем данные
with pd.ExcelWriter('./output/FLATS_STAT.xlsx', engine='openpyxl') as wrt:
    FA_STATS.to_excel(wrt, sheet_name='stat')

"""
Гистограммы количественных переменных
"""
plt.hist(FA['price'], color = 'blue', edgecolor = 'black', bins = 'fd', density=True)
plt.title('Распределение цен для квартир в Москве')
plt.xlabel('Цена в 1000$')
plt.ylabel('Частота')

# Создаем гистограммы для общей площади квартиры
plt.hist(FA['totsp'], color = 'blue', edgecolor = 'black', bins = 'fd', density=True)
plt.title('Распределение площади')
plt.xlabel('Площадь в кв. м.')
plt.ylabel('Квартиры')

# Создаем гистограммы для жилой площади квартиры
plt.hist(FA['livesp'], color = 'blue', edgecolor = 'black', bins = 'fd', density=True)
plt.title('Распределение жилой площади')
plt.xlabel('Площадь в кв. м.')
plt.ylabel('Квартиры')

# Создаем гистограммы для площади кухни
plt.hist(FA['kitsp'], color = 'blue', edgecolor = 'black', bins = 'sturges', density=True)
plt.title('Распределение площади кухни')
plt.xlabel('Площадь в кв. м.')
plt.ylabel('Квартиры')

# Создаем гистограммы для расстояния от центра
plt.hist(FA['dist'], color = 'blue', edgecolor = 'black', bins = 'sturges', density=True)
plt.title('Распределение расстояния до центра города')
plt.xlabel('Расстояние в км.')
plt.ylabel('Квартиры')

# Создаем гистограммы для расстояния до метро
plt.hist(FA['metrdist'], color = 'blue', edgecolor = 'black', bins = 'sturges', density=True)
plt.title('Распределение времени пути до метро')
plt.xlabel('Время в минутах')
plt.ylabel('Квартиры')

#Экспорт в pdf----------------
tit = {'price': 'Цена квартиры', 'totsp': 'Общая площадь квартиры', 
       'livesp':'Жилая площадь квартиры', 'kitsp': 'Площадь кухни',
       'dist': 'Расстояние до центра города', 'metrdist': 'Время пути до метро'}
dfn = FAF.select_dtypes(include='float64')
plt.figure(figsize=(15, 9)) # Создаем лист нужного размера в дюймах
# Добавляем пространство между рисунками, чтобы не перекрывались
plt.subplots_adjust(wspace=0.5, hspace=0.5) 
nplt = 1
k = 1
for s in dfn.columns:
    a = 'fd' if s in ['price', 'totsp', 'livesp'] else 'sturges'
    ax = plt.subplot(3, 1, nplt)
    dfn.hist(column=s, ax=ax, bins= a, density=True, grid=True,legend=False,
             color=None)
    ax.set_title(tit[s], fontdict={'fontsize':15, 'color':'blue'}, loc='left')
    nplt += 1
    if nplt > 3:
        nplt = 1
        plt.savefig(f'./graphics/flat_stat{k}.pdf', format='pdf')
        k+=1
        nrow = dfn.shape[1]
        plt.figure(figsize=(15, 9)) # Создаем лист нужного размера в дюймах
        # Добавляем пространство между рисунками, чтобы не перекрывались
        plt.subplots_adjust(wspace=0.5, hspace=0.5) 
plt.savefig(f'./graphics/flat_stat{k}.pdf', format='pdf')
plt.show()

"""
Круговые диграммы номинальных переменных (столбчатые в самом низу на ремонте)
"""
dfnc = FA.select_dtypes(include=["category"])
ftb = pd.crosstab(dfnc['walk'], 'walk')
ftb.plot.pie(title='Способ добраться до метро', subplots=True, grid=True,
             legend=False, colors=['blue', 'green'], figsize=(5, 5))


ftb = pd.crosstab(dfnc['brick'], 'brick')
ftb.plot.pie(title='Материал здания', subplots=True, grid=True,
             legend=False, colors=['blue', 'green'], figsize=(5, 5))

ftb = pd.crosstab(dfnc['floor'], 'floor')
ftb.plot.pie(title='Этаж', subplots=True, grid=True,
             legend=False, colors=['blue', 'green'], figsize=(5, 5))

# Экспорт в pdf-------------------
dfnc = FA.select_dtypes(include=["category"])
plt.figure(figsize=(15, 9)) 
plt.subplots_adjust(wspace=0.5, hspace=0.5)
nplt = 1
tit = {'walk': "Способ добраться до метро", "brick":"Материал здания", "floor": "Этаж"}
for s in dfnc.columns:
    ax = plt.subplot(3, 1, nplt)
    ftb = pd.crosstab(dfnc[s], s)
    ftb.index.name = 'Категории'
#    ftb.columns.name = s 
    ftb.plot.pie(subplots=True, table=True, ax=ax, grid=True, legend=False, 
                 colors=['blue', 'green'])
    ax.set_title(tit[s], fontdict={'fontsize':15, 'color':'blue'}, loc='left')
    nplt += 1
plt.savefig('./graphics/flat_num.pdf', format='pdf')
plt.show()

"""
Графический анализ
Анализ связи между количественной целевой переменной и 
количественными объясняющими переменными
"""
dfn = FA.select_dtypes(include='float64')
dfn.plot.scatter('totsp', 'price', title='Связь цены с общей площадью')
plt.xlabel('Общая площадь в м. кв.')
plt.ylabel('Цена, 1000$')

dfn.plot.scatter('livesp', 'price', title='Связь цены с жилой площадью')
plt.xlabel('Жилая площадь в м. кв.')
plt.ylabel('Цена, 1000$')

dfn.plot.scatter('kitsp', 'price', title='Связь цены с площадью кухни')
plt.xlabel('Площадь кухни в м. кв.')
plt.ylabel('Цена, 1000$')

dfn.plot.scatter('dist', 'price', title='Связь цены с отдаленностью от центра')
plt.xlabel('Растояние от центра в км.')
plt.ylabel('Цена, 1000$')

dfn.plot.scatter('metrdist', 'price', title='Связь цены с растоянием до метро')
plt.xlabel('Растояние до метро в минутах')
plt.ylabel('Цена, 1000$')


""" НА РЕМОНТЕ ЗАПИСЬ В ПДФ ГРАФИКОВ ЦЕЛЕВОЙ С КОЛИЧЕСТВЕННОЙ И НЕ ЦЕЛЕВОЙ
nrow = dfn.shape[1] - 1 # Учитываем, что одна переменная целевая - ось 'Y'
fig, ax_lst = plt.subplots(nrow, 1)
fig.figsize=(15, 9) # Дюймы
nplt = -1
for s in dfn.columns[:-1]: # Последняя переменная - целевая ('Y')
    nplt += 1
    dfn.plot.scatter(s, 'price', ax=ax_lst[nplt])
    ax_lst[nplt].grid(visible=True)
    ax_lst[nplt].set_title(f'Связь цены с {s}')
    break
fig.subplots_adjust(wspace=0.5, hspace=1.0)
fig.suptitle(f'Связь цены с {list(dfn.columns[:-1])}')
plt.savefig('./graphics/cars_scat.pdf', format='pdf')
plt.show()
"""

"""
Определяем коэффицент корреляции Пиросна И Спирмана
"""
from scipy.stats import pearsonr
from scipy.stats import spearmanr

#Значения оценок коэффициента корреляции Пирсона
C_P = pd.DataFrame([], index=FAF.columns, columns=FAF.columns)
#Значимость оценок корреляции Пирсона
P_P = pd.DataFrame([], index=FAF.columns, columns=FAF.columns)
#Значения оценок коэффициента корреляции Спирмана
C_S = pd.DataFrame([], index=FAF.columns, columns=FAF.columns)
#Значимость оценок корреляции Спирмана
P_S = pd.DataFrame([], index=FAF.columns, columns=FAF.columns)

for x in FAF.columns:
    for y in FAF.columns:
        C_P.loc[x, y], P_P.loc[x, y] = pearsonr(FAF[x], FAF[y])
        C_S.loc[x, y], P_S.loc[x, y] = spearmanr(FAF[x], FAF[y])

# Записываем в эксель полученные данные
with pd.ExcelWriter('./output/FLATS_STAT.xlsx', engine='openpyxl', mode='a', 
                    if_sheet_exists='overlay') as wrt:
    C_P.to_excel(wrt, sheet_name='Pearson')
    dr = C_P.shape[0] + 2
    P_P.to_excel(wrt, startrow=dr, sheet_name='Pearson')
    C_S.to_excel(wrt, sheet_name='Spirmen')
    dr = C_S.shape[0] + 2
    P_S.to_excel(wrt, startrow=dr, sheet_name='Spirmen')
    
"""
Графический анализ
Анализ связи между количественной и качественной переменной
В столбце не более трех графиков
"""
dfn = FA.copy()
cols = dfn.select_dtypes(include='category').columns

dfn.boxplot(column='price', by='walk', notch=False, 
            bootstrap=500, showmeans=True, color='blue', showfliers=False)
plt.title('')
plt.xlabel('Способ добраться до метро')
plt.ylabel('Цена, 1000$')

dfn.boxplot(column='price', by='brick', notch=False, 
            bootstrap=500, showmeans=True, color='blue', showfliers=False)
plt.title('')
plt.xlabel('Материал здания')
plt.ylabel('Цена, 1000$')

dfn.boxplot(column='price', by='floor', notch=False, 
            bootstrap=500, showmeans=True, color='blue', showfliers=False)
plt.title('')
plt.xlabel('Этаж')
plt.ylabel('Цена, 1000$')

# Записываем в pdf------------------
dfn = FA.copy()
cols = dfn.select_dtypes(include='category').columns
for s in cols:
    dfn.boxplot(column='price', by=s, grid=True, notch=False, 
                bootstrap=50, showmeans=True, color=None, showfliers=False)
    plt.savefig(f'./graphics/box_plot_{s}.pdf', format='pdf')
    plt.show()

"""
Анализируем колич. премен. с качественными
"""
from scipy.stats import kruskal

#Создаем подвыборки
sel_yes = FA['brick']=='кирпич или монолит'
x_1 = FA.loc[sel_yes, 'price']
sel_no = FA['brick']=='другой'
x_2 = FA.loc[sel_no, 'price']

sel_yes = FA['walk']=='пешком'
y_1 = FA.loc[sel_yes, 'price']
sel_no = FA['walk']=='на транспорте'
y_2 = FA.loc[sel_no, 'price']

sel_yes = FA['floor']=='1 или последний'
z_1 = FA.loc[sel_yes, 'price']
sel_no = FA['floor']=='другой'
z_2 = FA.loc[sel_no, 'price']

Price_brick = kruskal(x_1, x_2)
Price_walk = kruskal(y_1, y_2)
Price_floor = kruskal(z_1, z_2)

with open('./output/FLATS_STAT.txt', 'w') as fln:
    print('Критерий Крускала-Уоллиса для переменных \'price\' и \'brick\'',
          file=fln)
    print(Price_brick, file=fln)
    print('Критерий Крускала-Уоллиса для переменных \'price\' и \'walk\'',
          file=fln)
    print(Price_walk, file=fln)
    print('Критерий Крускала-Уоллиса для переменных \'price\' и \'floor\'',
          file=fln)
    print(Price_floor, file=fln)

"""
Графический анализ
Анализ связи между количественными независимыми переменнами
"""
dfn = FA.select_dtypes(include='float64')
dfn.plot.scatter('totsp', 'livesp', title='Связь общей с жилой площадьми')
plt.xlabel('Общая площадь в м. кв.')
plt.ylabel('Жилая площадь в м. кв.')

dfn.plot.scatter('totsp', 'dist', title='Связь общей площади с отдаленности до центра')
plt.xlabel('Общая площадь в м. кв.')
plt.ylabel('Расстояние до метро в км.')

dfn.plot.scatter('livesp', 'kitsp', title='Связь жилой площади с площадью кухни')
plt.xlabel('Жилая площадь в м. кв.')
plt.ylabel('Площадь кухни в м. кв.')

"""
Анализ связи между качественными независимыми переменнами
"""
import statsmodels.api as sm

FAC = FA.select_dtypes('category')
crtx = pd.crosstab(FAC['brick'], FAC['floor'], margins=True)
crtx.columns.name = 'brick'
crtx.index.name = 'brick-floor'
tabx = sm.stats.Table(crtx)

with pd.ExcelWriter('./output/FLATS_STAT.xlsx', engine='openpyxl',
                    if_sheet_exists='overlay', mode='a') as wrt:
    tabx.table_orig.to_excel(wrt, sheet_name="brick-floor")
    dr = tabx.table_orig.shape[0] + 2
    tabx.fittedvalues.to_excel(wrt, sheet_name="brick-floor", startrow=dr)

resx = tabx.test_nominal_association()
nr = tabx.table_orig.shape[0]
nc = tabx.table_orig.shape[1]
N = tabx.table_orig.iloc[nr-1, nc-1]
hisq = resx.statistic
CrV = np.sqrt(hisq/(N*min((nr-1, nc-1))))
with open('./output/FLATS_STAT.txt','a') as fln:
    print('Критерий HI^2 для переменных "brick" и "floor"', file=fln)
    print(resx, file=fln)
    print('Статистика Cramer V для переменных "brick" и "floor"', file=fln)
    print(CrV, file=fln)
    
"""
Анализ связи между независимой колличественной независимой качественной переменой
"""
dfn = FA.copy()
cols = dfn.select_dtypes(include='category').columns

dfn.boxplot(column='totsp', by='brick', notch=False, 
            bootstrap=50, showmeans=True, color='blue', showfliers=False)
plt.title('')
plt.xlabel('Материал')
plt.ylabel('Общая площадь в м. кв.')

dfn.boxplot(column='totsp', by='floor', notch=False, 
            bootstrap=50, showmeans=True, color='blue', showfliers=False)
plt.title('')
plt.xlabel('Этаж')
plt.ylabel('Общая площадь в м. кв.')

dfn.boxplot(column='metrdist', by='walk', notch=False, 
            bootstrap=50, showmeans=True, color='blue', showfliers=False)
plt.title('')
plt.xlabel('Способ добраться до метро')
plt.ylabel('Время пути до метро в минутах')

sel_yes = FA['brick']=='кирпич или монолит'
x_1 = FA.loc[sel_yes, 'totsp']
sel_no = FA['brick']=='другой'
x_2 = FA.loc[sel_no, 'totsp']

sel_yes = FA['floor']=='1 или последний'
y_1 = FA.loc[sel_yes, 'totsp']
sel_no = FA['floor']=='другой'
y_2 = FA.loc[sel_no, 'totsp']

sel_yes = FA['walk']=='пешком'
z_1 = FA.loc[sel_yes, 'metrdist']
sel_no = FA['walk']=='на транспорте'
z_2 = FA.loc[sel_no, 'metrdist']

Totsp_brick = kruskal(x_1, x_2)
Totsp_floor = kruskal(y_1, y_2)
Metrdist_walk = kruskal(z_1, z_2)

with open('./output/FLATS_STAT.txt', 'a') as fln:
    print('Критерий Крускала-Уоллиса для переменных \'totsp\' и \'brick\'',
          file=fln)
    print(Totsp_brick, file=fln)
    print('Критерий Крускала-Уоллиса для переменных \'totsp\' и \'floor\'',
          file=fln)
    print(Totsp_floor, file=fln)
    print('Критерий Крускала-Уоллиса для переменных \'metrdist\' и \'walk\'',
          file=fln)
    print(Metrdist_walk, file=fln)
    
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

"""
Визуальный анализ корреляции между 3 количественными переменными
"""
x = FA['price']
y = FA['dist']
z = FA['metrdist']

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
plot = ax.scatter(x, y, z,
           linewidths=1, alpha=.7,
           edgecolor='k',
           s = 200,
           c=z)
plt.title("3D корреляция цены-удаленности\n от центра-время до метро")
ax.set_xlabel('Цена в 1000$')
ax.set_ylabel('Раст. до центра в км.')
ax.set_zlabel('Время до метро в мин.')
fig.colorbar(plot, ax = ax, shrink = 0.5, aspect = 10)
plt.show()

fig = plt.figure(figsize=(6, 6))
plot = plt.scatter(x, y,
           linewidths=1, alpha=.7,
           edgecolor='k',
           s = 200,
           c=z)
plt.title("2D корреляция цены-удаленности\n от центра-время до метро")
plt.xlabel('Цена в 1000$')
plt.ylabel('Раст. до центра в км.')
plt.show()



