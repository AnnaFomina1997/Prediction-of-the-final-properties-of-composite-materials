# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 21:46:02 2023

@author: Saimon
"""

import numpy as np
import pickle
import streamlit as st
import pandas as pd

loaded_model = pickle.load(open(
    'model_regressor.sav', 'rb'))
min_max_scaler = pickle.load(open(
    'transformer.sav', 'rb'))

def y_prediction_with_normalization(input_data):
    cols = ['Соотношение матрица-наполнитель', 'Плотность, кг/м3',
           'модуль упругости, ГПа', 'Количество отвердителя, м.%',
           'Содержание эпоксидных групп,%_2', 'Температура вспышки, С_2',
           'Поверхностная плотность, г/м2', 
           'Модуль упругости при растяжении, ГПа',
           'Прочность при растяжении, МПа', 'Потребление смолы, г/м2',
           'Угол нашивки, град', 'Шаг нашивки', 'Плотность нашивки']
    input_359 = np.array(input_data).reshape(1,-1)
    df = pd.DataFrame(input_359, columns=cols)
    
    df_norm = pd.DataFrame(min_max_scaler.transform(df),columns=cols)
    
    y_pred = loaded_model.predict(df_norm.
                                  drop('Соотношение матрица-наполнитель', 
                                       axis=1))
    
    df_norm['Соотношение матрица-наполнитель'] = y_pred
    new_df = pd.DataFrame(min_max_scaler.inverse_transform(df_norm), 
                          columns = cols)
    return new_df['Соотношение матрица-наполнитель'][0]

def main():
    st.title('Предсказание параметра: соотношение матрица-наполнитель. Web app')
    matrix = 0
    density = st.text_input('Плотность, кг/м3')
    elasticity = st.text_input('Модуль упругости, ГПа')
    hardener = st.text_input('Количество отвердителя, м.%')
    epoxy = st.text_input('Содержание эпоксидных групп,%_2')
    temperature = st.text_input('Температура вспышки, С_2')
    surface_density = st.text_input('Поверхностная плотность, г/м2')
    elastic_modulus = st.text_input('Модуль упругости при растяжении, ГПа')
    toughness = st.text_input('Прочность при растяжении, МПа')
    resin_consumption = st.text_input('Потребление смолы, г/м2')
    patch_angle = st.text_input('Угол нашивки, град')
    patch_pitch = st.text_input('Шаг нашивки')
    patch_density = st.text_input('Плотность нашивки')
    
    pred = ''
    
    if st.button('Predict'):
        pred = y_prediction_with_normalization([matrix, 
                                                density,
                                                elasticity,
                                                hardener,
                                                epoxy,
                                                temperature,
                                                surface_density,
                                                elastic_modulus,
                                                toughness,
                                                resin_consumption,
                                                patch_angle,
                                                patch_pitch,
                                                patch_density])
    
    st.success(f'Соотношение матрица-наполнитель = {pred}')

if __name__ == '__main__':
    main()
    
