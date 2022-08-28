# LIBRERÍAS

from nltk import word_tokenize
from stop_words import get_stop_words
from bs4 import BeautifulSoup
import spacy
import unidecode
import os, os.path, sys
import glob
import random
import re
import string
from fastapi import FastAPI
import json, uvicorn
import pandas as pd
pd.options.display.max_colwidth = 230
import es_core_news_md
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.feature_extraction.text import TfidfTransformer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
import os, os.path, sys
import glob
import datetime
import dataframe_image as dfi
from fastapi.responses import FileResponse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from datetime import datetime
import math
from numpy.lib.function_base import average
import time


# FUNCIÓN PARA PREPROCESAR EL TEXTO DE LA FALLA INGRESADO EN EL CHATBOT

nlp = es_core_news_md.load()
# Retirando el no de las palabras de parada
deselect_stop_words = ['no']
for w in deselect_stop_words:
    nlp.vocab[w].is_stop = False

def strip_html_tags(text):
    """remove html tags from text"""
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text(separator=" ")
    return stripped_text

def remove_whitespace(text):
    """remove extra whitespaces from text"""
    text = text.strip()
    return " ".join(text.split())

def remove_accented_chars(text):
    """remove accented characters from text, e.g. caf"""
    text = unidecode.unidecode(text)
    return text

def remove_letters(text):
    """remove accented characters from text, e.g. caf"""
    text = [w for w in text.split() if len(w) > 1]
    return( " ".join(text))  

def text_preprocessing(text, accented_chars=True, extra_whitespace=True, 
                       lemmatization=True, lowercase=True, 
                       remove_html=True, special_chars=True, 
                       stop_words=True, remove_letter=True):
    global clean_text
    """preprocess text with default option set to true for all steps"""
    if remove_html == True: #Removiendo etiquetas html
        text = strip_html_tags(text)
    if remove_letter == True: #Removiendo palabras con longitud 1
        text = remove_letters(text)
    if extra_whitespace == True: #Removiendo espacios extra
        text = remove_whitespace(text)
    if accented_chars == True: #Removiendo acentos
        text = remove_accented_chars(text)
    if lowercase == True: #Convirtiendo todos los caracteres al mismo nivel
        text = text.lower()

    doc = nlp(text) #Tokenizando el texto

    clean_text = []

    for token in doc:
        flag = True
        edit = token.text
        #Removiendo stop words
        if stop_words == True and token.is_stop and token.pos_ != 'NUM': 
            flag = False
        # removiendo caracteres especiales
        if special_chars == True and token.pos_ == 'SYM' and flag == True: 
            flag = False
        # Lematizando (raiz de la palabra)
        elif lemmatization == True and token.lemma_ != "-PRON-" and flag == True:
            edit = token.lemma_
        # Generando el nuevo texto preprocesado 
        if edit != "" and flag == True:
            clean_text.append(edit)
    return clean_text

# CARGUE DE CONJUNTO DE DATOS 

df = pd.read_excel('Datos_Corregidos.xlsx')

# MODELO DE CLASIFICACIÓN

x_train, x_test, y_train, y_test = train_test_split(df['Reporte_Procesado'], df['ATA'],
                                                                        test_size=0.2, random_state=42)

count_vectorizer = CountVectorizer()
count_vect = CountVectorizer()
x_train_counts = count_vect.fit_transform(x_train)
tfidf_transformer = TfidfTransformer()
x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

clf = ComplementNB()
clf.fit(x_train_tfidf, y_train)

# EMBEDDING CON MODELO PREENTRENADO DE SentenceTransformer PARA BÚSQUEDA DE TEXTOS SIMILARES

df2 = df.dropna(subset=['Falla'])
#df2 = df2[:10000]
fallas=list(df2['Falla'])
#fallas = fallas[:10000]
print('Inicia proceso de embedding')
fallas_embeddings = model.encode(fallas)
print('Fin proceso de embedding')

# FASTAPI PARA EXPONER APLICACIONES REST

app = FastAPI()

## Método para llamar al modelo de clasificación y realizar la predicción con base en el texto ingresado

@app.get('/clasificar/{descripcion}')
async def clasificar(descripcion):
    text = text_preprocessing(descripcion, accented_chars=True, extra_whitespace=True, lemmatization=True, lowercase=True, 
	                      remove_html=True, special_chars=True, stop_words=True, remove_letter=True)
    prediction = clf.predict(count_vect.transform(text))
    prediction = prediction[0]
    return str(prediction)

# Método para encontrar un valor en el conjunto de datos

@app.get('/encontrarValor/{valor}')
async def encontrarValor(valor): 
    return str(valor in df2.values)

# Método para generar una imagen con las fallas similares teniendo en cuenta una falla ingresada y filtrando por flota

@app.get('/fallassimilares/{falla}/flota/{flota}')
def fallas_similares(falla, flota):
    global name, listaReporteF
    name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=20))
    global falla_emb
    falla_emb = model.encode(falla)
    flota = flota.upper()
    distancia=cosine_similarity([falla_emb],fallas_embeddings[:])[0].tolist()
    dfCopy = df2.copy()
    dfCopy['Distancia']=distancia
    dfCopy = dfCopy[dfCopy.Flota.isin([flota])]
    df3 = dfCopy[dfCopy.Distancia>=.2]
    df3.sort_values('Distancia',ascending=False, inplace=True)
    df3.reset_index(inplace=True, drop=True)
    resultFlota = df3[['Reporte','Falla']].head(5)
    resultFlota = resultFlota.sort_values('Reporte')
    resultFlota.reset_index(inplace=True, drop=True)
    dfi.export(resultFlota, name+'.jpeg')
    listaReporteF = list(resultFlota['Reporte'])
    return FileResponse(name+'.jpeg')

# Método para generar una imagen con las fallas similares teniendo en cuenta una falla ingresada y filtrando por matrícula

@app.get('/fallassimilaresMat/{falla}/matricula/{matricula}')
def fallas_similaresMat(falla, matricula):
    global name, listaReporte
    name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=20))
    global falla_emb
    falla_emb = model.encode(falla)
    matricula = matricula.upper()
    distancia=cosine_similarity([falla_emb],fallas_embeddings[:])[0].tolist()
    dfCopy = df2.copy()
    dfCopy['Distancia']=distancia
    dfCopy = dfCopy[dfCopy.Matricula.isin([matricula])]
    df3 = dfCopy[dfCopy.Distancia>=.2]
    df3.sort_values('Distancia',ascending=False, inplace=True)
    df3.reset_index(inplace=True, drop=True)
    result = df3[['Reporte','Falla']].head(5)
    result = result.sort_values('Reporte')
    result.reset_index(inplace=True, drop=True)
    dfi.export(result, name+'.jpeg')
    listaReporte = list(result['Reporte'])
    return FileResponse(name+'.jpeg')

# Método para generar una imagen con las soluciones de las fallas similares encontradas por flota

@app.get('/fallassimilaresMat_Sol')
def fallas_similaresMat_Sol():
    global nameSol
    nameSol = ''.join(random.choices(string.ascii_uppercase + string.digits, k=20))
    resultSol = df2[df2.Reporte.isin(listaReporte)]
    resultSol = resultSol[['Reporte','Solucion']]
    resultSol = resultSol.sort_values('Reporte')
    resultSol.reset_index(inplace=True, drop=True)
    dfi.export(resultSol, nameSol+'.jpeg')
    return FileResponse(nameSol+'.jpeg')

# Método para generar una imagen con las soluciones de las fallas similares encontradas por matrícula

@app.get('/fallassimilaresFlota_Sol')
def fallas_similaresFlota_Sol():
    global nameSol2
    nameSol2 = ''.join(random.choices(string.ascii_uppercase + string.digits, k=20))
    resultSolFlota = df2[df2.Reporte.isin(listaReporteF)]
    resultSolFlota = resultSolFlota[['Reporte','Solucion']]
    resultSolFlota = resultSolFlota.sort_values('Reporte')
    resultSolFlota.reset_index(inplace=True, drop=True)
    dfi.export(resultSolFlota, nameSol2+'.jpeg')
    return FileResponse(nameSol2+'.jpeg')

# Método para generar una imagen con los sistemas críticos en un período de tiempo por matrícula

@app.get('/detsistemasMat/{matricula}/fechaInicial/{fechaInicial}/fechaFinal/{fechaFinal}')
def sist_crit_matricula(matricula, fechaInicial, fechaFinal):
    name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=20))
    fechaI = datetime.strptime(fechaInicial, '%d-%m-%Y')
    fechaF = datetime.strptime(fechaFinal, '%d-%m-%Y')
    matricula = matricula.upper()
    dfMat=df[df['Matricula']==matricula].copy()
    dfMat=dfMat[dfMat['Fecha_Creacion_ Aviso']>=fechaI].copy()
    dfMat=dfMat[dfMat['Fecha_Creacion_ Aviso']<=fechaF].copy()
    tabla=dfMat.groupby("ATA").size().reset_index(name="Conteo").sort_values("Conteo", ascending=False)
    tabla["Porcentaje"]=100*tabla["Conteo"]/dfMat.shape[0]
    tabla["Acumulado"]=np.cumsum(tabla["Porcentaje"])
    #return dfMat['Matricula'].head(20)
    color1 = 'steelblue'
    color2 = 'red'
    line_size = 4
    fig = plt.figure(figsize=(8, 12))
    fig, ax = plt.subplots()
    ax.bar(tabla.ATA, tabla['Conteo'], color=color1)
    ax.xaxis.label.set_size(12)
    for x, y in zip(tabla.ATA, tabla['Conteo']):
        plt.text(x, y-0.1, str(y), ha='center', va='bottom', fontsize=10.5)
    ax2 = ax.twinx()
    ax2.plot(tabla.ATA, tabla['Acumulado'], color=color2, marker="D", ms=line_size)
    ax2.yaxis.set_major_formatter(PercentFormatter())
    ax2.xaxis.label.set_size(12)
    ax.tick_params(axis='y', colors=color1)
    ax2.tick_params(axis='y', colors=color2)
    plt.title('Diagrama de pareto sistemas críticos ', size = 14)
    ax.set(xlabel = 'Sistemas')
    ax.set(ylabel = 'Cant. Fallas')
    ax2.set(ylabel = '% Acumulado')
    plt.savefig(name+'.jpeg')
    return FileResponse(name+'.jpeg')

# Método para entregar datos estadísticos básicos de valor para la operación en un período de tiempo por matrícula

@app.get('/estadisticasFallasMat/{matricula}/fechaInicial/{fechaInicial}/fechaFinal/{fechaFinal}')
async def estadisticasFallasMat(matricula, fechaInicial, fechaFinal):
    fechaI = datetime.strptime(fechaInicial, '%d-%m-%Y')
    fechaF = datetime.strptime(fechaFinal, '%d-%m-%Y')
    matricula = matricula.upper()
    dfEst=df[df['Matricula']==matricula].copy()
    dfEst=dfEst[dfEst['Fecha_Creacion_ Aviso']>=fechaI].copy()
    dfEst=dfEst[dfEst['Fecha_Creacion_ Aviso']<=fechaF].copy()
    Cant_Fallas=len(dfEst)
    Horas_voladas=round(sum(dfEst['TBF']),1)
    MTBF=round(Horas_voladas/Cant_Fallas,1)
    Confiab=round(math.exp(-3.2/MTBF),2)
    Cost_total=round(sum(dfEst['Costo_Falla']),0)
    inds_prome=round(average(dfEst['Indisponibilidad']),1)
    TTR_prome=round(average(dfEst['TTR_(HORAS)']),1)
    return 'Para el periodo de tiempo seleccionado se encontró: '+str(Cant_Fallas)+' fallas, '+ str(Horas_voladas)+' horas de vuelo, '+'el tiempo medio entre fallas es de '+str(MTBF)+', la confiabilidad es de '+str("{:.2%}".format(Confiab))+', el costo total de las fallas es: COP '+str("${:,.2f}".format(Cost_total))+', la indisponibilidad promedio generada por este tipo de fallas es: '+str(inds_prome)+' horas'

# Método para generar una imagen con los sistemas críticos en un período de tiempo por flota

@app.get('/detsistemasFlota/{flota}/fechaInicial/{fechaInicial}/fechaFinal/{fechaFinal}')
def sist_crit_flota(flota, fechaInicial='01-01-2010', fechaFinal='31-12-2022'):
    name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=20))
    fechaI = datetime.strptime(fechaInicial, '%d-%m-%Y')
    fechaF = datetime.strptime(fechaFinal, '%d-%m-%Y')
    flota = flota.upper()
    dfFlota=df[df['Flota']==flota].copy()
    dfFlota=dfFlota[dfFlota['Fecha_Creacion_ Aviso']>=fechaI].copy()
    dfFlota=dfFlota[dfFlota['Fecha_Creacion_ Aviso']<=fechaF].copy()
    tabla=dfFlota.groupby("ATA").size().reset_index(name="Conteo").sort_values("Conteo", ascending=False)
    tabla["Porcentaje"]=100*tabla["Conteo"]/dfFlota.shape[0]
    tabla["Acumulado"]=np.cumsum(tabla["Porcentaje"])
    color1 = 'steelblue'
    color2 = 'red'
    line_size = 4
    fig = plt.figure(figsize=(8, 12))
    fig, ax = plt.subplots()
    ax.bar(tabla.ATA, tabla['Conteo'], color=color1)
    ax.xaxis.label.set_size(12)
    for x, y in zip(tabla.ATA, tabla['Conteo']):
        plt.text(x, y-0.1, str(y), ha='center', va='bottom', fontsize=10.5)
    ax2 = ax.twinx()
    ax2.plot(tabla.ATA, tabla['Acumulado'], color=color2, marker="D", ms=line_size)
    ax2.yaxis.set_major_formatter(PercentFormatter())
    ax2.xaxis.label.set_size(12)
    ax.tick_params(axis='y', colors=color1)
    ax2.tick_params(axis='y', colors=color2)
    plt.title('Diagrama de pareto sistemas críticos ', size = 14)
    ax.set(xlabel = 'Sistemas')
    ax.set(ylabel = 'Cant. Fallas')
    ax2.set(ylabel = '% Acumulado')
    plt.savefig(name+'.jpeg')
    return FileResponse(name+'.jpeg')

# Método para entregar datos estadísticos básicos de valor para la operación en un período de tiempo por flota

@app.get('/estadisticasFallasFlota/{flota}/fechaInicial/{fechaInicial}/fechaFinal/{fechaFinal}')
async def estadisticasFallasFlota(flota, fechaInicial='01-01-2010', fechaFinal='31-12-2022'):
    fechaI = datetime.strptime(fechaInicial, '%d-%m-%Y')
    fechaF = datetime.strptime(fechaFinal, '%d-%m-%Y')
    flota = flota.upper()
    dfEst=df[df['Flota']==flota].copy()
    dfEst=dfEst[dfEst['Fecha_Creacion_ Aviso']>=fechaI].copy()
    dfEst=dfEst[dfEst['Fecha_Creacion_ Aviso']<=fechaF].copy()
    Cant_Fallas=len(dfEst)
    Horas_voladas=round(sum(dfEst['TBF']),1)
    MTBF=round(Horas_voladas/Cant_Fallas,1)
    Confiab=round(math.exp(-3.2/MTBF),2)
    Cost_total=round(sum(dfEst['Costo_Falla']),0)
    inds_prome=round(average(dfEst['Indisponibilidad']),1)
    TTR_prome=round(average(dfEst['TTR_(HORAS)']),1)
    return 'Para el periodo de tiempo seleccionado se encontró: '+str(Cant_Fallas)+' fallas, '+ str(Horas_voladas)+' horas de vuelo, '+'el tiempo medio entre fallas es de '+str(MTBF)+', la confiabilidad es de '+str("{:.2%}".format(Confiab))+', el costo total de las fallas es: COP '+str("${:,.2f}".format(Cost_total))+', la indisponibilidad promedio generada por este tipo de fallas es: '+str(inds_prome)+' horas'

# Método para generar una imagen con la frecuencia de fallas en un período de tiempo por flota

@app.get('/frecuenciaFallasFlota/{flota}/fechaInicial/{fechaInicial}/fechaFinal/{fechaFinal}')
async def frecuenciaFallasFlota(flota, fechaInicial='01-01-2010', fechaFinal='31-12-2022'):
    name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=20))
    fechaI = datetime.strptime(fechaInicial, '%d-%m-%Y')
    fechaF = datetime.strptime(fechaFinal, '%d-%m-%Y')
    flota = flota.upper()
    dfFrec=df[df['Flota']==flota].copy()
    dfFrec=dfFrec[dfFrec['Fecha_Creacion_ Aviso']>=fechaI].copy()
    dfFrec=dfFrec[dfFrec['Fecha_Creacion_ Aviso']<=fechaF].copy()
    dfFrec['N_F']=1
    dfFrec['F_acum']=np.cumsum(dfFrec["N_F"])
    dfFrec['HV_acum']=round(np.cumsum(dfFrec["TBF"]),1)
    dfFrec['MTBF']=round(dfFrec['HV_acum']/dfFrec['F_acum'],2)
    color1 = 'steelblue'
    color2 = 'red'
    line_size = 4
    fig = plt.figure(figsize=(8, 12))
    fig, ax = plt.subplots()
    ax.plot(dfFrec.HV_acum, dfFrec['MTBF'], color=color1)
    ax2 = ax.twinx()
    ax2.plot(dfFrec.HV_acum, dfFrec['F_acum'], color=color2, ms=line_size)
    ax.tick_params(axis='y', colors=color1)
    ax2.tick_params(axis='y', colors=color2)
    plt.title('Frecuencia de fallas', size = 14)
    ax.set(xlabel = 'HV (horas)')
    ax.set(ylabel = 'MTBF')
    ax2.set(ylabel = 'Fallas')
    plt.savefig(name+'.jpeg')
    return FileResponse(name+'.jpeg')

# Método para generar una imagen con la frecuencia de fallas en un período de tiempo por Matrícula

@app.get('/frecuenciaFallasMat/{matricula}/fechaInicial/{fechaInicial}/fechaFinal/{fechaFinal}')
async def frecuenciaFallasMat(matricula, fechaInicial='01-01-2010', fechaFinal='31-12-2022'):
    name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=20))
    fechaI = datetime.strptime(fechaInicial, '%d-%m-%Y')
    fechaF = datetime.strptime(fechaFinal, '%d-%m-%Y')
    matricula = matricula.upper()
    dfFrecM=df[df['Matricula']==matricula].copy()
    dfFrecM=dfFrecM[dfFrecM['Fecha_Creacion_ Aviso']>=fechaI].copy()
    dfFrecM=dfFrecM[dfFrecM['Fecha_Creacion_ Aviso']<=fechaF].copy()
    dfFrecM['N_F']=1
    dfFrecM['F_acum']=np.cumsum(dfFrecM["N_F"])
    dfFrecM['HV_acum']=round(np.cumsum(dfFrecM["TBF"]),1)
    dfFrecM['MTBF']=round(dfFrecM['HV_acum']/dfFrecM['F_acum'],2)
    color1 = 'steelblue'
    color2 = 'red'
    line_size = 4
    fig = plt.figure(figsize=(8, 12))
    fig, ax = plt.subplots()
    ax.plot(dfFrecM.HV_acum, dfFrecM['MTBF'], color=color1)
    ax2 = ax.twinx()
    ax2.plot(dfFrecM.HV_acum, dfFrecM['F_acum'], color=color2, ms=line_size)
    ax.tick_params(axis='y', colors=color1)
    ax2.tick_params(axis='y', colors=color2)
    plt.title('Frecuencia de fallas', size = 14)
    ax.set(xlabel = 'HV (horas)')
    ax.set(ylabel = 'MTBF')
    ax2.set(ylabel = 'Fallas')
    plt.savefig(name+'.jpeg')
    return FileResponse(name+'.jpeg')