# importamos las librerías que necesitamos
from sklearn.tree import DecisionTreeClassifier # �rbol de decisi�n para clasificaci�n
from sklearn.model_selection import train_test_split
import pandas
import numpy as np

from io import StringIO ## for Python 3

def Clasificador(Caracteristicas):
    #Inicializa el algoritmo de clasificación y lo entrena con datos de prueba
    names = ['herramientas para manejo de la informacion', 'integracion de informacion', 'Anos de experiencia en  sistemas de informacion', 'Bases de Datos', 'informacion clinica', 'informacion salud publica', 'terminologias','class']
    namesFeatures = ['herramientas para manejo de la informacion', 'integracion de informacion', 'Anos de experiencia en  sistemas de informacion', 'Bases de Datos', 'informacion clinica', 'informacion salud publica', 'terminologias']
    namesTarget=['Alto','Medio','Bajo']
    dataframe = pandas.read_csv('data.csv', names=names)
    array = dataframe.values
    Datos = array[:,0:7]
    Clases = array[:,7]
    aFit,aTest,bFit,bTest=train_test_split(Datos,Clases) # datos de entrenamiento, datos de prueba, clases de entrenamiento, clases de prueba
    num_folds = 10
    num_instances = len(Datos)
    seed = 7
    tree = DecisionTreeClassifier(max_depth=10, random_state=7) # parametros opcionales max_depth=2, random_state=42
    tree.fit(aFit, bFit) 
    DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_split=None, class_weight=None, presort=False) # parametros opcionales max_depth=2, random_state=42
    predict=tree.predict(Caracteristicas)
    print (predict[0])
    return predict

