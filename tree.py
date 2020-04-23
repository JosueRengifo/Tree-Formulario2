# importamos las librerías que necesitamos
from sklearn.tree import DecisionTreeClassifier,plot_tree,export_graphviz # �rbol de decisi�n para clasificaci�n
from sklearn.model_selection import train_test_split
import pandas
#import graphviz
from io import StringIO ## for Python 3

names = ['herramientas para manejo de la informacion', 'integracion de informacion', 'Anos de experiencia en  sistemas de informacion', 'Bases de Datos', 'informacion clinica', 'informacion salud publica', 'terminologias','class']
namesFeatures = ['herramientas para manejo de la informacion', 'integracion de informacion', 'Anos de experiencia en  sistemas de informacion', 'Bases de Datos', 'informacion clinica', 'informacion salud publica', 'terminologias']
namesTarget=['Alto','Medio','Bajo']
dataframe = pandas.read_csv('data.csv', names=names)
array = dataframe.values
Datos = array[:,0:7]
Clases = array[:,7]
DPrueba=array[19:21,0:7]
aFit,aTest,bFit,bTest=train_test_split(Datos,Clases) # datos de entrenamiento, datos de prueba, clases de entrenamiento, clases de prueba
num_folds = 10
num_instances = len(Datos)
seed = 7
tree = DecisionTreeClassifier(max_depth=10, random_state=7) # parametros opcionales max_depth=2, random_state=42
tree.fit(aFit, bFit) 
DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_split=None, class_weight=None, presort=False) # parametros opcionales max_depth=2, random_state=42
#print("Aprendió en un: ")
print(tree.score(aTest,bTest))
print(tree.score(aFit, bFit))
#Datos de prueba para la predicci�n
print(DPrueba)
predict=tree.predict(DPrueba)
print(predict)
#plot_tree(tree.fit(aTest,bTest)) 
#Generaci�n de archivo para exportaci�n
#dot_data = graphviz.Source(export_graphviz(tree,out_file=None,class_names=namesTarget,feature_names=namesFeatures,filled=True,rounded=True, special_characters=True)) #impurity=False
#graph = graphviz.Source(dot_data)
#graph.render("tree")