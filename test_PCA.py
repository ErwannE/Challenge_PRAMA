import pandas as pd # Pour lire les fichiers
import numpy as np # Pour effectuer des calculs mathématiques
import matplotlib.pyplot as plt # Pour réaliser des graphiques
from scipy import stats # Pour effectuer des calculs statistiques
from sklearn.preprocessing import StandardScaler # Pour normaliser les données
from sklearn import decomposition # Pour effectuer une ACP

# Lecture du fichier
df_test = pd.read_csv("Data/train_data.csv", sep=",")
df_test = df_test.drop(columns=["id"])

df_test['cos_month'] = np.cos((2*np.pi/12)*pd.to_datetime(df_test['date']).dt.month)
print(df_test[['cos_month','date']])
df_test = df_test.drop(columns=["date"])
df_test = df_test[['nb_sdb','m2_interieur','m2_jardin','m2_etage']] 
print(df_test.head(5))

pca = decomposition.PCA()
pca.fit(df_test)

print("Composantes principales :")
print(pca.components_)

print("Variance expliquée par chaque composante :")
print(pca.explained_variance_)

plt.plot(range(1,5),pca.explained_variance_)
plt.show()