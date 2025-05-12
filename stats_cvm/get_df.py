import os
import pandas as pd

def get_df_from_csv(nom_fichier):
    dossier_actuel = os.getcwd()

    fichier = os.path.join(dossier_actuel, 'donnees', nom_fichier)

    with open(fichier, 'r') as f:
        first_line = f.readline()
        if ',' in first_line and ';' not in first_line:
            sep = ','
            decimal='.'
        elif ';' in first_line and '.' in first_line:
            sep = ';'
            decimal='.'
        elif ';' in first_line:
            sep = ';'
            decimal=','
        else:
            sep = ','
            decimal='.'
            print("Vérifier comment les valeurs sont séparées dans le fichier CSV")

    return pd.read_csv(fichier, sep=sep, decimal=decimal, encoding='utf-8')