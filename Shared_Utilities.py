import matplotlib.pyplot as plt
from io import StringIO
import seaborn as sns
import pandas as pd
import os

# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

class bcolors:
    ENDC = '\033[0m'
    BLUE = '\033[94m'
    ORNG = '\033[38;5;208m'

# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

def clearTerminal():
    '''Funzione per stampare il numero di valori nulli per attributo.'''
    
    os.system('cls') # il comando cls è per Windows 
    
# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

def load_dataset(full=False):
    '''Funzione per caricare il dataset.
    - full: booleano per decidere se caricare il dataset completo o splittato in X e y.'''
    
    dataset = pd.read_csv('weather_classification_data.csv') # import del dataset
    
    if full:
        return dataset # restituisce il dataset completo
    else:
        y = dataset['Weather Type']                # target da predire
        X = dataset.drop(['Weather Type'], axis=1) # rimozione del target dal dataset
        return X, y                                # restituisce il dataset splittato in X e y
