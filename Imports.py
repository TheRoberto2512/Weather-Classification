# -- Imports condivisi tra i vari file -- #
from sklearn.model_selection import train_test_split, cross_validate    # per la funzione per splittare il dataset e la funzione per la cross validation 
from sklearn.metrics import accuracy_score, classification_report       # per le metriche di valutazione    
import matplotlib.pyplot as plt                                         # per la libreria per i grafici                  
from io import StringIO                                                 # per la manipolazione di stringhe
import seaborn as sns                                                   # per la matrice di correlazione e i boxplot            
import pandas as pd                                                     # per gestire i dataframe                
import numpy as np                                                      # per i calcoli matematici        
import os                                                               # per la gestione del sistema operativo            

# -- Costanti -- #
RANDOM_STATE = 1407
'''Costante utilizzata come random_state per tutte le funzioni che lo richiedono.'''