from Imports import pd, train_test_split, RANDOM_STATE
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

def load_dataset(raw=False, one_hot=True):
    '''
    Funzione per caricare il dataset.
    
    Parametri:
    - raw: booleano per decidere se caricare il dataset senza nessuna modifica (grezzo).
    - one_hot: booleano per decidere se applicare la codifica one-hot agli attributi categorici.
    '''
    
    dataset = pd.read_csv('weather_classification_data.csv') # import del dataset
    
    if raw:
        return dataset                          # restituisce il dataset grezzo
    elif one_hot:
        dataset = pd.get_dummies(dataset, columns=['Cloud Cover', 'Season', 'Location'], drop_first=False)
        
    y = dataset['Weather Type']                 # target da predire
    X = dataset.drop(['Weather Type'], axis=1)  # rimozione del target dal dataset
    return X, y                                 # restituisce il dataset splittato in X e y

# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

def load_standardized_dataset(X=None, y=None):
    '''
    Funzione per caricare il dataset standardizzato. Se non viene fornito un 
    dataset, viene caricato quello originale con solo la one hot applicata.
    
    Parametri:
    - X: array di feature (default: None).
    - y: array di target (default: None).
    '''
    
    if X is None and y is None:             # se non viene fornito un dataset
        X, y = load_dataset(one_hot=True)   # carico il dataset con solo la one_hot applicata
    
    # colonne categoriche da non considerare per la standardizzazione
    categorical = ['Cloud Cover_clear', 'Cloud Cover_cloudy', 'Cloud Cover_overcast', 'Cloud Cover_partly cloudy',
                   'Season_Autumn', 'Season_Spring', 'Season_Summer', 'Season_Winter',
                   'Location_coastal', 'Location_inland', 'Location_mountain']
    
    X_cat = X[categorical]                  # seleziono le colonne categoriche
    X = X.drop(categorical, axis=1)         # rimuovo le colonne categoriche
    
    X.columns = X.columns.map(str)          # converto le colonne in stringhe e le salvo
    
    scaler = StandardScaler()               # creo l'oggetto scaler per la standardizzazione
    scaler.fit(X)                           # calcola media e deviazione standard per la standardizzazione
    X_scaled = scaler.transform(X)          # applica la standardizzazione (mu 0, devstd 1)
    
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns) # reimposto le colonne in stringa
    X_scaled_df.reset_index(drop=True, inplace=True)        # resetto l'indice per evitare problemi di allineamento
    
    X = pd.concat([X_scaled_df, X_cat.reset_index(drop=True)], axis=1) # ricongiungo le colonne
    
    return X, y

# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

def load_smaller_dataset(X=None, y=None, ratio=0.5):
    '''
    Funzione per caricare un dataset con meno record.
    
    Parametri:
    - X: array di feature.
    - y: array di target.
    - ratio: percentuale di record da restituire (default: 0.5).
    '''
    
    if X is None and y is None: # se non viene fornito un dataset
        X, y = load_dataset(one_hot=True)
    
    X_return, _, y_return, _ = train_test_split(X, y, test_size=ratio, stratify=y, random_state=RANDOM_STATE)

    return X_return, y_return  # restituisce solamente parte del dataset originale

# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

def load_bigger_dataset(X=None, y=None, multiplier=2):
    '''
    Funzione per caricare un dataset con più record.
    
    Parametri:
    - X: array di feature.
    - y: array di target.
    - multiplier: moltiplicatore per il numero di record (defeault: 2).
    '''
    
    if X is None and y is None: # se non viene fornito un dataset
        X, y = load_dataset(one_hot=True)
    
    # in questo caso vogliamo RADDOPPIARE i records per ogni classe
    dict_smote = {'Rainy': len(X[y == 'Rainy']) * multiplier,
                'Sunny': len(X[y == 'Sunny']) * multiplier,
                'Cloudy': len(X[y == 'Cloudy']) * multiplier,
                'Snowy': len(X[y == 'Snowy']) * multiplier}
    
    # i records saranno generati sinteticamente da SMOTE
    smote = SMOTE(random_state=RANDOM_STATE, sampling_strategy=dict_smote) # creo l'oggetto SMOTE
    
    X_over, y_over = smote.fit_resample(X, y) # applico l'oversampling
    
    return X_over, y_over
       
# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

def load_custom_dataset(apply_feature_selection=False, size="base", standardization=False):
    '''
    Funzione per il caricamento del dataset con preprocessing personalizzato.
    
    Parametri:
    - apply_feature_selection: se True, verrà applicata la selezione delle feature (default: False).
    - size: dimensione del dataset da caricare (default: base).
      - base: dataset originale.
      - small: dataset con meno record.
      - big: dataset con più record.
    - standardization: se True, il dataset verrà standardizzato (default: False).
    '''
    
    X, y = None, None # le variabili vengono inizializzate a None
    
    # si standardizza il dataset se richiesto
    if standardization:
        X, y = load_standardized_dataset() # carico il dataset standardizzato con già la one_hot
    
    # si applica la feature selection se richiesta
    if apply_feature_selection == True:
        X, y = feature_selection(X, y) 
    
    # si ingrandisce/rimpicciolisce il dataset se richiesto
    if size == "small":
        X, y = load_smaller_dataset(X, y) 
    elif size == "big":
        X, y = load_bigger_dataset(X, y)  
        
    return X, y
    
# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

def feature_selection(X=None, y=None):
    '''
    Funzione per la selezione delle feature.
    
    Parametri:
    - X: array di feature.
    - y: array di target.
    '''
    
    if X is None and y is None: # se non viene fornito un dataset
        X, y = load_dataset(one_hot=True)
    
    # ci sono più colonne Cloud Cover a causa della conversione in one_hot
    X = X.drop(['Cloud Cover_clear', 'Cloud Cover_cloudy', 'Cloud Cover_overcast', 'Cloud Cover_partly cloudy', 'Humidity'], axis=1)
    
    # in base alla matrice di correlazione è stato ritenuto più opportuno eliminare
    # le varie colonne 'Cloud Cover_ . . .' e 'Humidity' in quanto presentano
    # correlazioni molto alte con vari attributi, risultando ridondanti.
    
    # maggiori informazioni sulla feature selection sono presenti nella Relazione
    
    return X, y 

# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

def naive_outliers_handling(max_temp=50, records_per_class=6600):
    '''
    Funzione per eliminare gli outliers dal dataset.
    
    Parametri:
    - max_temp: temperatura massima ammissibile (default: 50°C).
    - records_per_class: numero di record per classe (default: 6600).
    '''
    
    df = load_dataset(raw=True) # carico il dataset grezzo
        
    df = df.drop(df[df['Temperature'] > max_temp].index)        # valori di temperatura > 50°C non sono (quasi mai) possibili
    df = df.drop(df[df['Humidity'] > 100].index)                # valori di umidità > 100% non sono possibili
    df = df.drop(df[df['Precipitation (%)'] > 100].index)       # valori di precipitazione > 100% non sono possibili
    df = df.drop(df[df['Atmospheric Pressure'] > 1050].index)   # valori di pressione > 1050 hPa non sono possibili
    
    # applica la hot encoding agli attributi categorici
    df = pd.get_dummies(df, columns=['Cloud Cover', 'Season', 'Location'], drop_first=False)
        
    # suddivide in X e y
    y = df['Weather Type']                 # target da predire
    X = df.drop(['Weather Type'], axis=1)  # rimozione del target dal dataset
    
    # si vogliono pareggiare i records del dataset ampliato (×2)
    dict_smote = {'Rainy': records_per_class, 'Sunny': records_per_class,
                'Cloudy': records_per_class, 'Snowy': records_per_class}
    
    # i records saranno generati sinteticamente da SMOTE
    smote = SMOTE(random_state=RANDOM_STATE, sampling_strategy=dict_smote) # creo l'oggetto SMOTE
    
    X, y = smote.fit_resample(X, y) # applico l'oversampling
    
    X, y = load_standardized_dataset(X, y) # carico il dataset standardizzato
    
    return X, y

# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #