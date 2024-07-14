from Imports import pd, train_test_split, RANDOM_STATE
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

def load_dataset(raw=False, one_hot=True):
    '''Funzione per caricare il dataset.
    
    Parametri:
    - raw: booleano per decidere se caricare il dataset senza nessuna modifica (grezzo).
    - one_hot: booleano per decidere se applicare la codifica one-hot agli attributi categorici.
    '''
    
    dataset = pd.read_csv('weather_classification_data.csv') # import del dataset
    
    if raw:
        return dataset # restituisce il dataset completo
    elif one_hot:
        dataset = pd.get_dummies(dataset, columns=['Cloud Cover', 'Season', 'Location'], drop_first=False)
        
    y = dataset['Weather Type']                # target da predire
    X = dataset.drop(['Weather Type'], axis=1) # rimozione del target dal dataset
    return X, y                                # restituisce il dataset splittato in X e y

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
    
    scaler = StandardScaler()               # creo l'oggetto scaler per la standardizzazione
    scaler.fit(X)                           # calcola media e deviazione standard per la standardizzazione
    X_scaled = scaler.transform(X)          # applica la standardizzazione (mu 0, devstd 1)
    
    return X_scaled, y

# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

def load_smaller_dataset(ratio=0.5):
    '''
    Funzione per caricare un dataset con meno record.
    
    Parametri:
    - ratio: percentuale di record da restituire (default: 0.5).
    '''
    
    X, y = load_dataset(one_hot=True)
    
    X_return, _, y_return, _ = train_test_split(X, y, test_size=ratio, stratify=y)

    return X_return, y_return  # restituisce solamente parte del dataset originale

# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

def load_bigger_dataset(multiplier=2):
    '''
    Funzione per caricare un dataset con più record.
    
    Parametri:
    - multiplier: moltiplicatore per il numero di record (defeault: 2).
    '''
    
    X, y = load_dataset(one_hot=True)
    
    # in questo caso vogliamo RADDOPPIARE i records per ogni classe
    dict_smo = {'Rainy': len(X[y == 'Rainy']) * multiplier,
                'Sunny': len(X[y == 'Sunny']) * multiplier,
                'Cloudy': len(X[y == 'Cloudy']) * multiplier,
                'Snowy': len(X[y == 'Snowy']) * multiplier}
    
    # i records saranno generati sinteticamente da SMOTE
    smote = SMOTE(random_state=RANDOM_STATE, sampling_strategy=dict_smo) # creo l'oggetto SMOTE
    
    X_over, y_over = smote.fit_resample(X, y) # applico l'oversampling
    
    return X_over, y_over
       
# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

def load_custom_dataset(size="base", standardization=False):
    '''Funzione per il caricamento del dataset con preprocessing personalizzato.
    
    Parametri:
    - size: dimensione del dataset da caricare (default: base).
      - base: dataset originale.
      - small: dataset con meno record.
      - big: dataset con più record.
    - standardization: se True, il dataset verrà standardizzato (default: False).
    '''
    
    # si sceglie prima il tipo di dataset da pre-processare
    if size == "base":
        X, y = load_dataset(one_hot=True) # la one hot è gia applicata a small e big
    elif size == "small":
        X, y = load_smaller_dataset() 
    elif size == "big":
        X, y = load_bigger_dataset()
        
    # si standardizza il dataset se richiesto
    if standardization:
        X, y = load_standardized_dataset(X, y)
    
    print(X.shape, y.shape)
    return X, y
    
# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

if __name__ == '__main__':
    load_custom_dataset(size='big', standardization=True) # esempio di utilizzo della funzione