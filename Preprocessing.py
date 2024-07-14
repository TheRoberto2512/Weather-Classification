from Imports import pd, train_test_split, RANDOM_STATE
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

def load_dataset(raw=False, one_hot=True):
    '''Funzione per caricare il dataset.
    - raw: booleano per decidere se caricare il dataset senza nessuna modifica (grezzo).
    - one:hot: booleano per decidere se applicare la codifica one-hot agli attributi categorici.'''
    
    dataset = pd.read_csv('weather_classification_data.csv') # import del dataset
    
    if raw:
        return dataset # restituisce il dataset completo
    elif one_hot:
        dataset = pd.get_dummies(dataset, columns=['Cloud Cover', 'Season', 'Location'], drop_first=False)
        
    y = dataset['Weather Type']                # target da predire
    X = dataset.drop(['Weather Type'], axis=1) # rimozione del target dal dataset
    return X, y                                # restituisce il dataset splittato in X e y

# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

def load_standardized_dataset():
    '''Funzione per caricare il dataset standardizzato.'''
    
    X, y = load_dataset(one_hot=True)  # carico il dataset con solo la one_hot applicata
    
    scaler = StandardScaler()          # creo l'oggetto scaler per la standardizzazione
    scaler.fit(X)                      # calcola media e deviazione standard per la standardizzazione
    X_scaled = scaler.transform(X)     # applica la standardizzazione (mu 0, devstd 1)
    
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
    - multiplier: moltiplicatore per il numero di record.
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