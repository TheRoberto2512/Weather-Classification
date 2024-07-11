from Shared_Utilities import pd
from Imports import RANDOM_STATE

# import metodi/classi di preprocessing
from sklearn.preprocessing import StandardScaler

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


