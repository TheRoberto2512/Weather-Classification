from Imports import pd, os, train_test_split
from Preprocessing import load_dataset, load_standardized_dataset, RANDOM_STATE

# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

class Colors():
    '''Classe per la gestione dei colori nel terminale.'''
    
    RESET = '\033[0m'           # colore di default
    BLUE = '\033[94m'           # colore blu
    ORNG = '\033[38;5;208m'     # colore arancione
       

# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #


def clear_terminal():
    '''Funzione per cancellare la cronologia del terminale.'''
    
    os.system('cls') # il comando cls è per Windows 
    
# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

def chose_dataset():
    '''Funzione per la scelta del dataset da utilizzare per l'addestramento del modello.'''
    
    scelta = -1 # scelta dell'utente
    
    while scelta != "scelto":
        clear_terminal()
        print("Scegliere il dataset con cui addestrare il modello:")
        print("[1] Dataset originale")
        print("[2] Dataset standardizzato")
        print("[3] Dataset normalizzato")
        print("[0] Torna al menu principale")
        scelta = input()
        
        if scelta == "0":
            return None
        elif scelta == "1":
            X, y = load_dataset(one_hot=True)
            scelta = "scelto"
        elif scelta == "2":
            X, y = load_standardized_dataset()
            scelta = "scelto"
        elif scelta == "3":
            # . . .
            scelta = "scelto"
         
    clear_terminal()        
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    return (X_train, X_test, y_train, y_test)