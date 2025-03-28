from Preprocessing import load_dataset, load_standardized_dataset, load_smaller_dataset, load_bigger_dataset, load_custom_dataset, feature_selection, naive_outliers_handling
from Imports import os, plt, sns, np, train_test_split, RANDOM_STATE
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

class Colors():
    '''Classe per la gestione dei colori nel terminale.'''
    
    RESET = '\033[0m'           # colore di default
    BLUE = '\033[94m'           # colore blu
    ORNG = '\033[38;5;208m'     # colore arancione
    RED = '\033[91m'            # colore rosso
    GREEN = '\033[92m'          # colore verde

# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

def clear_terminal():
    '''Funzione per cancellare la cronologia del terminale.'''
    
    os.system('cls') # il comando cls è per Windows 
    
# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

def chose_dataset(return_name = False):
    '''Funzione per la scelta del dataset da utilizzare per l'addestramento del modello.'''
    
    scelta = -1 # scelta dell'utente
    
    while scelta != "scelto":
        clear_terminal()
        print("Scegliere il dataset con cui addestrare il modello:")
        print(f"{Colors.BLUE}[1]{Colors.RESET} Dataset originale")
        print(f"{Colors.BLUE}[2]{Colors.RESET} Dataset standardizzato")
        print(f"{Colors.BLUE}[3]{Colors.RESET} Dataset con meno records (50%)")
        print(f"{Colors.BLUE}[4]{Colors.RESET} Dataset con più records  (200%)")
        print(f"{Colors.BLUE}[5]{Colors.RESET} Dataset con feature selection")
        print(f"{Colors.BLUE}[6]{Colors.RESET} Dataset con preprocessing personalizzato")
        print(f"{Colors.BLUE}[7]{Colors.RESET} Dataset senza outliers")
        print(f"{Colors.ORNG}[0]{Colors.RESET} Torna al menu principale")
        
        name = ""
        scelta = input()
        
        if scelta == "0":
            return None, None
        elif scelta == "1":
            X, y = load_dataset(one_hot=True)
            scelta = "scelto" ; name = "Originale"
        elif scelta == "2":
            X, y = load_standardized_dataset()
            scelta = "scelto" ; name = "STD"
        elif scelta == "3":
            X, y = load_smaller_dataset()
            scelta = "scelto" ; name = "Small"
        elif scelta == "4":
            X, y = load_bigger_dataset()
            scelta = "scelto" ; name = "Big"
        elif scelta == "5":
            X, y = feature_selection()
            scelta = "scelto" ; name = "FS"
        elif scelta == "6":
            X, y, name = chose_custom_dataset()
            if (X is None) & (y is None): # se non è stata effettuata una scelta
                scelta = -1
            else:
                scelta = "scelto"
        elif scelta == "7":
            X, y = naive_outliers_handling()
            scelta = "scelto" ; name = "No Outliers"
         
    clear_terminal()      
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    if return_name:
        return (X_train, X_test, y_train, y_test), name
    
    return (X_train, X_test, y_train, y_test)

# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

def chose_custom_dataset():
    '''Funzione per la scelta del dataset con preprocessing personalizzato.'''
    
    scelta = -1 # scelta dell'utente
    
    while scelta != "scelto":
        clear_terminal()
        print(f"Scegliere la combinazione di pre-processing che si preferisce:")
        print(f"{Colors.BLUE}[1]{Colors.RESET} Dataset con meno records (50%) + Standardizzazione")
        print(f"{Colors.BLUE}[2]{Colors.RESET} Dataset con più records (200%) + Standardizzazione")
        print(f"{Colors.BLUE}[3]{Colors.RESET} Dataset con feature selection + Standardizzazione")
        print(f"{Colors.BLUE}[4]{Colors.RESET} Dataset con feature selection + meno records")
        print(f"{Colors.BLUE}[5]{Colors.RESET} Dataset con feature selection + più records")
        print(f"{Colors.BLUE}[6]{Colors.RESET} Dataset con feature selection + meno records + Standardizzazione")
        print(f"{Colors.BLUE}[7]{Colors.RESET} Dataset con feature selection + più records + Standardizzazione")
        print(f"{Colors.ORNG}[0]{Colors.RESET} Torna indietro")
        
        scelta = input()
        
        if scelta == "0":
            return None, None, None
        elif scelta == "1":
            X, y = load_custom_dataset(size="small", standardization=True)
            scelta = "scelto" ; name = "Small+STD"
        elif scelta == "2":
            X, y = load_custom_dataset(size="big", standardization=True)
            scelta = "scelto" ; name = "Big+STD"
        elif scelta == "3":
            X, y = load_custom_dataset(apply_feature_selection=True, standardization=True)
            scelta = "scelto" ; name = "FS+STD"
        elif scelta == "4":
            X, y = load_custom_dataset(apply_feature_selection=True, size="small")
            scelta = "scelto" ; name = "Small+STD"
        elif scelta == "5":
            X, y = load_custom_dataset(apply_feature_selection=True, size="big")
            scelta = "scelto" ; name = "Big+FS"
        elif scelta == "6":
            X, y = load_custom_dataset(apply_feature_selection=True, size="small", standardization=True)
            scelta = "scelto" ; name = "Small+FS+STD"
        elif scelta == "7":
            X, y = load_custom_dataset(apply_feature_selection=True, size="big", standardization=True)
            scelta = "scelto" ; name = "Big+FS+STD"
         
    clear_terminal()      
    return X, y, name

# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

def print_confusion_matrix(y_true, y_pred):
    '''
    Funzione per la stampa della matrice di confusione.
    
    Parametri:
    - y_true: array di classi reali.
    - y_pred: array di classi predette.
    '''
    
    # codifica delle classi
    le = LabelEncoder()
    le.fit(np.concatenate((y_true, y_pred), axis=None))
    class_names = le.classes_
    
    cm = confusion_matrix(y_true, y_pred) # calcola la matrice di confusione
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    
    plt.xlabel('Classi Predette')
    plt.ylabel('Classi Reali')
    
    plt.title('Matrice di confusione')
    plt.show()
    
# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #