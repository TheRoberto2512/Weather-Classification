from Shared_Utilities import clear_terminal, load_dataset, Colors
from Imports import pd, plt, StringIO, sns
from dython.nominal import associations                 # per la matrice di correlazione completa (anche attributi categorici)

# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

def dataset_overview_menu():
    '''Funzione per l'avvio del sotto-menu per fare l'analisi del Dataset.'''
    
    df = load_dataset(raw=True) # carica il dataset senza nessuna modifica

    scelta = -1 # scelta dell'utente
    
    while scelta != 0:
        clear_terminal()
        print_choiches()
        scelta = input()
        
        if scelta == "0":
            clear_terminal()
            return
        elif scelta == "1":
            clear_terminal()
            print_info(df)
        elif scelta == "2":
            clear_terminal()
            print("Prime righe:")
            print(df.head())
            print("\nUltime righe:")
            print(df.tail())
        elif scelta == "3":
            clear_terminal()
            print_null_values(df)
        elif scelta == "4":
            clear_terminal()
            print("Bilanciamento classe target:")
            plot_class_distrib(df)
        elif scelta == "5":
            clear_terminal()
            plot_corr_matrix(df)
        elif scelta == "6":
            clear_terminal()
            print_boxplot(df)
        
        print(f"\nPremere {Colors.ORNG}INVIO{Colors.RESET} per tornare indietro")
        input()
        clear_terminal()

# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

def print_choiches():
    '''Funzione per stampare le scelte possibili per l'analisi del dataset.'''
    
    print("Scegliere il tipo di analisi da effettuare:")
    print(f"{Colors.BLUE}[1]{Colors.RESET} Informazioni generali sugli attributi")
    print(f"{Colors.BLUE}[2]{Colors.RESET} Visualizzazione delle prime e ultime righe")
    print(f"{Colors.BLUE}[3]{Colors.RESET} Visualizzazione del numero di valori nulli per attributo")
    print(f"{Colors.BLUE}[4]{Colors.RESET} Visualizzazione della distribuzione degli elementi nelle classi target")
    print(f"{Colors.BLUE}[5]{Colors.RESET} Matrice di correlazione degli attributi")
    print(f"{Colors.BLUE}[6]{Colors.RESET} Boxplot per l'analisi degli outliers")
    print(f"{Colors.ORNG}[0]{Colors.RESET} Ritorna al menu principale")

# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

def print_info(df):
    '''
    Funzione per stampare le informazioni generali sul dataset eliminando alcune righe superflue.
    
    Parametri:
    - df: DataFrame, il dataset da analizzare.
    '''
    
    buffer = StringIO()
    df.info(buf=buffer)
    info_string = buffer.getvalue()
    info_string = info_string.split('\n')

    for i in range(1, len(info_string)-2):
        print(info_string[i])

# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

def print_null_values(df):
    '''
    Funzione per stampare il numero di valori nulli per attributo.
    
    Parametri:
    - df: DataFrame, il dataset da analizzare.
    '''
    
    print("Numero di valori nulli per attributo:")
    null_string = df.isnull().sum()
    null_string = str(null_string).split('\n')

    for i in range(0, len(null_string)-1):
        print(null_string[i])

# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

def plot_class_distrib(df):
    '''
    Funzione per stampare la distribuzione delle classi target.
    
    Parametri:
    - df: DataFrame, il dataset da analizzare.
    '''
    
    classi = df['Weather Type'].unique() # estraggo le classi target
    counts = []
    
    for classe in classi:
        counts.append(df[df['Weather Type'] == classe].shape[0]) # conto il numero di elementi per ogni classe
        print(f"Classe %s:\t%d" % (classe, counts[-1]))          # stampo il numero di elementi per ogni classe
    
    colors = ["#0070c0", "#595959", "#ffc001", "#cbcbcb"]
    bars = plt.bar(classi, counts, color=colors)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), va='bottom', ha='center', color='black', size=10)
        # aggiungo il numero di elementi sopra ogni barra
        
    plt.title("Distribuzione classi target")
    plt.show()

# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

def plot_corr_matrix(df):
    '''
    Funzione per stampare la matrice di correlazione degli attributi.
    
    Parametri:
    - df: DataFrame, il dataset da analizzare.
    '''
        
    df = df.drop(['Weather Type'], axis=1) # si toglie la colonna target   
    
    complete_correlation = associations(df, figsize=(10,10), cmap="coolwarm")
    df_complete_corr = complete_correlation['corr']
    df_complete_corr.dropna(axis=1, how='all').dropna(axis=0, how='all')
    
# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # 

def print_boxplot(df):
    '''Funzione per stampare il boxplot per l'analisi degli outliers.'''
    
    numerical_cols = df.select_dtypes(include='number').columns # solo colonne numeriche
    _, ax = plt.subplots(2, 4, figsize=(8, 8))                  # subplot 2 righe × 4 colonne
    ax = ax.flatten()
    
    for i, colonna in enumerate(numerical_cols):
        sns.boxplot(data=df, y=colonna, ax=ax[i])               # boxplot per ogni colonna numerica
        ax[i].set_title(colonna)
        
    plt.tight_layout()
    plt.show()

# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #