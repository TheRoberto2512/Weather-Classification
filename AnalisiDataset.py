from Shared_Utilities import os, pd, plt, sns, StringIO
from Shared_Utilities import clearTerminal, load_dataset

# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

def datasetOverviewMenu():
    '''Funzione per l'avvio del sotto-menu per fare l'analisi del Dataset.'''
    
    df = load_dataset(full=True)
    
    scelta = -1 # scelta dell'utente
    
    while scelta != 0:
        print_choiches()
        scelta = input()
        
        if scelta == "0":
            clearTerminal()
            return
        elif scelta == "1":
            clearTerminal()
            print_info(df)
        elif scelta == "2":
            clearTerminal()
            print("Prime righe:")
            print(df.head())
            print("\nUltime righe:")
            print(df.tail())
        elif scelta == "3":
            clearTerminal()
            print_null_values(df)
        elif scelta == "4":
            clearTerminal()
            print("Bilanciamento classe target:")
            plot_class_distrib(df)
        elif scelta == 5:
            clearTerminal()
        
        print("\nPremere qualsiasi tasto per tornare indietro")
        input()
        clearTerminal()

# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

def print_choiches():
    '''Funzione per stampare le scelte possibili per l'analisi del dataset.'''
    
    print("Scegliere il tipo di analisi da effettuare:")
    print("[1] Informazioni generali sugli attributi")
    print("[2] Visualizzazione delle prime e ultime righe")
    print("[3] Visualizzazione del numero di valori nulli per attributo")
    print("[4] Visualizzazione della distribuzione degli elementi nelle classi target")
    print("[5] Matrice di correlazione degli attributi")
    print("[0] Ritorna al menu principale")

# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

def print_info(df):
    '''Funzione per stampare le informazioni generali sul dataset eliminando alcune righe superflue.'''
    
    buffer = StringIO()
    df.info(buf=buffer)
    info_string = buffer.getvalue()
    info_string = info_string.split('\n')

    for i in range(1, len(info_string)-2):
        print(info_string[i])

# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

def print_null_values(df):
    '''Funzione per stampare il numero di valori nulli per attributo.'''
    
    print("Numero di valori nulli per attributo:")
    null_string = df.isnull().sum()
    null_string = str(null_string).split('\n')

    for i in range(0, len(null_string)-1):
        print(null_string[i])

# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

def plot_class_distrib(df):
    '''Funzione per stampare la distribuzione delle classi target.'''
    
    classi = df['Weather Type'].unique()
    counts = []
    for classe in classi:
        counts.append(df[df['Weather Type'] == classe].shape[0])
        print(f"Classe %s:\t%d" % (classe, counts[-1]))
    
    colors = ["#0070c0", "#595959", "#ffc001", "#cbcbcb"]
    plt.bar(classi, counts, color=colors)
    plt.title("Distribuzione classi target")
    plt.show()

# DA RIMUOVERE
if __name__ == '__main__':
    datasetOverviewMenu()