from Shared_Utilities import clear_terminal
from AnalisiDataset import dataset_overview_menu
from DecisionTree import decision_tree_main
from NaiveBayes import naive_bayes_main
from SVM import SVM_main
from KNN import KNN_main

# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

def interactiveMenu():
    '''Funzione per l'avvio del menu interattivo.'''
    
    scelta = -1 # scelta dell'utente
    
    while scelta != 0:
        clear_terminal()
        printLogo()
        printChoiches()
        scelta = input()
        
        if scelta == "0":
            clear_terminal()
            return
        elif scelta == "1":
            dataset_overview_menu()
        elif scelta == "2":
            decision_tree_main()
        elif scelta == "3":
            naive_bayes_main()
        elif scelta == 4:
              

# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

def printLogo():
    '''Funzione per stampare il logo del progetto.'''
    
    print("▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀")
    print("▀▄ ▄▄         ▄▄    ▄▄       ▄▀")
    print("▀▄ ███▄     ▄███    ██       ▄▀")
    print("▀▄ ██ ▀█▄ ▄█▀ ██    ██       ▄▀")
    print("▀▄ ██   ▀█▀   ██    ██       ▄▀")
    print("▀▄ ██         ██    ██       ▄▀")
    print("▀▄ ██         ██    ██▄▄▄▄▄▄ ▄▀")
    print("▀▄ ▀▀         ▀▀    ▀▀▀▀▀▀▀▀ ▄▀")
    print("▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀")
    print("\n")
    
# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

def printChoiches():
    '''Funzione per stampare le scelte possibili per l'utente.'''
    
    print("Scegliere un'opzione:")
    print("[1] Analisi del dataset")
    print("[2] Decision Tree")
    print("[3] Naive Bayes")
    print("[4] SVM")
    print("[5] KNN [Custom]")
    print("[6] Classificatore Multiplo")
    print("[7] Confronto . . .")
    print("[0] Esci dal programma")

# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

if __name__ == '__main__':
    interactiveMenu()
