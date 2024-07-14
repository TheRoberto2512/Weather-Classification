from Shared_Utilities import clear_terminal, chose_dataset, Colors
from AnalisiDataset import dataset_overview_menu
from CustomEnsemble import ensemble_bagging_main
from DecisionTree import decision_tree_main
from NaiveBayes import naive_bayes_main
from CustomKNN import custom_KNN_main
from SVM import SVM_main

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
            dataset = chose_dataset()
            if dataset is not None:
                decision_tree_main(dataset)
        elif scelta == "3":
            dataset = chose_dataset()
            if dataset is not None:
                naive_bayes_main(dataset)
        elif scelta == "4":
            dataset = chose_dataset()
            if dataset is not None:
                SVM_main(dataset)
        elif scelta == "5":
            dataset = chose_dataset()
            if dataset is not None:
                custom_KNN_main(dataset)
        elif scelta == "6":
            dataset = chose_dataset()
            if dataset is not None:
                ensemble_bagging_main(dataset)
              
# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

def printLogo():
    '''Funzione per stampare il logo del progetto.'''
    
    print(f"▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄")
    print(f"▀▄     {Colors.BLUE}▄▄         ▄▄       ▄▄      {Colors.RESET} ▀▄")
    print(f"▀▄     {Colors.BLUE}███▄     ▄███       ██      {Colors.RESET} ▀▄")
    print(f"▀▄    {Colors.BLUE}██  ▀█▄ ▄█▀  ██      ██      {Colors.RESET} ▀▄")
    print(f"▀▄   {Colors.BLUE}██     ▀█▀     ██     ██      {Colors.RESET} ▀▄")
    print(f"▀▄  {Colors.BLUE}██               ██    ██▄▄▄▄▄▄{Colors.RESET} ▀▄")
    print(f"▀▄ {Colors.BLUE}▀▀                 ▀▀   ▀▀▀▀▀▀▀▀{Colors.RESET} ▀▄")
    print(f"▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄")
    print("\n")
    
# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

def printChoiches():
    '''Funzione per stampare le scelte possibili per l'utente.'''
    
    print("Scegliere un'opzione:")
    print(f"{Colors.BLUE}[1]{Colors.RESET} Analisi del dataset")
    print(f"{Colors.BLUE}[2]{Colors.RESET} Decision Tree")
    print(f"{Colors.BLUE}[3]{Colors.RESET} Naive Bayes")
    print(f"{Colors.BLUE}[4]{Colors.RESET} SVM")
    print(f"{Colors.BLUE}[5]{Colors.RESET} KNN (Custom)")
    print(f"{Colors.BLUE}[6]{Colors.RESET} Classificatore Multiplo (Custom)")
    print(f"{Colors.RED}[0]{Colors.RESET} Esci dal programma")

# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

if __name__ == '__main__':
    interactiveMenu()