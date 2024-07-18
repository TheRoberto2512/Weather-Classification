from Shared_Utilities import clear_terminal, Colors
from AnalisiDataset import dataset_overview_menu
from Models import confronti, chose_model

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
            chose_model()
        elif scelta == "3":
            confronti()
                     
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
    print(f"{Colors.BLUE}[2]{Colors.RESET} Addestramento di un classificatore")
    print(f"{Colors.BLUE}[3]{Colors.RESET} Confronto tra classificatori")
    print(f"{Colors.RED}[0]{Colors.RESET} Esci dal programma")

# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

if __name__ == '__main__':
    interactiveMenu()
    
''''''