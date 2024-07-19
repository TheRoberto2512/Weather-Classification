from Shared_Utilities import clear_terminal, chose_dataset, Colors
from DecisionTree import decision_tree_main
from CustomEnsemble import ensemble_main
from NaiveBayes import naive_bayes_main
from CustomKNN import custom_KNN_main
from SVM import SVM_main
from Imports import plt

# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

def confronti():
    '''Funzione per confrontare i risultati dei vari classificatori.'''
    
    nome1, acc1 = chose_model(print_results=False)
    if nome1 == "NoChoice":
        return
    
    nome2, acc2 = chose_model(print_results=False)
    if nome2 == "NoChoice":
        return    
    
    plot_data(nome1, nome2, acc1, acc2)

# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

def chose_model(print_results=True):
    '''
    Funzione per scegliere e addestrare un modello.
    
    Parametri:
    - print_results: se True, stampa i risultati del classificatore (default = True).
    '''
    
    scelta = -1 # scelta dell'utente
    
    while scelta != "scelto":
        clear_terminal()
        print_model_choices()
        scelta = input()
        
        if scelta == "0":
            return "NoChoice", 0
        
        elif scelta == "1":
            dataset, dataset_name = chose_dataset(return_name=True)
            if dataset is not None:
                acc = decision_tree_main(dataset, show_results=print_results)
                name = "Decision Tree (" + dataset_name + ")"
                return name, acc
            
        elif scelta == "2":
            dataset, dataset_name = chose_dataset(return_name=True)
            if dataset is not None:
                acc = naive_bayes_main(dataset, show_results=print_results)
                name = "Naive Bayes (" + dataset_name + ")"
                return name, acc
            
        elif scelta == "3":
            dataset, dataset_name = chose_dataset(return_name=True)
            if dataset is not None:
                acc =  SVM_main(dataset, show_results=print_results)
                name = "SVM (" + dataset_name + ")"
                return name, acc
            
        elif scelta == "4":
            dataset, dataset_name = chose_dataset(return_name=True)
            if dataset is not None:
                acc =  custom_KNN_main(dataset, show_results=print_results)
                name = "KNN (" + dataset_name + ")"
                return name, acc
            
        elif scelta == "5":
            dataset, dataset_name = chose_dataset(return_name=True)
            if dataset is not None:
                acc =  ensemble_main(dataset, votazione="hard", show_results=print_results)
                name = "Multiplo (" + dataset_name + ")"
                return name, acc

# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

def print_model_choices():
    '''Funzione per stampare la lista dei modelli disponibili.'''
        
    print("Scegli un modello:")
    print(f"{Colors.BLUE}[1]{Colors.RESET} Decision Tree")
    print(f"{Colors.BLUE}[2]{Colors.RESET} Naive Bayes")
    print(f"{Colors.BLUE}[3]{Colors.RESET} SVM")
    print(f"{Colors.BLUE}[4]{Colors.RESET} KNN (Custom)")
    print(f"{Colors.BLUE}[5]{Colors.RESET} Classificatore Multiplo (Custom)")
    print(f"{Colors.ORNG}[0]{Colors.RESET} Torna indietro")

# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

def plot_data(nome1, nome2, acc1, acc2):
    '''
    Funzione per plottare i risultati dei classificatori.
    
    Parametri:
    - nome1: string, nome del primo classificatore.
    - nome2: string, nome del secondo classificatore.
    - acc1: float, accuratezza del primo classificatore.
    - acc2: float, accuratezza del secondo classificatore.
    '''
    
    bars = plt.bar([nome1, nome2], [acc1, acc2], color=["#0070c0", "#ed7d31"])
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, round(yval, 3), va='bottom', ha='center', fontweight="bold")
    
    plt.title("%s VS %s" % (nome1, nome2), fontweight="bold")
    plt.xlabel("Classificatori")
    plt.ylabel("Accuratezza")
    plt.ylim([0, 1.05])
    plt.show()

# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #