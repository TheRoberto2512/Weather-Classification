from Shared_Utilities import chose_dataset, print_confusion_matrix, Colors
from Imports import np, accuracy_score, classification_report
from DecisionTree import decision_tree_main
from NaiveBayes import naive_bayes_main
from CustomKNN import custom_KNN_main
from SVM import SVM_main

# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

def ensemble_bagging_main(dataset, votazione = "soft", pesata = True, bagging = False):
    '''
    Funzione per addestrare un classificatore composto con il metodo di bagging.
    
    Parametri:
    - dataset: tupla contenente i dati di training e di test già splittati.
    - votazione: tipologia di votazione da utilizzare.
    - pesata: se True, i contributi dei classificatori sono pesati in base all'accuratezza.
    - bagging: se True, la funzione restituisce l'accuratezza e le predizioni.
    '''
    
    _, _, _, y_test = dataset

    # prendiamo le predizioni effettuate da ogni classificatore per istanza
    accuracy_KNN, pred_KNN = custom_KNN_main(dataset = dataset, votazione = votazione)
    accuracy_NaiveBayes, pred_NaiveBayes = naive_bayes_main(dataset = dataset, votazione = votazione)
    accuracy_Decision_Tree, pred_Decision_Tree = decision_tree_main(dataset = dataset, votazione = votazione)
    accuracy_SVM, pred_SVM = SVM_main(dataset = dataset, votazione = votazione)

    # decidiamo quanto deve pesare il penso di ogni classificatore
    peso_KNN = (accuracy_KNN*100)**2
    peso_NaiveBayes = (accuracy_NaiveBayes*100)**2
    peso_Decision_Tree = (accuracy_Decision_Tree*100)**2
    peso_SVM = (accuracy_SVM*100)**2
    
    # calcoliamo il contributo di ogni classificatore in base alla propria accuratezza 
    if (pesata) & (votazione == "soft"):
        pred_KNN = np.array([{classe: prob * peso_KNN for classe, prob in istanza.items()} for istanza in pred_KNN])
        pred_NaiveBayes = np.array([{classe: prob * peso_NaiveBayes for classe, prob in istanza.items()} for istanza in pred_NaiveBayes])
        pred_Decision_Tree = np.array([{classe: prob * peso_Decision_Tree for classe, prob in istanza.items()} for istanza in pred_Decision_Tree])
        pred_SVM = np.array([{classe: prob * peso_SVM for classe, prob in istanza.items()} for istanza in pred_SVM])
        # si moltiplicano le probabilità di appartenenza ad ogni classe per il peso del classificatore
        
    pesi = [peso_KNN, peso_NaiveBayes, peso_Decision_Tree, peso_SVM] # pesi per la versione pesata dell'hard

    # assembliamo le varie predizioni, ottenendo una riga per ogni classificatore e in colonna le predizioni per istanza  
    pred_per_clf = np.array([pred_KNN, pred_NaiveBayes, pred_Decision_Tree, pred_SVM])

    # sommiamo la singola vitazione per istanza dei classificatori
    if votazione == "hard":
        pred_trasposte = pred_per_clf.T # trasponiamo la matrice per avere le predizioni per istanza
        pred_ensamble = [] # array che conterrà la predizione finale
        
        if pesata == False:
            for predizioni_per_istanza in pred_trasposte:
                # trova le classi uniche e conta le loro occorrenze
                unique, counts = np.unique(predizioni_per_istanza, return_counts=True)
                # trova la classe con il numero massimo di occorrenze
                predizione = unique[np.argmax(counts)]
                pred_ensamble.append(predizione)    # ricostruisco l'intero array di predizioni
        else:
            for predizioni_per_istanza in pred_trasposte:
                # crea un dizionario per accumulare i voti ponderati
                conteggio_voti = {}
                for pred, peso in zip(predizioni_per_istanza, pesi):
                    if pred in conteggio_voti:
                        conteggio_voti[pred] += peso
                    else:
                        conteggio_voti[pred] = peso
                # trova la classe con il punteggio totale ponderato più alto
                predizione = max(conteggio_voti, key=conteggio_voti.get)
                pred_ensamble.append(predizione) # ricostruisco l'intero array di predizioni
        
        pred_ensamble = np.array(pred_ensamble) # trasformo l'array in un numpy array
    
    elif votazione == "soft":
        # ottieni le etichette delle classi
        class_labels = list(pred_per_clf[0][0].keys())

        # numero di istanze
        n_istanze = len(pred_per_clf[0])

        # numero di classificatori
        n_clf = len(pred_per_clf)

        # inizializza un array per le somme delle probabilità
        somma_predizioni = np.zeros((n_istanze, len(class_labels)))

        # itera su ogni classificatore
        for clf in range(n_clf):
            # itera su ogni istanza
            for istanza in range(n_istanze):
                # ottieniamo le predizioni del classificatore per l'istanza corrente
                predizione = pred_per_clf[clf][istanza]
                # aggiungi le probabilità al totale
                for j, classe in enumerate(class_labels):
                    somma_predizioni[istanza][j] += predizione[classe]

        # calcola la media delle probabilità
        media_predizioni = somma_predizioni/n_clf

        # trova la classe con la probabilità media più alta per ogni istanza
        pred_ensamble = [class_labels[np.argmax(pred)] for pred in media_predizioni]

    accuracy = accuracy_score(y_test, pred_ensamble)       # calcolo dell'accuratezza
    if bagging == True: return accuracy
    report = classification_report(y_test, pred_ensamble)  # report di classificazione

    print(f'{Colors.GREEN}Accuratezza{Colors.RESET}: {accuracy:.3}')
    print('\nReport sulle performance:')
    print(report)
    
    print_confusion_matrix(y_test, pred_ensamble)          # stampa della matrice di confusione
    
    input(f"\nPremere {Colors.ORNG}INVIO{Colors.RESET} per continuare . . .")
    return None

# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

def tuning_iperparametri(dataset):
    '''
    Funzione per scegliere il migliore sistema di votazione per il classificatore composto custom.
    
    Parametri:
    - dataset: tupla contenente i dati di training e di test già splittati.
    '''
    
    # iperparametri da testare
    contributo_pesato = [True, False]
    tipi_votazione = ["hard", "soft"]

    # matrice per salvare tutti i risultati delle accuratezze
    acc_bagging = np.zeros((len(contributo_pesato), len(tipi_votazione)))

    # for annidati per testare tutte le combinazioni di iperparametri
    for j, tipo in enumerate(tipi_votazione):
        for i, contributo in enumerate(contributo_pesato):
            
            acc_bagging[i, j] = ensemble_bagging_main(dataset, votazione = tipo, pesata = contributo, bagging = True)
            # salva l'accuratezza del classificatore composto per ogni combinazione di iperparametri
            
            print("Tipo di votazione: \"%s\", contributo pesato: %s - Accuratezza: %.5f" % (tipo, contributo, acc_bagging[i,j]))
        
        print("\n")
    
    # indice della combinazione di iperparametri con l'accuratezza più alta
    i, j = np.unravel_index(np.argmax(acc_bagging, axis=None), acc_bagging.shape)
    print(f"Miglior {Colors.GREEN}Accuratezza{Colors.RESET}: %.5f (Usando tipo di votazione \"%s\" e con contributo pesato = %s)" % (acc_bagging[i,j], tipi_votazione[j], contributo_pesato[i]) )

# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

if __name__ == '__main__':
    tuning_iperparametri(dataset = chose_dataset())