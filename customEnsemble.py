from Shared_Utilities import chose_dataset, print_confusion_matrix
from Imports import np, accuracy_score, classification_report
from DecisionTree import decision_tree_main
from NaiveBayes import naive_bayes_main
from CustomKNN import custom_KNN_main
from SVM import SVM_main

# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

def ensemble_bagging_main(dataset, votazione = "hard", pesata = True, bagging = False):
    '''
    Funzione per addestrare un classificatore composto con il metodo di bagging.
    
    Parametri:
    - dataset: tupla contenente i dati di training e di test già splittati.
    - votazione: tipologia di votazione da utilizzare.
    - pesata: se True, i contributi dei classificatori sono pesati in base all'accuratezza.
    - bagging: se True, la funzione restituisce l'accuratezza del classificatore composto.
    '''
    
    _, _, _, y_test = dataset

    # prendiamo le predizioni effettuate da ogni classificatore per istanza
    pred_KNN = custom_KNN_main(dataset = dataset, votazione = votazione)
    pred_NaiveBayes = naive_bayes_main(dataset = dataset, votazione = votazione)
    pred_Decision_Tree = decision_tree_main(dataset = dataset, votazione = votazione)
    pred_SVM = SVM_main(dataset = dataset, votazione = votazione)

    # calcoliamo il contributo di ogni classificatore in base alla propria accuratezza 
    if (pesata) & (votazione == "soft"):
        pred_KNN *= accuracy_score(y_test, pred_KNN)
        pred_NaiveBayes *= accuracy_score(y_test, pred_NaiveBayes)
        pred_Decision_Tree *= accuracy_score(y_test, pred_Decision_Tree)
        pred_SVM *= accuracy_score(y_test, pred_SVM)
        
    pesi = [accuracy_score(y_test, pred_KNN), accuracy_score(y_test, pred_NaiveBayes), accuracy_score(y_test, pred_Decision_Tree), accuracy_score(y_test, pred_SVM)] 

    # assembliamo le varie predizioni, ottenendo una riga per ogni classificatore e in colonna le predizioni per istanza  
    pred_per_clf = np.array([pred_KNN, pred_NaiveBayes, pred_Decision_Tree, pred_SVM])

    # sommiamo la singola vitazione per istanza dei classificatori
    if votazione == "hard":
        pred_trasposte = pred_per_clf.T
        pred_ensamble = [] #l'array che conterrà la predizione finale
        
        if pesata == False:
            for predizioni_per_istanza in pred_trasposte:
                # trova le classi uniche e conta le loro occorrenze
                unique, counts = np.unique(predizioni_per_istanza, return_counts=True)
                # trova la classe con il numero massimo di occorrenze
                predizione = unique[np.argmax(counts)]
                pred_ensamble.append(predizione)    #ricostruisco l'intero array di predizioni
        else:
            for instance_preds in pred_trasposte:
                # crea un dizionario per accumulare i voti ponderati
                vote_counts = {}
                for pred, weight in zip(instance_preds, pesi):
                    if pred in vote_counts:
                        vote_counts[pred] += weight
                    else:
                        vote_counts[pred] = weight
                # trova la classe con il punteggio totale ponderato più alto
                predizione = max(vote_counts, key=vote_counts.get)
                pred_ensamble.append(predizione)
        
        pred_ensamble = np.array(pred_ensamble)
    
    elif votazione == "soft":
        # calcola la media delle probabilità lungo l'asse 0 (classificatori)
        media_predizioni = np.mean(pred_per_clf, axis=0)
        # trova la classe con la probabilità media più alta per ogni istanza
        pred_ensamble = np.argmax(media_predizioni, axis=1)

    accuracy = accuracy_score(y_test, pred_ensamble)       # calcolo dell'accuratezza
    if bagging == True: return accuracy
    report = classification_report(y_test, pred_ensamble)  # report di classificazione

    print(f'Accuratezza: {accuracy:.3}')
    print('\nReport sulle performance:')
    print(report)
    
    print_confusion_matrix(y_test, pred_ensamble)
    
    input("\nPremere INVIO per continuare . . .")
    return

# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

def tuning_iperparametri(dataset):
    '''Funzione per scegliere il migliore sistema di votazione per il classificatore composto custom.'''
    
    # iperparametri da testare
    contributo_pesato = [True, False]
    tipi_votazione = ["hard", "soft"]

    # matrice per salvare tutti i risultati delle accuratezze
    acc_bagging = np.zeros((len(contributo_pesato), len(tipi_votazione)))

    # for annidati per testare tutte le combinazioni di iperparametri
    for j, tipo in enumerate(tipi_votazione):
        for i, contributo in enumerate(contributo_pesato):
            
            acc_bagging[i, j] = ensemble_bagging_main(dataset, votazione = tipo, pesata = contributo, bagging = True)
            
            print("Tipo di votazione: \"%s\", contributo pesato: %s - Accuratezza: %.5f" % (tipo, contributo, acc_bagging[i,j]))
        
        print("\n")
    
    # indice della combinazione di iperparametri con l'accuratezza più alta
    i, j = np.unravel_index(np.argmax(acc_bagging, axis=None), acc_bagging.shape)
    print("Miglior accuratezza: %.5f (Usando tipo di votazione \"%s\" e con contributo pesato = %b)" % (acc_bagging[i,j], tipi_votazione[j], contributo_pesato[i]) )

# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

if __name__ == '__main__':
    tuning_iperparametri(chose_dataset())