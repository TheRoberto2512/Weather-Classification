from Imports import pd, os, np, accuracy_score, classification_report
from customKNN import custom_KNN_main
from DecisionTree import decision_tree_main
from NaiveBayes import naive_bayes_main
from SVM import SVM_main

def ensemble_bagging_main(dataset, votazione = "hard", pesata = True, bagging = False):
    
    _, _, _, y_test = dataset

    #Prendiamo le predizioni effettuate da ogni classificatore per istanza
    pred_KNN = custom_KNN_main(dataset = dataset, votazione = votazione, ensemble = True)
    pred_NaiveBayes = naive_bayes_main(dataset = dataset, votazione = votazione, ensemble = True)
    pred_Decision_Tree = decision_tree_main(dataset = dataset, votazione = votazione, ensemble = True)
    pred_SVM = SVM_main(dataset = dataset, votazione = votazione, ensemble = True)

    #Calcoliamo il contributo di ogni classificatore in base alla propria accuratezza 
    if(pesata):
        pred_KNN *= accuracy_score(y_test, pred_KNN)
        pred_NaiveBayes *= accuracy_score(y_test, pred_NaiveBayes)
        pred_Decision_Tree *= accuracy_score(y_test, pred_Decision_Tree)
        pred_SVM *= accuracy_score(y_test, pred_SVM)

    #Assembliamo le varie predizioni, ottenendo una riga per ogni classificatore e in colonna le predizioni per istanza  
    pred_per_clf = np.array([pred_KNN, pred_NaiveBayes, pred_Decision_Tree, pred_SVM])

    #Sommiamo la singola vitazione per istanza dei classificatori
    if votazione == "hard":
        pred_trasposte = pred_per_clf.T
        pred_ensamble = [] #l'array che conterrà la predizione finale
        for predizioni_per_istanza in pred_trasposte:
            # Trova le classi uniche e conta le loro occorrenze
            unique, counts = np.unique(predizioni_per_istanza, return_counts=True)
            # Trova la classe con il numero massimo di occorrenze
            predizione = unique[np.argmax(counts)]
            pred_ensamble.append(predizione)    #ricostruisco l'intero array di predizioni
        pred_ensamble = np.array(pred_ensamble)
    
    elif votazione == "soft":
        # Calcola la media delle probabilità lungo l'asse 0 (classificatori)
        media_predizioni = np.mean(pred_per_clf, axis=0)
        # Trova la classe con la probabilità media più alta per ogni istanza
        pred_ensamble = np.argmax(media_predizioni, axis=1)

    accuracy = accuracy_score(y_test, pred_ensamble)       # calcolo dell'accuratezza
    if bagging == True: return accuracy
    report = classification_report(y_test, pred_ensamble)  # report di classificazione

    print(f'Accuratezza: {accuracy:.3}')
    print('\nReport sulle performance:')
    print(report)
    
    input("\nPremere INVIO per continuare . . .")
    return

# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

def tuning_bagging(dataset):
    '''Funzione per scegliere il migliore sistema di votazione per il classificatore composto custom.'''
    
    # Definiamo il range di valori k e la misura di distanza da adottare
    contributo_pesato = [True, False]
    tipi_votazione = ["Hard", "Soft"]

    # Inizializzamo array per memorizzare le varie accuratezze durante il tuning
    acc_bagging = np.empty((len(contributo_pesato), len(tipi_votazione)))

    for j, tipo in enumerate(tipi_votazione):
        for i, contributo in enumerate(contributo_pesato):
                acc_bagging[i, j] = ensemble_bagging_main(dataset, votazione = tipo, pesata = contributo, bagging = True)
                print("Tipo di votazione: %s, contributo_pesato = %b - Accuratezza: %.5f"%(tipo, contributo, acc_bagging[i,j]))
        print("\n")
    
    i, j = np.unravel_index(np.argmax(acc_bagging, axis=None), acc_bagging.shape)
    print("Miglior accuratezza: %.5f (Usando tipo di votazione \"%s\" e con contributo pesato = %b)" % (acc_bagging[i,j], tipi_votazione[j], contributo_pesato[i]) )

# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

if __name__ == '__main__':
    tuning_bagging()