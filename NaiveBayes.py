from Imports import np, accuracy_score, classification_report, cross_validate
from Shared_Utilities import chose_dataset, print_confusion_matrix, Colors
from sklearn.naive_bayes import GaussianNB

# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

def naive_bayes_main(dataset, var_smoothing = 0.001, votazione = "none", show_results = True):
    '''
    Funzione per addestrare il Naive Bayes in base al dataset scelto.
    
    Parametri:
    - dataset: tupla contenente i dati di training e di test già splittati.
    - var_smoothing: porzione di varianza massima aggiunta (default: 0.001).
    - votazione: tipologia di votazione da utilizzare
      - none: il modello non fa parte di un ensemble (default).
      - hard: il modello fa parte di un esemble e restituisce le sue predizioni.
      - soft: il modello fa parte di un esemble e restituisce le probabilità delle sue predizioni.
    '''
    
    X_train, X_test, y_train, y_test = dataset              # recupero dei dati
    
    NB = GaussianNB(var_smoothing=var_smoothing)            # inizializzazione del modello
    NB.fit(X_train, y_train)                                # addestramento del modello
    classi = NB.classes_                                    # estrapolazione delle classi

    y_pred = NB.predict(X_test)                             # previsioni sul test set
    accuracy = accuracy_score(y_test, y_pred)               # calcolo dell'accuratezza   
    
    if votazione == "hard":
        return accuracy, y_pred
    elif votazione == "soft":
        probabilità = NB.predict_proba(X_test)
        return accuracy, np.array([dict(zip(classi, probs)) for probs in probabilità])
        # restituisce l'accuratezza e un array di dizionari con le probabilità di appartenenza ad ogni classe
    elif show_results:
        report = classification_report(y_test, y_pred)      # report di classificazione

        print(f'{Colors.GREEN}Accuratezza{Colors.RESET}: {accuracy:.3}')
        print('\nReport sulle performance:')
        print(report)
        
        print_confusion_matrix(y_test, y_pred)              # stampa della matrice di confusione
        
        input(f"\nPremere {Colors.ORNG}INVIO{Colors.RESET} per continuare . . .")
    return accuracy

# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

def tuning_iperparametri():
    '''Funzione per il tuning degli iperparametri del Naive Bayes.'''
    
    # iperparametro da testare
    smoothing_values = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
    
    # menu interattivo per scelta del dataset da usare per il tuning
    X_train, _, y_train, _ = chose_dataset()
    
    # vettore per salvare tutti i risultati delle accuratezze
    accuracies = np.zeros(len(smoothing_values))
    
    # for per testare tutte le combinazioni di valori di smoothing
    for i, smoothing in enumerate(smoothing_values):
        
        NB = GaussianNB(var_smoothing=smoothing)
        
        all_scores = cross_validate(estimator=NB, X=X_train, y=y_train, cv=10, n_jobs=10)
        
        accuracies[i] = all_scores['test_score'].mean()
        
        print("Smoothing: {} - Accuratezza: {:.5f}".format(smoothing, accuracies[i]))
    
    print(f"\nMiglior {Colors.GREEN}Accuratezza{Colors.RESET}: %.5f (Usando smoothing: %s)" % (accuracies.max(), smoothing_values[np.argmax(accuracies)] ) )
    
# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

if __name__ == '__main__':
    tuning_iperparametri()