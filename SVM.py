from Imports import np, accuracy_score, classification_report, cross_validate
from Shared_Utilities import chose_dataset, print_confusion_matrix
from sklearn.svm import SVC

# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

def SVM_main(dataset, C=1, kernel="linear", votazione = "none"):
    '''
    Funzione per addestrare il Naive Bayes in base al dataset scelto.
    
    Parametri:
    - dataset: tupla contenente i dati di training e di test già splittati.
    - C: parametro di regolarizzazione (default: 1).
    - kernel: tipo di kernel da utilizzare (default: linear).
    - votazione: tipologia di votazione da utilizzare
      - none: il modello non fa parte di un ensemble (default).
      - hard: il modello fa parte di un esemble e restituisce le sue predizioni.
      - soft: il modello fa parte di un esemble e restituisce le probabilità delle sue predizioni.
    '''
    
    X_train, X_test, y_train, y_test = dataset              # recupero dei dati

    SVM = SVC(C=C, kernel=kernel, probability=True)         # inizializzazione del modello
    classi = SVM.classes_
    SVM.fit(X_train, y_train)                               # addestramento del modello

    y_pred = SVM.predict(X_test)                            # previsioni sul test set
    accuracy = accuracy_score(y_test, y_pred)           # calcolo dell'accuratezza

    
    if votazione == "hard":
        return accuracy, y_pred
    elif votazione == "soft":
        probabilità = SVM.predict_proba(X_test)
        return accuracy, [dict(zip(classi, probs)) for probs in probabilità]
    else:
        report = classification_report(y_test, y_pred)      # report di classificazione

        print(f'Accuratezza: {accuracy:.3}')
        print('\nReport sulle performance:')
        print(report)
        
        print_confusion_matrix(y_test, y_pred)
        
        input("\nPremere INVIO per continuare . . .")
        return

# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

def tuning_iperparametri():
    '''Funzione per il tuning degli iperparametri dell'SVM.'''

    # iperparametri da testare
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    C_values = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    
    # menu interattivo per scelta del dataset da usare per il tuning
    X_train, _, y_train, _ = chose_dataset()
    
    # una matrice per salvare tutti i risultati delle accuratezze
    accuracies = np.zeros((len(kernels), len(C_values)))
    
    # for annidati per testare tutte le combinazioni di iperparametri
    for i, kernel in enumerate(kernels):
        for j, C in enumerate(C_values):
            
            SVM = SVC(C=C, kernel=kernel)
            
            all_scores = cross_validate(estimator=SVM, X=X_train, y=y_train, cv=10, n_jobs=10)
            
            accuracies[i][j] = all_scores['test_score'].mean()
            
            print("Kernel: {} e C: {} - Accuratezza: {:.5f}".format(kernel, C, accuracies[i][j]))
            
        print("\n")
        
    # indice della combinazione di iperparametri con l'accuratezza più alta
    i, j = np.unravel_index(np.argmax(accuracies, axis=None), accuracies.shape)
    print("\nMiglior accuratezza: %.5f (Usando kernel \"%s\" e C \"%s\")" % (accuracies[i][j], kernels[i], C_values[j]) )
    
# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

if __name__ == '__main__':
    tuning_iperparametri()