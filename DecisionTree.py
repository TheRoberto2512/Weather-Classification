from Imports import np, accuracy_score, classification_report, cross_validate, RANDOM_STATE
from sklearn.tree import DecisionTreeClassifier
from Shared_Utilities import chose_dataset, print_confusion_matrix

# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

def decision_tree_main(dataset, depth=10, criterion="entropy", votazione="none"):
    '''
    Funzione per addestrare il DecisionTree in base al dataset scelto.
    
    Parametri:
    - dataset: tupla contenente i dati di training e di test già splittati.
    - depth: profondità dell'albero (default: 10).
    - criterion: criterio di split (default: entropy).
    - votazione: tipologia di votazione da utilizzare
      - none: il modello non fa parte di un ensemble (default).
      - hard: il modello fa parte di un esemble e restituisce le sue predizioni.
      - soft: il modello fa parte di un esemble e restituisce le probabilità delle sue predizioni.
    '''
    
    X_train, X_test, y_train, y_test = dataset              # recupero dei dati
    
    clf = DecisionTreeClassifier(max_depth=depth, criterion=criterion, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)                               # addestramento del modello    

    y_pred = clf.predict(X_test)                            # previsioni sul test set
       
    if votazione == "hard":
        return y_pred
    elif votazione == "soft":
        return clf.predict_proba(X_test)
    else:
        accuracy = accuracy_score(y_test, y_pred)           # calcolo dell'accuratezza
        report = classification_report(y_test, y_pred)      # report di classificazione

        print(f'Accuratezza: {accuracy:.3}')
        print('\nReport sulle performance:')
        print(report)
        
        print_confusion_matrix(y_test, y_pred)
        
        input("\nPremere INVIO per continuare . . .")
        return
            
# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

def tuning_iperparametri():
    '''Funzione per il tuning degli iperparametri del Decision Tree.'''
    
    # iperparametri da testare
    criterions = ['gini', 'entropy'] ; all_depths = [i for i in range(1, 26)]

    # menu interattivo per scelta del dataset da usare per il tuning
    X_train, _, y_train, _ = chose_dataset()
    
    # matrice per salvare tutti i risultati delle accuratezze
    accuracies = np.zeros((len(criterions), len(all_depths)))
    
    # for annidati per testare tutte le combinazioni di iperparametri
    for i, criterion in enumerate(criterions):
        for j, depth in enumerate(all_depths):
            
            Tree = DecisionTreeClassifier(max_depth=depth, criterion=criterion, random_state=RANDOM_STATE)
            
            all_scores = cross_validate(estimator=Tree, X=X_train, y=y_train, cv=10, n_jobs=10)
            
            accuracies[i][j] = all_scores['test_score'].mean()
            
            print("Criterio: \"{}\", profondità: {} - Accuratezza: {:.5f}".format(criterion, depth, accuracies[i][j]))
            
        print("\n")
    
    # indice della combinazione di iperparametri con l'accuratezza più alta
    i,j = np.unravel_index(np.argmax(accuracies, axis=None), accuracies.shape)
    print("Miglior accuratezza: %.5f (Usando criterio \"%s\" e profondita' \"%s\")" % (accuracies[i][j], criterions[i], all_depths[j]) )

# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

if __name__ == '__main__':
    tuning_iperparametri()