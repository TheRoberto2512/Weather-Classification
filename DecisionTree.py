from Imports import pd, os, np, train_test_split, accuracy_score, classification_report, cross_validate, RANDOM_STATE
from Shared_Utilities import chose_dataset, clear_terminal

from sklearn.tree import DecisionTreeClassifier
from Preprocessing import load_dataset

# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

def decision_tree_main(depth=10, criterion='entropy'):
    '''Funzione per addestrare il DecisionTree in base al dataset scelto.'''
    
    X, y = chose_dataset() # permette di scegliere il dataset tramite un menu interattivo
    
    clear_terminal()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    
    clf = DecisionTreeClassifier(max_depth=depth, criterion=criterion, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)                            # previsioni sul test set

    accuracy = accuracy_score(y_test, y_pred)               # calcolo dell'accuratezza
    report = classification_report(y_test, y_pred)          # report di classificazione

    print(f'Accuratezza: {accuracy:.3}')
    print('\nReport sulle performance:')
    print(report)
    
    input("\nPremere INVIO per continuare . . .")
    return
            
# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

def tuning_iperparametri():
    '''Funzione per il tuning degli iperparametri del Decision Tree.'''
    
    # gli iperparametri da testare sono criterio di split e la profondità dell'albero
    criterions = ['gini', 'entropy'] ; all_depths = [i for i in range(1, 26)]
    
    # dataset da caricare e splittare
    X, y = load_dataset(one_hot=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    
    # definisco una matrice per salvare tutti i risultati delle accuratezze
    accuracies = np.zeros((len(criterions), len(all_depths)))
    
    for i, criterion in enumerate(criterions):
        for j, depth in enumerate(all_depths):
            
            Tree = DecisionTreeClassifier(max_depth=depth, criterion=criterion, random_state=RANDOM_STATE)
            
            all_scores = cross_validate(estimator=Tree, X=X_train, y=y_train, cv=10, n_jobs=10)
            
            accuracies[i][j] = all_scores['test_score'].mean()
            
            print("Criterio: {} e profondità: {} - Accuratezza: {:.5f}".format(criterion, depth, accuracies[i][j]))
            
        print("\n")
    
    max_index = np.unravel_index(np.argmax(accuracies, axis=None), accuracies.shape)
    print("Miglior accuratezza: %.5f (Usando criterio \"%s\" e profondita' \"%s\")" % (accuracies[max_index[0]][max_index[1]], criterions[max_index[0]], all_depths[max_index[1]]) )

# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

if __name__ == '__main__':
    tuning_iperparametri()