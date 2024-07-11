from Imports import pd, os, train_test_split, accuracy_score, classification_report, RANDOM_STATE
from Shared_Utilities import chose_dataset, clear_terminal
from sklearn.tree import DecisionTreeClassifier

# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

def decision_tree_main():
    '''Funzione per addestrare il DecisionTree in base al dataset scelto.'''
    
    X, y = chose_dataset() # permette di scegliere il dataset tramite un menu interattivo
    
    clear_terminal()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    
    clf = decision_tree(X_train, y_train)                   # addestramento del modello

    y_pred = clf.predict(X_test)                            # previsioni sul test set

    accuracy = accuracy_score(y_test, y_pred)               # calcolo dell'accuratezza
    report = classification_report(y_test, y_pred)          # report di classificazione

    print(f'Accuratezza: {accuracy:.2f}')
    print('\nReport sulle performance:')
    print(report)
    
    input("\nPremere un INVIO per continuare . . .")
    return
            
# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #
    
def decision_tree(train_x, train_y, depth=6):
    '''Funzione per creare e addestrare un modello Decision Tree.
    Restituisce l'albero decisionale già addestrato.'''
    
    dTree_clf = DecisionTreeClassifier(max_depth=depth, random_state=0)

    dTree_clf.fit(train_x, train_y)

    return dTree_clf

# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

def tuning_iperparametri():
    '''Funzione per il tuning degli iperparametri del Decision Tree.'''
    
    pass

# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

if __name__ == '__main__':
    decision_tree_main()