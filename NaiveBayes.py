from Imports import pd, os, np, train_test_split, accuracy_score, classification_report, cross_validate, RANDOM_STATE
from Shared_Utilities import chose_dataset, clear_terminal

from sklearn.naive_bayes import GaussianNB
from Preprocessing import load_dataset

# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

def naive_bayes_main(var_smoothing = 1e-9, votazione = "hard", ensemble = False):
    '''Funzione per addestrare il Naive Bayes in base al dataset scelto.'''
    
    X, y = chose_dataset() # permette di scegliere il dataset tramite un menu interattivo
    
    clear_terminal()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    
    NB = GaussianNB(var_smoothing=var_smoothing)
    NB.fit(X_train, y_train)

    y_pred = NB.predict(X_test)                     # previsioni sul test set
    if ensemble & votazione == "hard":
        return y_pred
    elif ensemble:
        return NB.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)       # calcolo dell'accuratezza
    report = classification_report(y_test, y_pred)  # report di classificazione

    print(f'Accuratezza: {accuracy:.3}')
    print('\nReport sulle performance:')
    print(report)
    
    input("\nPremere INVIO per continuare . . .")
    return

# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

def tuning_iperparametri():
    '''Funzione per il tuning degli iperparametri del Naive Bayes.'''
    
    X, y = load_dataset(one_hot=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    
    smoothing_values = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
    accuracies = np.zeros(len(smoothing_values))
    
    for i, smoothing in enumerate(smoothing_values):
        NB = GaussianNB(var_smoothing=smoothing)
        
        all_scores = cross_validate(estimator=NB, X=X_train, y=y_train, cv=10, n_jobs=10)
        
        accuracies[i] = all_scores['test_score'].mean()
        
        print("Smoothing: {} - Accuratezza: {:.5f}".format(smoothing, accuracies[i]))
    
    print("\nMiglior accuratezza: %.5f (Usando smoothing: %s)" % (accuracies.max(), smoothing_values[np.argmax(accuracies.max())]) )
    

# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

if __name__ == '__main__':
    tuning_iperparametri()