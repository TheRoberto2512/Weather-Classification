from Imports import np, accuracy_score, classification_report, cross_validate
from Shared_Utilities import chose_dataset
from sklearn.naive_bayes import GaussianNB

# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

def naive_bayes_main(dataset, var_smoothing = 0.001, votazione = "hard", ensemble = False):
    '''Funzione per addestrare il Naive Bayes in base al dataset scelto.'''
    
    X_train, X_test, y_train, y_test = dataset 
    
    NB = GaussianNB(var_smoothing=var_smoothing)
    NB.fit(X_train, y_train)

    y_pred = NB.predict(X_test)                     # previsioni sul test set
    if ensemble & (votazione == "hard"):
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
    
    X_train, _, y_train, _ = chose_dataset()
    
    smoothing_values = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    accuracies = np.zeros(len(smoothing_values))
    
    for i, smoothing in enumerate(smoothing_values):
        NB = GaussianNB(var_smoothing=smoothing)
        
        all_scores = cross_validate(estimator=NB, X=X_train, y=y_train, cv=10, n_jobs=10)
        
        accuracies[i] = all_scores['test_score'].mean()
        
        print("Smoothing: {} - Accuratezza: {:.5f}".format(smoothing, accuracies[i]))
    
    print("\nMiglior accuratezza: %.5f (Usando smoothing: %s)" % (accuracies.max(), smoothing_values[np.argmax(accuracies)] ) )
    
# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

if __name__ == '__main__':
    tuning_iperparametri()