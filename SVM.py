from Imports import np, accuracy_score, classification_report, cross_validate
from Shared_Utilities import chose_dataset
from sklearn.svm import SVC

# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

def SVM_main(dataset, votazione = "hard", ensemble = False):
    '''Funzione per addestrare il Naive Bayes in base al dataset scelto.'''
    
    X_train, X_test, y_train, y_test = dataset 
    
    SVM = SVC(C=1, kernel="linear")
    SVM.fit(X_train, y_train)

    y_pred = SVM.predict(X_test)                    # previsioni sul test set
    if ensemble & (votazione == "hard"):
        return y_pred
    elif ensemble:
        return SVM.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)       # calcolo dell'accuratezza
    report = classification_report(y_test, y_pred)  # report di classificazione

    print(f'Accuratezza: {accuracy:.3}')
    print('\nReport sulle performance:')
    print(report)
    
    input("\nPremere INVIO per continuare . . .")
    return

# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

def tuning_iperparametri():
    '''Funzione per il tuning degli iperparametri dell'SVM.'''
    
    X_train, _, y_train, _ = chose_dataset()

    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    C_values = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    
    accuracies = np.zeros((len(kernels), len(C_values)))
    
    for i, kernel in enumerate(kernels):
        for j, C in enumerate(C_values):
            SVM = SVC(C=C, kernel=kernel)
            
            all_scores = cross_validate(estimator=SVM, X=X_train, y=y_train, cv=10, n_jobs=10)
            
            accuracies[i][j] = all_scores['test_score'].mean()
            
            print("Kernel: {} e C: {} - Accuratezza: {:.5f}".format(kernel, C, accuracies[i][j]))
            
        print("\n")
        
    i, j = np.unravel_index(np.argmax(accuracies, axis=None), accuracies.shape)
    print("\nMiglior accuratezza: %.5f (Usando kernel \"%s\" e C \"%s\")" % (accuracies[i][j], kernels[i], C_values[j]) )
    
# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

if __name__ == '__main__':
    tuning_iperparametri()