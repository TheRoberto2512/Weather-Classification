from Imports import np, accuracy_score, cross_validate, classification_report
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
from Shared_Utilities import chose_dataset, print_confusion_matrix
from sklearn.base import BaseEstimator

# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

def custom_KNN_main(dataset, k=7, misura_distanza="Manhattan", votazione="none"):
    '''
    Funzione per addestrare il KNN in base al dataset scelto.
    
    Parametri:
    - dataset: tupla contenente i dati di training e di test già splittati.
    - k: numero di vicini da considerare (default: 7).
    - misura_distanza: tipo di distanza da utilizzare (default: Manhattan).
    - votazione: tipologia di votazione da utilizzare
      - none: il modello non fa parte di un ensemble (default).
      - hard: il modello fa parte di un esemble e restituisce le sue predizioni.
      - soft: il modello fa parte di un esemble e restituisce le probabilità delle sue predizioni.
    '''
    
    X_train, X_test, y_train, y_test = dataset              # recupero dei dati
    
    custom_knn = CustomKNN(k, misura_distanza)              # inizializzazione del modello
    custom_knn.fit(X_train, y_train)                        # addestramento del modello 

    if votazione != "none": 
        return custom_knn.predict(X_test, votazione)        # previsioni sul test set calcolate per l'ensemble
    else:
        y_pred = custom_knn.predict(X_test)                 # previsioni sul test set
        
        accuracy = accuracy_score(y_test, y_pred)           # calcolo dell'accuratezza
        report = classification_report(y_test, y_pred)      # report di classificazione

        print(f'Accuratezza: {accuracy:.3}')
        print('\nReport sulle performance:')
        print(report)
        
        print_confusion_matrix(y_test, y_pred)              # stampa della matrice di confusione
        
        input("\nPremere INVIO per continuare . . .")
        return

# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #
  
class CustomKNN(BaseEstimator):
    '''
    BaseEstimator utilizzati come superclassi forniscono una struttura e un'implementazione
    che rendono il modello compatibile col cross_validate di scikit-learn. '''
    
    def __init__(self, k=3, misura_distanza="Euclidea"):
        ''''Metodo costruttore del modello KNN custom.
        
        Parametri:
        - k: numero di vicini da considerare (default: 7).
        - misura_distanza: tipo di distanza da utilizzare (default: Manhattan).
        '''
        self.k = k
        self.misura_distanza = misura_distanza
        self.fittato=False

    def fit(self, X, y):
        '''
        Metodo per addestrare il modello KNN.
        
        Parametri:
        X: array di training.
        y: array di target.
        '''
        
        self.mX = X 
        self.my = y
        self.classi = np.unique(y)  # estraggo le classi presenti nel dataset di training
        self.fittato = True
    
    def predict(self, X, votazione = "hard"):
        '''
        Metodo per effettuare le previsioni sui dati forniti.
        
        Parametri:
        X: array di training.
        votazione: tipo di votazione da utilizzare (default: hard).
        '''
        
        # effettuo la previsione solo se il modello è già stato addestrato 
        if self.fittato:
            # creo una np array che conterrà il target previsto per ogni record
            (istanze,attributi)=X.shape
            pred_y=np.zeros(istanze, dtype=object)
            X=X.values

            # ciclo che scorre i record di X e effettua la previsione per ciascun record inserisce il valore pedetto nella lista pred_y
            for istanza in range(istanze):
                # reshape dell'istanza per essere un 2D array
                istanza_reshaped = X[istanza].reshape(1, -1)
                
                # calcolo la distanza del record corrente dai record del training set
                if self.misura_distanza == "Euclidea":
                    distanze = euclidean_distances(self.mX, istanza_reshaped).flatten()
                elif self.misura_distanza == "Manhattan":
                    distanze = manhattan_distances(self.mX, istanza_reshaped).flatten()
                else:
                    return print("\nLa misura di distanza richiesta non è supportata\n")
                
                # prendo gli indici delle k istanze più vicine
                indici=np.argsort(distanze)[:self.k]
                
                # prendo le distanze delle k istanze più vicine 
                k_distanze = distanze[indici]
                
                # calcolo i pesi come l'inverso della distanza
                pesi = [(1/d) for d in k_distanze]
                
                # prendo i target dei k record più vicini
                k_vicini=self.my.values[indici]
                
                # somma delle distanze pesate per classe
                somma_per_classe = {classe: 0 for classe in self.classi} 
                               
                # calcolo la somma pesata per classe
                for vicino, peso in zip(k_vicini, pesi):
                    somma_per_classe[vicino] += peso
                    
                # prevedo la classe con la somma pesata maggiore
                if votazione == "hard":
                    pred_y[istanza] = max(somma_per_classe, key=somma_per_classe.get)
                elif votazione == "soft":
                    pred_y[istanza] = somma_per_classe
            
            return pred_y
        else:
            print("E' prima necessario addestrare il modello")
        
    def score(self, X_test, y_test, votazione = "hard"):
        '''
        Metodo per calcolare l'accuratezza del modello.
        
        Parametri:
        - X: array di test.
        - y: array di target.
        - votazione: tipo di votazione da utilizzare (default: hard).
        '''
        
        if self.fittato:
            pred_y = self.predict(X_test, votazione)
            return accuracy_score(y_test, pred_y)
        else:
            print("E' prima necessario addestrare il modello")

# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

def tuning_iperparametri():
    '''Funzione per il tuning degli iperparametri del KNN custom.'''
    
    # iperparametri da testare
    valori_k = [k for k in range(1, 26)]
    misure_distanze = ["Euclidea", "Manhattan"]

    # menu interattivo per scelta del dataset da usare per il tuning
    X_train, _, y_train, _ = chose_dataset()

    # inizializzamo array per memorizzare le varie accuratezze durante il tuning
    acc_val = np.empty((len(valori_k), len(misure_distanze)))

    # for annidati per testare tutte le combinazioni di iperparametri
    for j, misura_distanza in enumerate(misure_distanze):
        for i, k in enumerate(valori_k):
            
                custom_knn = CustomKNN(k, misura_distanza)
                
                cv_accuratezze = cross_validate(estimator=custom_knn, X=X_train, y=y_train, cv=10, n_jobs=10)
                
                acc_val[i, j] = cv_accuratezze['test_score'].mean()
                
                print("Distanza: \"%s\", k = %d - Accuratezza: %.5f"%(misura_distanza, k, acc_val[i,j]))
                
        print("\n")
    
    # indice della combinazione di iperparametri con l'accuratezza più alta
    i, j = np.unravel_index(np.argmax(acc_val, axis=None), acc_val.shape)
    print("Miglior accuratezza: %.5f (Usando distanza di \"%s\" e con k = %d)" % (acc_val[i,j], misure_distanze[j], valori_k[i]) )

# -- -- # -- -- # -- -- # -- -- # -- -- # -- -- # -- -- #

if __name__ == '__main__':
    tuning_iperparametri()
