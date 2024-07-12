from Preprocessing import load_dataset
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


from sklearn.metrics.pairwise import euclidean_distances,manhattan_distances
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score


"""
def tuningCustomKnn(X_train, y_train, X_test, y_test):
    # Definiamo il range di valori k e la misura di distanza da adottare
    valori_k = [k for k in range(1, 25)]
    misure_distanze = ["Euclidea", "Manhattan"]

    # Inizializzamo array per memorizzare le varie accuratezze durante il tuning
    acc_val = np.empty((len(valori_k), len(misure_distanze)))

    for i, k in enumerate(valori_k):
        for j, misura_distanza in enumerate(misure_distanze):
                custom_knn = CustomKNN(k, misura_distanza)
                cv_accuratezze = cross_validate(estimator=custom_knn, X=X_train, y=y_train, cv=3, n_jobs=6)
                # Memorizza le accuratezze medie nei rispettivi array
                acc_val[i, j] = cv_accuratezze['test_score'].mean()
                print("\nAggiunta la media delle accuratezze per k uguale a %d e misura di distanza %s:\t%.5f"%(k, misura_distanza, acc_val[i,j]))
    i, j = np.unravel_index(np.argmax(acc_val, axis=None), acc_val.shape)
    k_migliore = valori_k[i]
    misura_distanza_migliore = misure_distanze[j]
    print("La miglior accuratezza è %.5f ed è stata ottenuta con la distanza di %s e con k uguale a %d\n"% (acc_val[i,j], misura_distanza_migliore, k_migliore))
"""

###IMPOSTATO COI VALORI MIGLIORI TROVATI COL TUNING
def customKNNClassifier(X_train, y_train, X_test, y_test, k=3, misura_distanza = "Manhattan"):
    # Inizializo il modello
    custom_knn = CustomKNN(k, misura_distanza)

    # Addestro il modello
    custom_knn.fit(X_train, y_train)

    # Eseguo le predizioni
    predizioni = custom_knn.predict(X_test)
    accuracy = custom_knn.score(X_test, y_test)
    print("%.5f"%(accuracy))


  

#classe custom del knn che permette di effettuare le previsioni pesando il contributo di ogni vicino in base alla sua vicinanza al nuovo record
class CustomKNN(BaseEstimator):
    """
    BaseEstimator e ClassifierMixin utilizzati come superclassi forniscono
    una struttura e un'implementazione che rendono il modello 
    compatibile col cross_validate di scikit-learn.
    """
    #nizializza il valore di k
    def __init__(self, k=3, misura_distanza = "Euclidea"):
        self.k = k
        self.misura_distanza = misura_distanza
        self.fittato=False

    #metodo che addestra il modello memorizzando il dataset
    def fit(self, X, y):
        self.mX=X
        self.my=y
        self.classi = np.unique(y)  # Estraggo le classi presenti nel dataset di training
        self.fittato=True
    
    #metodo che effettua le previsioni dei record del test set X
    def predict(self, X):
        #effettuo la previsione solo se il modello è già stato addestrato 
        if self.fittato:
            #creto una np array che conterrà il target previsto per ogni record
            (istanze,attributi)=X.shape
            pred_y=np.zeros(istanze, dtype=object)
            X=X.values

            #ciclo che scorre i record di X e effettua la previsione per ciascun record inserisce il valore pedetto nella lista pred_y
            for istanza in range(istanze):
                # Reshape della istanza per essere un 2D array
                istanza_reshaped = X[istanza].reshape(1, -1)
                #calcolo la distanza del record corrente dai record del training set
                if self.misura_distanza == "Euclidea":
                    distanze = euclidean_distances(self.mX, istanza_reshaped).flatten()
                elif self.misura_distanza == "Manhattan":
                    distanze = manhattan_distances(self.mX, istanza_reshaped).flatten()
                else:
                    return print("\nLa misura di distanza richiesta non è supportata\n")
                #prendo gli indici delle k istanze più vicine
                indici=np.argsort(distanze)[:self.k]
                #prendo le distanze delle k istanze più vicine 
                k_distanze = distanze[indici]
                #calcolo i pesi come l'inverso della distanza
                pesi = [(1/d) for d in k_distanze]
                #prendo i target dei k record più vicini
                k_vicini=self.my.values[indici]
                #somma delle distanze pesate per classe
                somma_per_classe = {classe: 0 for classe in self.classi}                
                #calcolo la somma pesata per classe
                for vicino, peso in zip(k_vicini, pesi):
                    somma_per_classe[vicino] += peso
                #prevedo la classe con la somma pesata maggiore
                pred_y[istanza] = max(somma_per_classe, key=somma_per_classe.get)
            
            return pred_y
        else:
            print("E' prima necessario addestrare il modello")
        
    #metodo che calcola l'accuracy del modello
    def score(self, X_test, y_test):
        if self.fittato:
           #DIPENDE DA COME VOGLIAMO VALUTARE IL MODELLO
            pred_y=self.predict(X_test)
            return accuracy_score(y_test, pred_y)
        else:
            print("E' prima necessario addestrare il modello")

# DA RIMUOVERE
if __name__ == '__main__':
    print("\nInizio codice\n")
    X, y = load_dataset()
    train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=0, test_size=0.25)
    #PER TROVARE I MIGLIORI IPERPARAMETRI
    #tuningCustomKnn(train_X, train_y, test_X, test_y)

    #PER TESTARE IL CLASSIFICATORI CON I MIGLIORI IPERPARAMETRI TROVATI COL TUNING
    customKNNClassifier(train_X, train_y, test_X, test_y, k=7, misura_distanza = "Manhattan")

    print("\n\nFine codice\n\n")
