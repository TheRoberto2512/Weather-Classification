from Preprocessing import load_dataset
import numpy as np

from sklearn.metrics.pairwise import euclidean_distances,manhattan_distances
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



def customKNNClassifier(X_train, y_train, X_test, y_test, k=3):
    # Inizializo il modello
    custom_knn = CustomKNN(k)

    # Addestro il modello
    custom_knn.fit(X_train, y_train)

    # Eseguo le predizioni
    predizioni = custom_knn.predict(X_test)
    custom_knn.score(X_test, y_test)


  

#classe custom del knn che permette di effettuare le previsioni pesando il contributo di ogni vicino in base alla sua vicinanza al nuovo record
class CustomKNN:
    #nizializza il valore di k
    def __init__(self, k=3):
        self.k = k
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
                distanze = euclidean_distances(self.mX, istanza_reshaped).flatten()
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
            accuracy = accuracy_score(y_test, pred_y)
            print(accuracy)
        else:
            print("E' prima necessario addestrare il modello")

# DA RIMUOVERE
if __name__ == '__main__':
    print("\nInizio codice\n")
    X, y = load_dataset()
    train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=0, test_size=0.25)

    customKNNClassifier(train_X, train_y, test_X, test_y)
    print("\n\nFine codice\n\n")
