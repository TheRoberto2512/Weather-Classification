![Windows](https://img.shields.io/badge/Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white)
![Anaconda](https://img.shields.io/badge/Anaconda-%2344A833.svg?style=for-the-badge&logo=anaconda&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

# Weather Classifcation

<p align="justify">Il progetto indaga l'applicazione di tecniche di machine learning per la classificazione delle condizioni meteorologiche utilizzando un dataset tabellare sintetico. Il <a href="https://www.kaggle.com/datasets/nikhil7280/weather-type-classification">dataset</a> non rispecchia dati realistici, sono stati tutti generati sinteticamente includendo volutamente outliers e valori anomali. L'obiettivo principale è sviluppare un modello predittivo in grado di classificare con precisione diversi tipi di condizioni meteorologiche, quali sole, pioggia, neve e nuvolosità.</p>

<p align="justify">Per maggiori informazioni sul progetto è possibile consultare il report finale a <a href="INSERIRE LINK">questo link<a>.</p>

# Installazione 
## **Requisiti:**   
È necessario aver installato <a href="https://docs.anaconda.com/miniconda/">miniconda<a> sul proprio dispositivo Windows. Tutte le librerie e i pacchetti necessari sono già configurati all'interno dell'ambiente conda fornito.   
    **NOTA:** _Il progetto è stato pensato per essere eseguito sul sistema operativo Windows, non si garantisce il completo funzionamento su sistemi operativi differenti._

## **Istruzioni step-by-step:**   
1) Scaricare i file dal repository GitHub tramite Git Clone o download diretto del file .zip;     

2) <p align="justify">Aprire il terminale di Conda, spostarsi nella cartella contenente i file del progetto (basta anche solo <i>WP_requirements.yml</i>) e digitare il comando <code>conda env create -f WP_requirements.yml.</code> Questo creerà l'ambiente e darà inizio all'installazione di tutte le librerie necessarie per l esecuzione del progetto.</p>

3) <p align="justify">Una volta terminata l'installazione dell'ambiente e delle librerie è neccessario, qualora non lo si fosse già, dirigersi nella cartella contenente i file del progetto e attivare l'ambiente col comando <code>conda activate WeatherProject</code>.</p>

4) <p align="justify">La procedura di installazione è completata, ora è possibile eseguire il progetto col comando <code>python main.py</code>, assicurandosi sempre di aver attivato l'ambiente <i>WeatherProject</i> e di trovarsi nella directory del progetto.</p>

# Struttura del progetto
Il progetto è stato organizzato in più file in base ai modelli e alle operazioni da effettuare: 

<div align="center">

| Nome File | Descrizione |
| :---: | :---: |
| `WP_requirements.yml` | File per l'installazione dell'ambiente su Conda |
| `main.py` | File principale da cui è possibile eseguire l'applicazione completa | 
| `Imports.py` | Contiene gli import più ricorrenti tra tutti gli altri file |
| `AnalisiDataset.py` | Contiene le funzioni relative alle varie analisi del dataset | 
| `Shared_Utilities.py` | Contiene le funzioni condivise ricorrenti tra i vari file |  
| `Models.py` | Contiene le funzioni necessarie per l'esecuzione e i confronti tra modelli |        
| `DecisionTree.py` | Contiene le funzioni necessarie per l'esecuzione di modelli DecisionTree |     
| `NaiveBayes.py` | Contiene le funzioni necessarie per l'esecuzione di modelli NaiveBayes |      
| `SVM.py` | Contiene le funzioni necessarie per l'esecuzione di modelli SVM |
| `CustomKNN.py` |Contiene le funzioni necessarie per l'esecuzione di modelli KNN custom |
| `CustomEnsemble.py` | Contiene le funzioni necessarie per l'esecuzione di modelli Ensemble custom |
| `Preprocessing.py` | Contiene tutte le varie funzioni di pre-processing del dataset | 

</div>

# Librerie utilizzate
È stato utilizato `python` 3.10.4 e nell'ambiente Conda sono state installate le seguenti librerie:

<div align="center">

| Nome Libreria | Versione | Utilizzo |
| :---: | :---: | :---: |
| `dython` | 0.6.7  | Plot della matrice di correlazione completa |
| `imbalanced-learn` | 0.12.3 | Tecniche di Oversampling |
| `matplotlib` | 3.5.2 | Plot dei grafici dei risultati |
| `numpy` | 1.21.5 | Operazioni aritmetiche varie|
| `pandas` |1.4.4  | Gestione del dataset come dataframe|
| `scikit-learn` | 1.1.1  | Per i modelli di machine learning |
| `seaborn` | 0.13.2 | Plot dei boxplot |

</div>

# Autori
<a href="https://github.com/TheRoberto2512">Roberto A. Usai</a>, <a href="https://github.com/Cipe96">Marco Cipollina</a>, <a href="https://github.com/GabriDemu02">Gabriele Demurtas</a>, <a href="https://github.com/Chiaras97">Chiara Scalas</a> 
