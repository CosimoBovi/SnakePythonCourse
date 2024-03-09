# Loss

## Cos'è la funzione di loss

La funzione di loss, o funzione di costo, è una misura che quantifica quanto il modello predittivo differisca dai valori di output desiderati durante il processo di addestramento. In sostanza, rappresenta l'errore tra le predizioni del modello e i valori reali del dataset di addestramento.

### A cosa serve la funzione di loss

La funzione di loss è fondamentale nell'addestramento di modelli di machine learning e reti neurali. Serve a guidare l'ottimizzazione del modello durante il processo di apprendimento, fornendo un segnale di retroazione che indica come devono essere aggiornati i parametri del modello per ridurre l'errore tra le predizioni e i valori reali.

### Cos'è MSE (Mean Squared Error)

MSE, o Mean Squared Error, è una comune funzione di loss utilizzata per misurare l'errore quadratico medio tra le predizioni del modello e i valori reali del dataset di addestramento. Si calcola come la media delle differenze quadrate tra ciascuna previsione e il valore reale corrispondente.

### In quali ambiti si usa MSE

MSE è ampiamente utilizzato in problemi di regressione, dove l'obiettivo è predire un valore numerico continuo. Questo può includere previsioni finanziarie, previsioni del mercato azionario, analisi delle serie temporali, previsioni meteorologiche e molti altri ambiti in cui è necessario stimare un valore numerico.

### Alternative a MSE e perché usarle

*** TLTR: MSE e le sue varianti sono utili nel caso di valori, Cross-Entropy Loss è utilizzato per classificare le immagini ***

Esistono diverse alternative a MSE, a seconda delle esigenze specifiche del problema e delle caratteristiche del dataset:

1. **MAE (Mean Absolute Error)**: È simile a MSE, ma calcola la media delle differenze assolute tra le previsioni del modello e i valori reali. È meno sensibile agli outlier rispetto a MSE.

2. **Huber Loss**: Combina le caratteristiche di MSE e MAE, utilizzando la penalizzazione quadratica per errori più piccoli e lineare per errori più grandi. È utile quando ci sono outlier nel dataset.

3. **Cross-Entropy Loss**: Utilizzato principalmente in problemi di classificazione, misura la discrepanza tra la distribuzione delle probabilità previste dal modello e la distribuzione reale dei dati di classificazione.

4. **Kullback-Leibler Divergence**: Misura la divergenza tra due distribuzioni di probabilità e viene utilizzata come funzione di loss nelle reti neurali per la classificazione multiclasse.

Le alternative a MSE possono essere preferite in determinate situazioni, ad esempio quando si desidera penalizzare in modo diverso gli errori più grandi rispetto a quelli più piccoli, o quando si lavora con problemi di classificazione anziché regressione. La scelta della funzione di loss dipende dall'obiettivo del problema e dalle caratteristiche del dataset.

# Ottimizzatore

## Cos'è un ottimizzatore

Un ottimizzatore è un algoritmo utilizzato per modificare i parametri di un modello al fine di ridurre la funzione di loss durante il processo di addestramento. In sostanza, un ottimizzatore regola i pesi e i bias del modello in modo che le predizioni del modello si avvicinino sempre più ai valori desiderati.

### A cosa serve un ottimizzatore

L'obiettivo principale di un ottimizzatore è quello di minimizzare la funzione di loss del modello, adattando iterativamente i parametri del modello durante il processo di addestramento. Questo consente al modello di apprendere dai dati e di migliorare le sue prestazioni nel compito di interesse.

### Cos'è Adam

Adam (Adaptive Moment Estimation) è un algoritmo di ottimizzazione ampiamente utilizzato nell'addestramento di reti neurali. Si tratta di un metodo di ottimizzazione basato sul concetto di momenti del primo e secondo ordine dei gradienti dei pesi del modello.

### In quali ambiti si usa Adam

Adam è ampiamente utilizzato in applicazioni di deep learning e machine learning per l'addestramento di reti neurali. È particolarmente popolare per la sua efficacia e facilità d'uso. Adam è adatto per una vasta gamma di problemi, inclusi problemi di classificazione, regressione, generazione di testo, visione artificiale e molto altro.

### Alternative ad Adam e perché usarle

*** TLTR: Adam è più efficiente ma richiede più risorse, SGD richiede meno risorse ma può bastare per modelli con pochi dati ***

Ci sono diverse alternative ad Adam, ciascuna con le proprie caratteristiche e vantaggi:

1. **SGD (Stochastic Gradient Descent)**: È l'ottimizzatore più fondamentale e utilizzato ampiamente. È più semplice di Adam ma può richiedere più tempo per convergere e può soffrire di problemi come il punto di sella. Richeide meno risorse

2. **RMSprop (Root Mean Square Propagation)**: È simile ad Adam ma utilizza una media mobile esponenziale dei gradienti quadrati anziché i momenti del primo e secondo ordine. RMSprop può essere più stabile rispetto ad Adam in determinate situazioni.

3. **Adagrad (Adaptive Gradient Algorithm)**: Adagrad adatta il tasso di apprendimento per ciascun parametro in base alla frequenza dei suoi aggiornamenti. Tuttavia, può diminuire troppo rapidamente il tasso di apprendimento, portando a prestazioni peggiori nel lungo termine.

4. **Adadelta**: È una variazione di Adagrad che tenta di risolvere il problema del deterioramento del tasso di apprendimento. È più robusto rispetto ad Adagrad ma richiede più risorse computazionali.

La scelta dell'ottimizzatore dipende spesso dalle caratteristiche del dataset, dalla complessità del modello e dalle esigenze specifiche del problema. È importante sperimentare con diversi ottimizzatori e regolazioni dei parametri per trovare quello più adatto al problema in questione.