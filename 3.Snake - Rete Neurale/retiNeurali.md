# Reti neurali con PyTorch

## Introduzione alle reti neurali

Le reti neurali artificiali sono modelli computazionali ispirati al funzionamento del cervello umano. Esse sono in grado di apprendere da dati complessi e di estrarre pattern utili per compiere previsioni o prendere decisioni. Le reti neurali sono costituite da strati di neuroni artificiali interconnessi, ognuno dei quali elabora l'input ricevuto e trasmette l'output agli strati successivi.

## Cos'è una rete neurale in PyTorch e come definirla

PyTorch è una libreria open-source per il deep learning che fornisce strumenti potenti per la creazione e l'addestramento di reti neurali. Le reti neurali in PyTorch sono definite come sequenze di strati, ognuno dei quali può essere personalizzato con diverse architetture e parametri.

Per definire una rete neurale in PyTorch, è necessario prima importare il modulo `torch.nn`. Successivamente, è possibile definire una classe che eredita da `nn.Module`, la classe base per tutte le reti neurali in PyTorch. All'interno di questa classe, è possibile definire gli strati della rete utilizzando i moduli forniti da `torch.nn`.

## Cos'è nn.Linear

`nn.Linear` è uno strato di rete neurale in PyTorch utilizzato per applicare trasformazioni lineari ai dati di input. Esso rappresenta un layer completamente connesso, in cui ogni neurone è connesso a tutti i neuroni dello strato precedente. In altre parole, ogni neurone riceve un input da ogni neurone nello strato precedente. Questa connessione completa consente al layer di catturare e combinare le informazioni provenienti da tutte le features di input.

## In quali ambiti viene usato nn.Linear

`nn.Linear` viene utilizzato in diversi contesti, tra cui:

- **Classificazione di dati tabulari**: È adatto per l'elaborazione di dati strutturati, come dataset tabulari, in cui è necessario trasformare le features di input in uno spazio comune prima di applicare funzioni di attivazione.
- **Regressione lineare**: È utilizzato per modellare direttamente relazioni lineari tra le features di input e l'output desiderato.

## Alternative a nn.Linear e perché usarle

Sebbene `nn.Linear` sia utile in molti casi, ci sono situazioni in cui potrebbero essere preferite alternative:

- **Classificazione di immagini**: In questo caso, potrebbe essere preferibile utilizzare `nn.Conv2d` per catturare features spaziali nelle immagini.
- **Analisi di sequenze temporali**: `nn.LSTM` o `nn.GRU` potrebbero essere preferiti per modellare dipendenze temporali nei dati sequenziali.

In generale, è importante valutare il contesto del problema e le caratteristiche dei dati per scegliere l'approccio più appropriato.

## Approfondmento su nn.Linear

1. **Operazione Lineare**:
   - `nn.Linear` implementa un'operazione lineare definita dalla seguente equazione: `y = xA^T + b`, dove `x` è l'input al layer, `A` è una matrice di pesi (weights), `b` è un vettore bias, e `y` è l'output del layer.
   
2. **Generazione dei Parametri**:
   - Quando viene creato un oggetto `nn.Linear`, vengono generati automaticamente due set di parametri: i pesi (weights) `A` e il bias `b`. Questi parametri vengono inizializzati casualmente durante la creazione dell'oggetto e vengono aggiornati durante il processo di addestramento della rete neurale.

3. **Dimensioni dei Parametri**:
   - Le dimensioni dei pesi `A` dipendono dalle dimensioni dell'input e dell'output del layer. Ad esempio, se l'input ha dimensione `input_size` e l'output ha dimensione `output_size`, allora la matrice dei pesi `A` avrà dimensione `(output_size, input_size)`.
   - Il vettore bias `b` ha dimensione `(output_size)` e viene aggiunto all'output del layer.

4. **Utilizzo**:
   - `nn.Linear` è comunemente utilizzato all'interno di reti neurali per trasformare l'input da uno spazio di dimensione `input_size` a uno spazio di dimensione `output_size`. Questa trasformazione lineare è seguita da funzioni di attivazione non lineari come ReLU, Sigmoid, Tanh, ecc., che introducono non linearità nel modello.

In sintesi, `nn.Linear` è uno strumento essenziale per costruire reti neurali in PyTorch e viene utilizzato per implementare trasformazioni lineari dei dati all'interno della rete.


## Approfondimento su ReLU (Rectified Linear Unit)

ReLU, o Rectified Linear Unit, è una delle funzioni di attivazione non lineari più comuni utilizzate nelle reti neurali. Essa introduce non linearità all'interno del modello e aiuta a risolvere problemi realtivi all'addestramento della rete.
#### Definizione

La funzione ReLU è definita come:

\[ f(x) = \max(0, x) \]

Dove:
- \( x \) è l'input alla funzione.
- \( f(x) \) è l'output della funzione.

#### Caratteristiche principali

- **Semplicità**: ReLU è una funzione semplice e veloce da calcolare. È una semplice funzione di attivazione basata su un'operazione elementare, il che la rende efficace in termini computazionali.
- **Sparsità**: ReLU produce output sparsi, cioè si attiva solo quando l'input è positivo. Questa sparsità può aiutare a ridurre il rischio di overfitting, limitando l'interazione tra neuroni. Fa quindi in modo di ridurre il rumore.

#### Utilizzo in reti neurali

ReLU è ampiamente utilizzata come funzione di attivazione nei layer nascosti delle reti neurali. Essa è particolarmente efficace in reti neurali profonde, in cui può contribuire a mitigare il problema della scomparsa del gradiente e accelerare la convergenza durante l'addestramento della rete.

Inoltre, ReLU viene spesso utilizzata come funzione di attivazione nei layer convoluzionali delle reti neurali convoluzionali (CNN), dove aiuta a catturare estraendo features significative dalle immagini.
