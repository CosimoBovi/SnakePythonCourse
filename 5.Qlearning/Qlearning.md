### Cos'è il Q-Learning

Il Q-Learning è un algoritmo di apprendimento per rinforzo che viene utilizzato per addestrare agenti intelligenti a prendere decisioni ottimali in ambienti basati su azioni. L'obiettivo principale del Q-Learning è quello di apprendere una funzione Q, chiamata funzione di valore azione, che assegna a ogni coppia stato-azione un valore che rappresenta la somma delle ricompense future che ci si aspetta di ottenere partendo dallo stato corrente e scegliendo quell'azione specifica.

Nel contesto di un dato stato, selezionare l'azione con il valore Q maggiore significa scegliere l'azione che, con maggiore probabilità, massimizzerà le ricompense future.

### Funzionamento del Q-Learning

Il Q-Learning si basa sull'idea di apprendere iterativamente i valori Q attraverso l'interazione dell'agente con l'ambiente. In ogni passo dell'addestramento, l'agente seleziona un'azione in base alla politica di esplorazione, esegue l'azione, osserva la ricompensa ottenuta e lo stato successivo, e quindi aggiorna il valore Q per la coppia stato-azione corrispondente.

L'aggiornamento dei valori Q avviene utilizzando l'equazione di Bellman, che esprime il valore Q per uno stato e un'azione come la somma della ricompensa immediata e del massimo valore Q previsto per lo stato successivo. Questo aggiornamento viene eseguito iterativamente fino a quando la funzione Q converge a una soluzione stabile.

### Equazione di Bellman per il Q-Learning

L'equazione di Bellman per il Q-Learning è espressa come segue:

\[ Q(s, a) = R(s, a) + \gamma \cdot \max_{a'}[Q(s', a')] \]



Dove:
- \( Q(s, a) \) è il valore Q per lo stato \( s \) e l'azione \( a \).
- \( R(s, a) \) è la ricompensa immediata ottenuta eseguendo l'azione \( a \) nello stato \( s \).
- \( \gamma \) è il fattore di sconto che controlla l'importanza delle ricompense future rispetto a quelle immediate.
- \( \max_{a'}[Q(s', a')] \) rappresenta il valore Q massimo previsto per lo stato successivo \( s' \), considerando tutte le possibili azioni \( a' \).

### Parallelismo con il Codice Python

Vediamo ora come questa equazione viene implementata nel codice Python fornito:

```python
Q_new = reward
if not done:
    Q_new = reward + self.gamma * torch.max(self.model(next_state))  
target[torch.argmax(action).item()] = Q_new
```

- La variabile `Q_new` rappresenta il valore Q aggiornato per lo stato e l'azione corrispondenti.
- La riga `if not done:` controlla se l'episodio è terminato. Se non è terminato, viene calcolato il valore Q aggiornato utilizzando l'equazione di Bellman.
- Il termine `reward + self.gamma * torch.max(self.model(next_state))` rappresenta \( R(s, a) + \gamma \cdot \max_{a'}[Q(s', a')] \) nella formula di Bellman. Viene calcolata la somma della ricompensa immediata e del valore Q massimo previsto per lo stato successivo, moltiplicato per il fattore di sconto \( \gamma \).
- Infine, il valore Q aggiornato viene assegnato al target per l'azione corrispondente utilizzando `target[torch.argmax(action).item()] = Q_new`. Questo corrisponde all'aggiornamento del valore Q per la coppia stato-azione secondo l'equazione di Bellman.
