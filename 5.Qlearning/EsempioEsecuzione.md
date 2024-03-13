# Esempio Esecuzione Q-Learning Model

Supponiamo che per un dato stato all'esecuzione di `pred = self.model(state)` e `target = pred.clone()` ottengo `target = [0.5, 40, 20]`. In questo caso, l'azione scelta è la seconda, che nel nostro caso è associata a destra.

Supponiamo che a destra ci sia un frutto, quindi eseguendo quell'azione ottengo una reward positiva, ad esempio di 400, il che significa che in `Q_new = reward` avrò 400.

Ora abbiamo due casi:

### Caso 1: Ottengo un "done"
Nel nostro sistema è in realtà impossibile, in quanto non si può prendere un frutto e andare in game over nella stessa azione. Ma nel caso fosse possibile, allora `Q_new` resterebbe 400.

### Caso 2: Non ottengo un "done"
Devo considerare anche cosa succede in futuro, e lo so perché ho passato il `next_state` alla mia funzione. In questo caso, tramite la formula `Q_next = torch.max(self.model(next_state))` e `Q_new = reward + self.gamma * Q_next`, ottengo un `Q_new` diverso che si basa anche sullo stato futuro.

Infine, con `target[torch.argmax(action).item()] = Q_new`, cambio a tutti gli effetti quello che avevo predetto modificando il valore dell'azione scelta. Vuol dire che il `target` avrà qualcosa di simile a `target = [0.5, Q_new, 20]`. Quindi, avendo trovato un frutto, l'azione di girare a destra diventerà più probabile e influirà di più quando richiamata da `Q_next = torch.max(self.model(next_state))`.

Da considerare che questo `target` dovrà essere associato allo stato attraverso la rete neurale, con le istruzioni:

```python
self.optimizer.zero_grad()
loss = self.criterion(target, pred)
loss.backward()
self.optimizer.step()
```

E quindi non è detto che, richiamando quello stato, avrò come risultato il target, ma sicuramente la rete si sarà spostata per raggiungere quel risultato.