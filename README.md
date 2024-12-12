Appunti:

1) Divergenza KL:

Ha valori molto grandi all'inizio del training ma scende molto velocemente anche se beta è nullo quando la ricostruzione scende da 1.4 a 0.8
Gli ordini di grandezza però sono diversi ( Recon Loss: o(1)  KL loss = o(100) o anche 20)
Per valori di beta anche molto bassi devia l'apprendimento in favore della KL mandando la Recon in plateau. Non si esce facilmente da qua perchè lo scheduler agisce sulla loss totale. Bisogna vedere come combinare bene i parametri.
Idea: warm up del learning rate in cicli


2) Ricostruzione: Il Decoder ricostruisce in maniera ottima Z1 mentre Z3 in maniera pessima. 
La loss di ricostruzione rimane stagnante rispetto alla KL quindi si ipotizza che la rete preferisca sacrificare la rappresentazione di Z3 e ricostruire meglio Z1 per ottimizzare la loss.

Idea: Allenare encoder a ricostruire se stesso (y1) e poi usare gli stessi pesi dell'encoder aggiungendo una parte in cui alleno un decoder che parte da una rappresentazione ottimale di y1 nello spazio latente e specializza solamente il decoder a ricostruire Z1 e Z3.
Ma a questo punto cosa faccio: Alleno due decoder quindi faccio due reti neurali con stesso encoder e due differenti decoder?
Oppure al posto di fare due differenti decoder ne faccio uno unico come ora forzato ad apprendere le rappresentazioni di entrambi?
Posso usare il weight sharing di alcuni layer del decoder per avere una rappresentazione condivisa che poi si differenzia nei layer finali del decoder?
