Idea Behind this branch:

1. Train the encoder to reconstruct clearly Z2 (y1) from it's embedding to have a meaningfull rappresentatio in the latent space.

2. Train a configuration of decoder to reconstruct Z2 and Z3 from the y1 representation learned before.

3. Make 3 Network with the same encoder.

3.1) Network to reconstruct both Z1 and Z3
3.2) Two single decoder focused on reconstructing on variable from Z2
3.3) Shared decoder with shared layers (weight sharing) and last layer to divide representations of both variables




Work in program:

- ENCODER TRAINING:

1) Adjust architecture to create an Encoder and save it's parameters
2) Create a Dummy decoder for the quality reconstruction of Z2 analysis
3) Analysisof the meaningfull of the latent rapresentation

Problems:

Lr going down too much before the KLD loss become considerable -> Training loss stuck because of too little lr
-> Have to find a way to implement an increasing lr to use when recon loss is stable and low but KLD is giant
Noticed that 0.05 for beta is too high and make recon loss go from 0.1 to 0.4 (4 times more) and KL is a little bit
better -> Try to use a lower beta with sigmoid (KL real value about 3.0) 

ENCODER MODELLO 2-1: NON ELIMINARE è il MIGLIORE (Noise 3,0, beta max = 0.01) 
Prova a mantenere la struttura e far variare i parametri di Noise


- DECODER TRAINING:

Learn how to partially train network.

1) Define the Decoder Class and the training.

