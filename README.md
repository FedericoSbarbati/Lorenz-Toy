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


- DECODER TRAINING:

Learn how to partially train network.

1) Define the Decoder Class and the training.

