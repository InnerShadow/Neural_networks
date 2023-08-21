Make autoencoder of digits images, mnist train data pack.
Show resualt of encoding on first 10 images.
Show difference between direct himotopy and model that build NN.
2dims.py has 2 dimension hidden layer, visualizations of hidden layer spots and some special cases of vectors.
VAE.py - VAE has codder with 28 * 28 input, and hidden layer output, that perform calculation the Kullback-Leibler divergence,
this value is decoder input layer. Decoder output - 28 * 28 array.
CVAE.py - baisicly VAE.py, but now we have input encoder & decoder class mark. This provide opportunity to
have diffrent numbers from the same spot of hiddent layer dimension. 
Like test of this we have "Test style transfer using z_meaner & tr_style" block, where we took same spots of hidden layer dimension &
generate diffrent images with the style of "dig1" writing.
GUN.py using generator & discriminator to compite with each other to make more realistic images.
GUn+VAE.py has VAE on generator input unlike GUN.py has just noizer + generator.x``