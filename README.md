# Searching-Similar-Images-using-VAE
Image Search Using Embeddings Extracted from VAE. This repository contains code a simple image search engine using Flask & Pytorch trained model.

## Credits

* This repository is based on the [Simple Image Search Project by matsui528](https://github.com/matsui528/sis)

* The Variational Autoencoder was trained using Pytorch . This video tutorial by [Dr Yan Le Cunn & Dr Alfredo Canziani](https://www.youtube.com/watch?v=7Rb4s9wNOmc&list=PLLHTzKZzVU9eaEyErdV26ikyolxOsz6mq&index=15&t=1639s) was helpful.


## Files & their functions

```
server.py - running this file will activate the local application, where the user will choose images. This python file also generates embedding.npy & img_paths.json
embedding.npy - embedding of each image stored in '.static/img/'
img_paths.json - This keeps a record of all the images in '.static/img/'
requirements.txt - List of libraries used for this application.
feature_extractor.py - This contains functions that will be used by the application to extract image embeddings.






```

