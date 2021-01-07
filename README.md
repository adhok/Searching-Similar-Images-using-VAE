# Searching Similar Images of fruits using VAE
This repository contains code for a simple image search engine using Flask & Pytorch trained model. The VAE model was trained using the Fruits 360 data that can be found on [Kaggle](https://www.kaggle.com/moltean/fruits). The script to get the trained model object (checkpoint.pth) can be found [here](https://www.kaggle.com/adhok93/unsupervised-learning-using-vae)


## Credits

* This repository is based on the [Simple Image Search Project by matsui528](https://github.com/matsui528/sis)

* The Variational Autoencoder was trained using Pytorch . This video tutorial by [Dr Yan Le Cunn & Dr Alfredo Canziani](https://www.youtube.com/watch?v=7Rb4s9wNOmc&list=PLLHTzKZzVU9eaEyErdV26ikyolxOsz6mq&index=15&t=1639s) was helpful.


## Files & their functions

```
* server.py - running this file will activate the local application, where the user will choose images. This python file also generates embedding.npy & img_paths.json
* embedding.npy - embedding of each image stored in '.static/img/'
* img_paths.json - This keeps a record of all the images in '.static/img/'
* requirements.txt - List of libraries used for this application.
* feature_extractor.py - This contains functions that will be used by the application to extract image embeddings.

* model/ 
  - checkpoint.pth - This is the saved model file
* templates/
  - index.html - File that contains information about the UI of the app
* static
  - img/ 
    - Contains All the images that are used to construct the embeddings
  - uploaded/
    - Contains images that are uploaded into the application.
```

## Demo

![alt text](https://raw.githubusercontent.com/adhok/Searching-Similar-Images-using-VAE/main/image_search.gif)


## Running the Demo

Run the following and open `http://0.0.0.0:5000/`

```
git@github.com:adhok/Searching-Similar-Images-using-VAE.git
cd Searching-Similar-Images-using-VAE
pip install -r requirements.txt
python server.py

```

