import numpy as np
from PIL import Image
from feature_extractor import VAE,extract
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path
import pandas as pd
import os
from numpy.linalg.linalg import norm
from numpy import set_printoptions
from numpy import array as a
import json

app = Flask(__name__)

# Read image features

model = VAE()
features = []
img_paths = []
# for feature_path in Path("./static/feature").glob("*.npy"):
#     features.append(np.load(feature_path))
#     img_paths.append(Path("./static/img") / (feature_path.stem + ".jpg"))
# features = np.array(features)

for image in os.listdir('./static/img/'):
    if '.jpg' in image:
        img_paths.append("./static/img/"+image)
        feature_image = extract(img="./static/img/"+image)
        features.append(feature_image)
print(len(img_paths))
print(len(features))

features_numpy = np.array(features)
features_numpy = features_numpy.reshape(features_numpy.shape[0]*features_numpy.shape[1],features_numpy.shape[2])
features = features_numpy

## Save List of image names

with open("img_paths.json", 'w') as f:
    # indent=2 is not needed but makes the file human-readable
    json.dump(img_paths, f, indent=2) 


np.save('embeddings.npy',features)    # .npy extension is added if not given

features = np.load('embeddings.npy')
with open("img_paths.json", 'r') as f:
    img_paths = json.load(f)





@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':

        features = np.load('embeddings.npy')
        with open("img_paths.json", 'r') as f:
            img_paths = json.load(f)



        # ###### Image Embedding ########
        # for image in os.listdir('./static/img/'):
        #     if '.jpg' in image:
        #         img_paths.append("./static/img/"+image)
        #         feature_image = extract(img="./static/img/"+image)
        #         features.append(feature_image)
        # print(len(img_paths))
        # print(len(features))

        # features_numpy = np.array(features)
        # features_numpy = features_numpy.reshape(features_numpy.shape[0]*features_numpy.shape[1],features_numpy.shape[2])
        # features = features_numpy







        file = request.files['query_img']

        # Save query image
        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
        img.save(uploaded_img_path)

        # Run search
        query = extract(uploaded_img_path)
        query = query.reshape(1,20)



        ### Create Cosine Function

        features = np.append(features,query,axis=0)
        
        img_paths.append(uploaded_img_path)

        ## Calculation
        M = features

        DotProducts = M.dot(M.T)

        # kronecker product of row norms
        NormKronecker = a([norm(M, axis=1)]) * a([norm(M, axis=1)]).T

        CosineSimilarity = DotProducts / NormKronecker
        import pandas as pd
        df_vae = pd.DataFrame(CosineSimilarity)
        df_vae.index = img_paths
        df_vae.columns = img_paths

        filtered_df = df_vae[[uploaded_img_path]].sort_values(by=uploaded_img_path,ascending=False)

        filtered_df = filtered_df.head(30)

        recommended_images = filtered_df.index

        score_list = filtered_df[uploaded_img_path].to_list()













        # dists = np.linalg.norm(features-query, axis=1)  # L2 distances to features
        # ids = np.argsort(dists)[:5]  # Top 5 results
        scores = [(score_list[i], recommended_images[i]) for i in range(30)]
        print(scores)

        return render_template('index.html',
                               query_path=uploaded_img_path,
                               scores=scores)
    else:
        return render_template('index.html')


if __name__=="__main__":
    app.run("0.0.0.0")
