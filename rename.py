import os


for root, dirs, files in os.walk('/Users/padhokshaja/Documents/Python/fruits360/testing_data/'):
    print('ROot')
    print(root)
    fruit = root.rsplit("/",1)[1]
    fruit_clean = fruit.replace(' ','_')
    for file_ in files:
        img_path_original = root+"/"+file_
        img_path_new= root+"/"+fruit_clean+file_
        os.rename(img_path_original,img_path_new) 


