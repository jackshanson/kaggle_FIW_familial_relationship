import numpy as np 
import cv2, glob,tqdm,os,pickle
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import sys
sys.path.insert(0, '/home/jack/mnt/jack/home/jack/Documents/experiments/kaggle/FIW_family_prediction/facerecmodel')
from model import create_model
from align import AlignDlib
img1 = cv2.imread('dat/F0002/MID1/P00009_face3.jpg')
#img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
plt.imshow(img1)

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def read_two_images(imgpath1,imgpath2):
	img1=cv2.imread(imgpath1)
	img2=cv2.imread(imgpath2)
	#img1mean=np.mean(img1.mean(0),0)
	#img2mean=np.mean(img2.mean(0),0)
	#img1std=np.sqrt(np.mean(np.mean((img1-img1mean)**2,0),0))
	#img2std=np.sqrt(np.mean(np.mean((img2-img2mean)**2,0),0))
	#return np.concatenate([(img1-img1mean)/img1std,(img2-img2mean)/img2std],2)
	return np.concatenate([img1/255.,img2/255.],2)

with open('train_images','r') as f:
	train_ids = f.read().splitlines()

with open('test_faces','r') as f:
	test_ids = ['dat/test/'+i for i in f.read().splitlines()]

train_families = list(set([i.split('/')[1] for i in train_ids]))

np.random.seed(1)
np.random.shuffle(train_families)

val_families = train_families[-40:]
train_families = train_families[:-40]

val_ids = [i for i in train_ids if i.split('/')[1] in val_families]
train_ids = [i for i in train_ids if i not in val_ids]

with open('train_relationships.csv','r') as f:
	labels = pd.read_csv(f).values


true_samples = [[glob.glob('dat/'+i[0]+'/*'),glob.glob('dat/'+i[1]+'/*')] for i in labels]
true_samples2 = [(j,k) for i in true_samples for j in i[0] for k in i[1]]

train_true = [i for i in true_samples2 if i[0].split('/')[1] in train_families]
val_true = [i for i in true_samples2 if i[0].split('/')[1] in val_families]

np.random.seed(1)
false_samples = []

def both_ways(tuple1,listoftuples):
	return ((tuple1[1],tuple1[0]) not in listoftuples) and ((tuple1[0],tuple1[1]) not in listoftuples)

#choice = zip(np.random.choice(train_ids,len(true_samples2)*10),np.random.choice(train_ids,len(true_samples2)*10))
#false_samples = []
#for i in tqdm.tqdm(choice):
#	if i[0]!=i[1] and both_ways(i,false_samples) and both_ways(i,true_samples2):
#		false_samples.append((i[0],i[1]))








nn4_small2_pretrained = create_model()
nn4_small2_pretrained.load_weights('facerecmodel/weights/nn4.small2.v1.h5')



def distance(emb1, emb2):
    return np.sum(np.square(emb1 - emb2))
alignment = AlignDlib('facerecmodel/models/landmarks.dat')
def align_image(img):
    return alignment.align(96, img, alignment.getLargestFaceBoundingBox(img), 
                           landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)

def show_pair(idx1, idx2):
    plt.figure(figsize=(8,3))
    #plt.suptitle(f'Distance = {distance(embedded[idx1], embedded[idx2]):.2f}')
    plt.subplot(121)
    plt.imshow(load_image(metadata[idx1].image_path()))
    plt.subplot(122)
    plt.imshow(load_image(metadata[idx2].image_path()));  

all_ids = train_ids + val_ids
try:
	with open('embeddings.pkl','rb') as f:
		embeddings = pickle.load(f)
except:
	embeddings = {}
	for i in tqdm.tqdm(all_ids):
			image = cv2.imread(i)[:,:,::-1]
			try:
				embeddings[i] = nn4_small2_pretrained.predict(np.expand_dims(align_image(image),axis=0))[0]
			except:
				embeddings[i] = nn4_small2_pretrained.predict(np.expand_dims(cv2.resize(image,(96,96)),axis=0))[0]
	with open('embeddings.pkl','wb') as f:
		pickle.dump(embeddings,f)

try:
	with open('wrong_samples.pkl','rb') as f:
		not_related = pickle.load(f)
except:
	not_related = [(all_ids[I],all_ids[i]) for I,i in enumerate(np.random.choice(len(all_ids),len(all_ids),replace=False)) if all_ids[i]!=all_ids[I] and (all_ids[I],all_ids[i]) not in true_samples2 and (all_ids[i],all_ids[I]) not in true_samples2]	
	with open('wrong_samples.pkl','wb') as f:
		pickle.dump(not_related,f)



true_distances = [distance(embeddings[i[0]],embeddings[i[1]]) for i in true_samples2]
false_distances = [distance(embeddings[i[0]],embeddings[i[1]]) for i in not_related]

try:
	with open('test_embeddings.pkl','rb') as f:
		test_embeddings = pickle.load(f)
except:
	test_embeddings = {}
	for i in tqdm.tqdm(test_ids):
		image = cv2.imread(i)[:,:,::-1]
		try:
			test_embeddings[i] = nn4_small2_pretrained.predict(np.expand_dims(align_image(image)/255.,axis=0))[0]
		except:
			test_embeddings[i] = nn4_small2_pretrained.predict(np.expand_dims(cv2.resize(image,(96,96))/255.,axis=0))[0]
	with open('test_embeddings.pkl','wb') as f:
		pickle.dump(test_embeddings,f)


with open('sample_submission.csv','r') as f:
	dat = pd.read_csv(f)

test_faces=dat.img_pair.str.split('-')
test_distances = [distance(test_embeddings['dat/test/'+i[0]],test_embeddings['dat/test/'+i[1]]) for i in test_faces]



def plot_two_ids(datset,ind):
	plt.subplot(2,1,1)
	plt.imshow(cv2.imread('dat/test/'+test_faces[ind][1])[:,:,::-1])
	plt.subplot(2,1,2)
	plt.imshow(cv2.imread('dat/test/'+test_faces[ind][0])[:,:,::-1])

all_distances = np.sum(test_distances)
dat['is_related'] = 1-(test_distances-np.min(test_distances))/(np.max(test_distances)-np.min(test_distances))













