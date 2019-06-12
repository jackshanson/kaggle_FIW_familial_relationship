import numpy as np 
import cv2, glob,tqdm,os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
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
	test_ids = ['dat/'+i for i in f.read().splitlines()]

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


sample_input = read_two_images(train_ids[0],train_ids[1])

inputs = tf.keras.Input(shape=(sample_input.shape[0],sample_input.shape[1],sample_input.shape[2]), name='features')
n1 = inputs
for i in range(4):
	c1 = tf.keras.layers.Conv2D(64,(3,3), activation='relu',padding='SAME',bias_initializer=tf.constant_initializer(0.01))(n1)
	m1 = tf.keras.layers.MaxPooling2D((2,2))(c1)
	n1 = tf.keras.layers.experimental.LayerNormalization()(m1)

c5 = tf.keras.layers.Conv2D(64,(3,3), activation='relu',padding='SAME',bias_initializer=tf.constant_initializer(0.01))(n1)
a5 = tf.keras.layers.GlobalAveragePooling2D()(c5)

outputs = tf.keras.layers.Dense(1, name='predictions')(a5)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(),
      loss=tf.keras.losses.binary_crossentropy,
      metrics=[tf.keras.metrics.binary_crossentropy])

batch_size = 32
def data_generator(set_ids,set_true,batch_size):
	while True:
		ids = [np.random.choice(set_ids,2) for i in range(batch_size)]
		ids += [np.array(set_true[i]) if I%2==0 else np.array([set_true[i][1],set_true[i][0]]) for I,i in enumerate(np.random.randint(0,len(set_true),[5]))]
		data = np.array([read_two_images(i[0],i[1]) for i in ids])
		labels = np.array([1 if (i[0],i[1]) in true_samples2 or (i[1],i[0]) in true_samples2 else 0 for i in ids])
		yield data,labels
hey = hi
overfitCallback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience = 20,restore_best_weights=True)
history = model.fit_generator(data_generator(train_ids,train_true,batch_size),steps_per_epoch=10,epochs=20,verbose=1,
							validation_data=data_generator(val_ids,val_true,batch_size),validation_steps=20,callbacks=[overfitCallback])