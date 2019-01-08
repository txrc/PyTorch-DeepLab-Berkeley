import os

# dataset_path = './BDD_Deepdrive/bdd100k/images/100k/train'
# dataset_path = './BDD_Deepdrive/bdd100k/images/100k/val'
dataset_path = './BDD_Deepdrive/bdd100k/images/100k/test'


dataset = os.listdir(dataset_path)

# Dataset file names 
with open('BDD_train.txt', 'w') as f:
	for file in dataset:
		f.write("{} \n".format(file.strip('.jpg')))


