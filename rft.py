import pickle
import random
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import collections

def add_labels_cleanTAC():

	# Read the clean tac, append label corresponding to TAC threshold

	labels = list()
	clean_tac = pd.read_csv("data/clean_tac/BK7610_clean_TAC.csv")


	tot_rows = len(clean_tac)
	for i in range(tot_rows):
		if clean_tac.loc[i, 'TAC_Reading'] >= 0.08:
			labels.append(1)
		else:
			labels.append(0)
	clean_tac['y'] = labels

	print("Clean tac shape",clean_tac.shape)
	del clean_tac['TAC_Reading']

	clean_tac.to_csv('BK7610_label.csv', encoding='utf-8')
	return 'BK7610_label.csv'


def find_ts_labels():

	# Generate TAC label for all rows corresponding its Timestamp

	clean_tac = pd.read_csv("BK7610_label.csv")

	# Contains timestamp with second values
	clean_ts = clean_tac.loc[:, 'timestamp'] 

	
    # Read Pickle
	infile = open("Metric_4_36.pkl",'rb')
	fea = pickle.load(infile)
	infile.close()

	# Contains timestamp with millisecond values
	fea_ts = fea.loc[:, 't']//1000

	all_labels = list()
		
	offset_tac, offset_fea = 0, 0

	while offset_tac < len(clean_ts) and offset_fea < len(fea_ts):
		
		while fea_ts[offset_fea] < clean_ts[offset_tac]:

			all_labels.append(clean_tac.loc[offset_tac, 'y'])
			offset_fea += 1
			if (offset_fea >= len(fea_ts)):
				break

		offset_tac += 1
	# print(all_labels)
	print(collections.Counter(all_labels))

	return all_labels


def combine_features():

	infile = open("Metric_1_36.pkl",'rb')
	df = pickle.load(infile)
	infile.close()
	
	for i in range(2, 18):
		if i == 12: continue
		if i < 14:
			filename = "Metric_"+str(i)+"_36.pkl"
		else:
			filename = "Metric_"+str(i)+"_18.pkl"
		infile = open(filename, 'rb')
		x = pickle.load(infile)
		infile.close()

		df = df.join(x.set_index('t'), on='t')

	# print(df.shape)
	del df['t']
	# print(df.head(3))

	outfile = open("X.pkl",'wb')
	pickle.dump(df, outfile)
	outfile.close()

	return df

def classifier(X, y):
	

	zipped= list(zip(X.values, np.array(y)))	
	random.shuffle(zipped)
	X, y  = zip(*zipped)
	X = np.array(X)
	y = np.array(y)


	train_idx = int(X.shape[0] * 0.75)
	train_data = X[:train_idx]
	train_label = y[:train_idx]
	test_data = X[train_idx:]
	test_label =  y[train_idx:]

	print(collections.Counter(test_label))
	print(collections.Counter(train_label))
	print(test_label)
	print(train_label)

	print("Fitting")
	clf = RandomForestClassifier(n_estimators = 700)
	
	clf.fit(X, y)
	print("Fitted")
	# print(clf.predict(test_data))
	score = clf.score(test_data, test_label)
	print("Score:", score)


if __name__="__main__":
	features = combine_features()
	all_labels = find_ts_labels()
	classifier(features, all_labels)

