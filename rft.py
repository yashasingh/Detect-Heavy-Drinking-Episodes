import pickle
import random
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import collections
import matplotlib.pyplot as plt


def add_labels_cleanTAC():
	# Read the clean tac, append label corresponding to TAC threshold
	labels = list()
	clean_tac = pd.read_csv("data/BK7610_clean_TAC.csv")

	tot_rows = len(clean_tac)
	for i in range(tot_rows):
		if clean_tac.loc[i, 'TAC_Reading'] >= 0.08:
			labels.append(1)
		else:
			labels.append(0)
	clean_tac['y'] = labels

	print("Clean tac shape",clean_tac.shape)
	del clean_tac['TAC_Reading']
	clean_tac.to_csv('data/BK7610_label.csv', encoding='utf-8')

	return 'BK7610_label.csv'


def find_ts_labels():
	# Generate TAC label for all rows corresponding its Timestamp
	clean_tac = pd.read_csv("data/BK7610_label.csv")

	# Contains timestamp with second values
	clean_ts = clean_tac.loc[:, 'timestamp'] 

    # Read Pickle
	infile = open("Pickles/Metric_4_36.pkl",'rb')
	fea = pickle.load(infile)
	infile.close()

	# Contains timestamp with millisecond values
	fea_ts = fea.loc[:, 't']//1000

	all_labels = list()
	offset_tac, offset_fea = 0, 0
	while offset_tac < len(clean_ts) and offset_fea < len(fea_ts):
		
		while fea_ts[offset_fea] < clean_ts[offset_tac]:

			all_labels.append([clean_tac.loc[offset_tac, 'y'], clean_tac.loc[offset_tac, 'timestamp']])
			offset_fea += 1
			if (offset_fea >= len(fea_ts)):
				break

		offset_tac += 1

	print("All labels: ", collections.Counter(i[0] for i in all_labels))

	return all_labels


def combine_features():
	infile = open("Pickles/Metric_0_36.pkl",'rb')
	df = pickle.load(infile)
	infile.close()
	
	for i in range(1, 18):
		if i == 12: continue
		if i < 14:
			filename = "Pickles/Metric_"+str(i)+"_36.pkl"
		else:
			filename = "Pickles/Metric_"+str(i)+"_18.pkl"
		infile = open(filename, 'rb')
		x = pickle.load(infile)
		infile.close()

		df = df.join(x.set_index('t'), on='t')

	del df['t']
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
	train_label = y[:train_idx, 0]
	test_data = X[train_idx:]
	test_label =  y[train_idx:, 0]
	test_label_ts = y[train_idx:, 1]

	print("Test Labels:", collections.Counter(test_label))
	print("Train Labels:", collections.Counter(train_label))

	print("Fitting")
	clf = RandomForestClassifier(n_estimators = 700)
	
	clf.fit(train_data, train_label)
	y_pred = clf.predict(test_data)

	score = np.mean(y_pred == test_label)
	# score = clf.score(test_data, test_label)
	print("Score:", score)

	fPos = []
	fNeg = []
	clean_tac = pd.read_csv("data/BK7610_clean_TAC.csv")
	tot_rows = len(clean_tac)
	for i, d in enumerate(test_label_ts):
		if y_pred[i] != test_label[i]:
			for j in range(tot_rows):
				if clean_tac.loc[j, 'timestamp'] == d:
					if y_pred[i] == 1: fPos.append(clean_tac.loc[j, 'TAC_Reading'])
					else: fNeg.append(clean_tac.loc[j, 'TAC_Reading'])
					break

	print('Number of False Positives: ', len(fPos))
	print('Number of False Negatives: ', len(fNeg))

	# fPos = np.sort(fPos)
	# fNeg = np.sort(fNeg)
	# df = pd.DataFrame(fPos, columns=['False Positives'])
	# boxplot = df.boxplot(column=['False Positives'])
	# df1 = pd.DataFrame(fNeg, columns=['False Negatives'])
	# boxplot1 = df1.boxplot(column=['False Negatives'])
	# plt.show()


if __name__ == "__main__":
	# add_labels_cleanTAC()
	features = combine_features()
	all_labels = find_ts_labels()
	classifier(features, all_labels)

