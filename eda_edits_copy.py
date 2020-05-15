import csv
import numpy as np 
import pandas as pd
from datetime import datetime
import pickle
import statistics



def seperate_pid_data():
	acc_data = pd.read_csv("data/all_accelerometer_data_pids_13.csv")
	pids = list(set(acc_data['pid']))
	for pid in pids:
		df = acc_data.loc[acc_data['pid'] == pid]
		df.to_csv(pid, encoding='utf-8')
		print(str(pid)+".csv created")
	print(pids)
	return pids

def create_per_second_data(pid_filename, metric_name):
	acc_data = pd.read_csv(pid_filename)
	prev_ts = 0 
	full_frame = list()
	sub_frame = list()
	tot_rows = len(acc_data)

	for idx in range(0, tot_rows):

		if idx%10000 == 0: print(idx, "**")

		r = acc_data.iloc[idx]
		curr_ts = r['time']%1000

		if idx != 0: prev_ts = acc_data.loc[idx-1, 'time']%1000

		if curr_ts > prev_ts:
			sub_frame.append([r['time'], r['x'], r['y'], r['z']])

		else:
			# Do calculations for all enteries of one second
			sub_frame = np.array(sub_frame)
			metrics_axis = []

			# Add the last timstamp in that second window
			metrics_axis.append(sub_frame[-1, 0])

			# Iterating over x, y, z axis
			for col in range(1,4):
				if metric_name == 'Mean':
					metrics_axis.append(sub_frame[:, col].mean())
				elif metric_name == 'Std_Dev':
					metrics_axis.append(statistics.stdev(sub_frame[:, col]))
				elif metric_name == 'Median':
					metrics_axis.append(statistics.median(sub_frame[:, col]))

			full_frame.append(metrics_axis)
			sub_frame = list()

	full_frame = np.array(full_frame)

	# Pickling this data

	filename = metric_name+'_all_per_sec_all_axis.pkl'
	outfile = open(filename,'wb')
	pickle.dump(full_frame,outfile)
	outfile.close()

	print("Full Frame Shape - ", full_frame.shape)
	return filename


def create_per_window_data(filename, metric_no):

	# Read the pickle file that contains entry for each second

	infile = open(filename,'rb')
	mean_all = pickle.load(infile)
	infile.close()

	tot_rows = len(mean_all)
	full_frame = list()
	single_row = list()
	i = 0
	print("Shape of data obatined from pickle - ", mean_all.shape)

	# Calculate summary statistics for this metric

	while i+10 < tot_rows:
		single_row.append(mean_all[i+9:i+10, 0][0])

		for col in range(1, 4):

			sub_frame = mean_all[i:i+10, col]



			single_row.append(sub_frame.mean())
			single_row.append(sub_frame.var())
			single_row.append(sub_frame.max())
			single_row.append(sub_frame.min())
			sub_frame = sorted(sub_frame)
			single_row.append(np.array(sub_frame[0:4]).mean())
			single_row.append(np.array(sub_frame[8:11]).mean())

		full_frame.append(single_row)
		single_row = list()
		i += 10
	full_frame = np.array(full_frame)

	
	print("Shape of generated frame for each 10 sec window ", full_frame.shape)

	col_names = ['xMe', 'xVr', 'xMx', 'xMi', 'xUM', 'xLM', 'yMe', 'yVr', 'yMx', 'yMn', 'yUM', 'yLM', 'zMe', 'zVr', 'zMx', 'zMi', 'zUM', 'zLM']

	df1 = pd.DataFrame.from_records(full_frame, columns = ['t']+[str(metric_no+names) for names in col_names] )
	print("df1 created !!!!")

	# Calculating the values out of difference of two windows

	diff_frame = list()
	for i in range(len(full_frame)):
		if i==0: diff_frame.append(full_frame[:1, 1:][0])
		else:
			diff = full_frame[i:i+1, 1:] - full_frame[i-1:i, 1:]
			diff_frame.append(diff[0])
	diff_frame = np.array(diff_frame)

	print("diff_frame created with shape", diff_frame.shape)


	# Generating set2 col name
	
	df2 = pd.DataFrame.from_records(diff_frame, columns = [str("d"+metric_no+names) for names in col_names])
	print("df2 created !!!!")


	result_df = pd.concat([df1, df2], axis=1)
	print(result_df.shape)

	# Pickle this data

	outfile = open("Metric_"+str(metric_no)+"_36.pkl",'wb')
	pickle.dump(result_df, outfile)
	outfile.close()
	return "Metric_"+str(metric_no)+"_36.pkl"





if __name__ == "__main__":
	pids = seperate_pid_data()
	features, pid = ["Mean", "Median", "Std_dev"], pids[0]
	for i, fea in enumerat(features):
		filename = create_per_second_data(pid, fea)
		filename = create_per_window_data(filename, i)















