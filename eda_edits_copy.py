import csv
import numpy as np 
import pandas as pd
import pickle
import statistics
import utility
import enum

from datetime import datetime
from scipy.fftpack import fft
from scipy.stats import kurtosis


class Features(enum.Enum):
	Mean = 0
	Median = 1
	Std_Dev = 2
	ZeroCrsRate = 3
	Max_Raw = 4
	Min_Raw = 5
	Max_Abs = 6
	Min_Abs = 7
	Spec_Ent_Time = 8
	Spec_Ent_Freq = 9
	Spec_Cent = 10
	Spec_Sprd = 11
	Spec_Flux = 12
	Spec_RollOff = 13

	Max_freq = 14
	Skewness = 15
	Kurtosis = 16
	Avg_Power = 17


FeatureType = {}
FeatureType[Features.Mean] = 'Mean'
FeatureType[Features.Median] = 'Median'
FeatureType[Features.Std_Dev] = 'Std_Dev'
FeatureType[Features.ZeroCrsRate] = 'ZeroCrsRate'
FeatureType[Features.Max_Raw] = 'Max_Raw'
FeatureType[Features.Min_Raw] = 'Min_Raw'
FeatureType[Features.Max_Abs] = 'Max_Abs'
FeatureType[Features.Min_Abs] = 'Min_Abs'
FeatureType[Features.Spec_Ent_Time] = 'Spec_Ent_Time'
FeatureType[Features.Spec_Ent_Freq] = 'Spec_Ent_Freq'
FeatureType[Features.Spec_Cent] = 'Spec_Cent'
FeatureType[Features.Spec_Sprd] = 'Spec_Sprd'
FeatureType[Features.Spec_Flux] = 'Spec_Flux'
FeatureType[Features.Spec_RollOff] = 'Spec_RollOff'
FeatureType[Features.Max_freq] = 'Max_freq'
FeatureType[Features.Skewness] = 'Skewness'
FeatureType[Features.Kurtosis] = 'Kurtosis'
FeatureType[Features.Avg_Power] = 'Avg_Power'


def seperate_pid_data():
	acc_data = pd.read_csv("data/all_accelerometer_data_pids_13.csv")
	pids = list(set(acc_data['pid']))
	for pid in pids:
		df = acc_data.loc[acc_data['pid'] == pid]
		df.to_csv(pid, encoding='utf-8')
		print(str(pid)+".csv created")
	print(pids)

	return pids


def create_per_second_data(pid_filename, metric_no):
	acc_data = pd.read_csv(pid_filename)
	prev_ts = 0 
	full_frame = list()
	sub_frame = list()
	tot_rows = len(acc_data)
	# count = 0
	# fft_magnitude_previous = {}

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

			# if count == 0:
			# 	fft_magnitude_previous[0] = 0
			# 	fft_magnitude_previous[1] = 0
			# 	fft_magnitude_previous[2] = 0
			# 	count += 1

			# Add the last timstamp in that second window
			metrics_axis.append(sub_frame[-1, 0])

			# Iterating over x, y, z axis
			for col in range(1,4):
				if metric_no == Features.Mean.value:
					metrics_axis.append(sub_frame[:, col].mean())
				elif metric_no == Features.Median.value:
					metrics_axis.append(statistics.stdev(sub_frame[:, col]))
				elif metric_no == Features.Std_Dev.value:
					metrics_axis.append(statistics.median(sub_frame[:, col]))
				elif metric_no == Features.ZeroCrsRate.value:
					metrics_axis.append(utility.zero_crossing_rate(sub_frame[:, col]))
				elif metric_no == Features.Max_Raw.value:
					metrics_axis.append(max(sub_frame[:, col]))
				elif metric_no == Features.Min_Raw.value:
					metrics_axis.append(min(sub_frame[:, col]))
				elif metric_no == Features.Max_Abs.value:
					metrics_axis.append(max(abs(sub_frame[:, col])))
				elif metric_no == Features.Min_Abs.value:
					metrics_axis.append(min(abs(sub_frame[:, col])))
				elif metric_no == Features.Spec_Ent_Time.value:
					metrics_axis.append(utility.spectral_entropy(sub_frame[:, col]))
				elif metric_no == Features.Spec_Ent_Freq.value:
					fft_magnitude = abs(fft(sub_frame[:, col]))
					fft_magnitude = fft_magnitude[0:int(sub_frame.shape[0]/2)]
					fft_magnitude = fft_magnitude / len(fft_magnitude)
					metrics_axis.append(utility.spectral_entropy(fft_magnitude))
				elif metric_no == Features.Spec_Cent.value:
					fft_magnitude = abs(fft(sub_frame[:, col]))
					fft_magnitude = fft_magnitude[0:int(sub_frame.shape[0]/2)]
					fft_magnitude = fft_magnitude / len(fft_magnitude)
					metrics_axis.append(utility.spectral_centroid(fft_magnitude))
				elif metric_no == Features.Spec_Sprd.value:
					fft_magnitude = abs(fft(sub_frame[:, col]))
					fft_magnitude = fft_magnitude[0:int(sub_frame.shape[0]/2)]
					fft_magnitude = fft_magnitude / len(fft_magnitude)
					metrics_axis.append(utility.spectral_spread(fft_magnitude))
				# elif metric_no == Features.Spec_Flux.value:
					# fft_magnitude = abs(fft(sub_frame[:, col]))
					# fft_magnitude = fft_magnitude[0:sub_frame.shape[0]/2]
					# fft_magnitude = fft_magnitude / len(fft_magnitude)
					# metrics_axis.append(utility.spectral_flux(fft_magnitude, fft_magnitude_previous[col-1]))
					# fft_magnitude_previous[col-1] = fft_magnitude.copy()
				elif metric_no == Features.Spec_RollOff.value:
					fft_magnitude = abs(fft(sub_frame[:, col]))
					fft_magnitude = fft_magnitude[0:int(sub_frame.shape[0]/2)]
					fft_magnitude = fft_magnitude / len(fft_magnitude)
					metrics_axis.append(utility.spectral_rolloff(fft_magnitude))
				elif metric_no == Features.Max_freq.value:
					fft_magnitude = abs(fft(sub_frame[:, col]))
					fft_magnitude = fft_magnitude[0:int(sub_frame.shape[0]/2)]
					fft_magnitude = fft_magnitude / len(fft_magnitude)
					metrics_axis.append(max(fft_magnitude))
				elif metric_no == Features.Skewness.value:
					metrics_axis.append(utility.skewness(sub_frame[:, col]))
				elif metric_no == Features.Kurtosis.value:
					metrics_axis.append(kurtosis(sub_frame[:, col]))
				elif metric_no == Features.Avg_Power.value:
					metrics_axis.append(utility.avg_power(sub_frame[:, col]))

			full_frame.append(metrics_axis)
			sub_frame = list()

	full_frame = np.array(full_frame)

	# Pickling this data
	filename = str('Pickles/' + str(FeatureType[Features(metric_no)]) + '_all_per_sec_all_axis.pkl')
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

	df1 = pd.DataFrame.from_records(full_frame, columns = ['t'] + [str(str(metric_no) + names) for names in col_names] )
	print("df1 created !!!!")

	if (metric_no <= 13):
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
		df2 = pd.DataFrame.from_records(diff_frame, columns = [str("d" + str(metric_no) + names) for names in col_names])
		print("df2 created !!!!")

		result_df = pd.concat([df1, df2], axis=1)
		outputFileName = "Pickles/Metric_" + str(metric_no) + "_36.pkl"
	else:
		result_df = df1
		outputFileName = "Pickles/Metric_" + str(metric_no) + "_18.pkl"
	
	print(result_df.shape)

	# Pickle this data
	outfile = open(outputFileName, 'wb')
	pickle.dump(result_df, outfile)
	outfile.close()
	return outputFileName


if __name__ == "__main__":
	# pids = seperate_pid_data()
	pidFile = 'data/BK7610'
	for i in Features:
		if i.value == 12: continue
		filename = create_per_second_data(pidFile, i.value)
		filename = create_per_window_data(filename, i.value)
