# Implementation Project for Detect-Heavy-Drinking-Episodes paper
Implementation of Research work in "Learning to Detect Heavy Drinking Episodes Using Smartphone Accelerometer Data" - Jackson A Killian et al.

# Requirements
1. Pandas
2. Scikit
3. Scipy
4. Numpy
5. Other Python3 libs

# Dataset:
http://archive.ics.uci.edu/ml/datasets/Bar+Crawl%3A+Detecting+Heavy+Drinking  

# References:
http://ceur-ws.org/Vol-2429/paper6.pdf  
https://github.com/tyiannak/pyAudioAnalysis  
https://web.cs.wpi.edu/~emmanuel/publications/PDFs/C17.pdf  
https://librosa.github.io/librosa/index.html  

# How to run:
All the features generated are saved as pickle files. Hence we can directly run the classifier and see the output.  
So if you just want to run the classifier:  
$ python3 rft.py  

If you want to generate all the features again (This may take close to a few hour depending on the CPU):  
$ python3 eda_edits_copy.py  
$ python3 rft.py  
