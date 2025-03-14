import os
import pandas as pd
import matplotlib.pyplot as plt


results_path = 'runs/classify/train/results.csv'
results = pd.read_csv(results_path)

#Loss vs epochs
plt.figure()
plt.plot(results['epoch'], results['train/loss'], label='train loss')
plt.plot(results['epoch'], results['val/loss'], label='val loss', c='red')
plt.grid()
plt.title('Loss vs epochs')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend()

#Validation accuracy vs epochs
plt.figure()
plt.plot(results['epoch'], results['metrics/accuracy_top1'] * 100)
plt.grid()
plt.title('Validation accuracy vs epochs')
plt.ylabel('accuracy (%)')
plt.xlabel('epochs')

#epoch vs time
plt.figure()
plt.plot(results['epoch'], results['time'])
plt.grid()
plt.title('Epoch vs time')
plt.ylabel('time (s)')
plt.xlabel('epochs')


plt.show()