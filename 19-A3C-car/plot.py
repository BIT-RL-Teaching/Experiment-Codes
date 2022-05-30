from matplotlib import pyplot as plt
import csv
import numpy as np 

def moving_average(arr, alpha = 0.05):
	newarr = []
	temp = 0

	for v in arr:
		temp = (1 - alpha) * temp + alpha * v
		newarr.append(temp)

	return newarr


rs, ls = [], []
cr = csv.reader(open('log.csv', encoding='utf-8'))

# 跳过标题
next(cr)
# 得到周期和奖励的长度
for row in cr:
    rs.append(float(row[-2]))

plt.plot(rs, label = "raw reward", color = 'g')
plt.plot(moving_average(rs, 0.05), label = "moving average", color = 'r')

plt.plot(750 * np.ones(len(rs) + 200), label = "human level estimate", color = "c", linestyle='dashed')
plt.plot(950 * np.ones(len(rs) + 200), label = "max score estimate", color = "m", linestyle='dashed')

plt.xlabel("training time / 100s")
plt.ylabel("episode reward")
plt.legend()

plt.show()