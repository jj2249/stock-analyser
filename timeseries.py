import numpy as np
from plotting import dates, closing_prices
import pandas as pd
import matplotlib.pyplot as plt

N = closing_prices.shape[0]
print(N)

all_ones = np.ones(dates.shape[0])
dates_num = pd.to_numeric(dates)
dates_sq = np.square(dates_num)
dates_enum = np.array([i for i in range(dates.shape[0])])
dates_enum_sq = np.array([i*i for i in range(dates.shape[0])])
dates_enum_cu = np.array([i*i*i for i in range(dates.shape[0])])
dates_enum_4 = np.array([i*i*i*i for i in range(dates.shape[0])])

XT1 = np.vstack((all_ones, dates_enum))
X1 = np.transpose(XT1)

XT2 = np.vstack((all_ones, dates_enum, dates_enum_sq))
X2 = np.transpose(XT2)

XT3 = np.vstack((all_ones, dates_enum, dates_enum_sq, dates_enum_cu))
X3 = np.transpose(XT3)

XT4 = np.vstack((all_ones, dates_enum, dates_enum_sq, dates_enum_cu, dates_enum_4))
X4 = np.transpose(XT4)

beta1 = np.linalg.inv(XT1.dot(X1)).dot(XT1).dot(closing_prices)
beta2 = np.linalg.inv(XT2.dot(X2)).dot(XT2).dot(closing_prices)
beta3 = np.linalg.inv(XT3.dot(X3)).dot(XT3).dot(closing_prices)
beta4 = np.linalg.inv(XT4.dot(X4)).dot(XT4).dot(closing_prices)


fit1 = X1.dot(beta1)
fit2 = X2.dot(beta2)
fit3 = X3.dot(beta3)
fit4 = X4.dot(beta4)

fitMA = [(closing_prices[i-1]+closing_prices[i]+closing_prices[i+1])/3 for i in range(1, (closing_prices.shape[0]-1))]
fitMA.append(closing_prices[-1])
fitMA.insert(0, closing_prices[0])

plt.scatter(dates_enum, closing_prices, s=1)
plt.plot(dates_enum, fit1)
plt.plot(dates_enum, fit2)
plt.plot(dates_enum, fit3)
plt.plot(dates_enum, fit4)
plt.plot(dates_enum, fitMA)
plt.show()

SE1 = np.array([(fit1[i] - closing_prices[i])**2 for i in range(fit1.shape[0])])
SE2 = np.array([(fit2[i] - closing_prices[i])**2 for i in range(fit2.shape[0])])
SE3 = np.array([(fit3[i] - closing_prices[i])**2 for i in range(fit3.shape[0])])
SE4 = np.array([(fit4[i] - closing_prices[i])**2 for i in range(fit4.shape[0])])

SSE1 = np.sum(SE1)
SSE2 = np.sum(SE2)
SSE3 = np.sum(SE3)
SSE4 = np.sum(SE4)

MSE1 = SSE1/N
MSE2 = SSE2/N
MSE3 = SSE3/N
MSE4 = SSE4/N


print("Linear MSE: {}, Quadratic MSE: {}, Cubic MSE: {}, Quartic MSE: {}".format(MSE1, MSE2, MSE3, MSE4))

# Quadratic fit is very very very slightly better
resids = closing_prices - fit3
plt.plot(dates_enum, resids)
plt.show()
plt.hist(resids, bins=15)
plt.show()

pgram = np.abs(np.fft.fft(resids, N)/N)**2
indices = np.linspace(0, (N-1), num = N)
plt.plot(indices, pgram)
plt.show()

top_inds = indices[(pgram > 0.5*np.max(pgram))]
#note that this is in rad/s and the period is '1' week
max_freq = 2*np.pi*top_inds[0]/N 

XTf = np.vstack((np.sin(max_freq*dates_enum), np.cos(max_freq*dates_enum)))
Xf = np.transpose(XTf)
betaf = np.linalg.inv(XTf.dot(Xf)).dot(XTf).dot(resids)
fitf = Xf.dot(betaf)
print(betaf)

plt.plot(dates_enum, resids)
plt.plot(dates_enum, fitf)
plt.show()

comb_fit = fitf + fit3
plt.plot(dates_enum, closing_prices)
plt.plot(dates_enum, comb_fit)
plt.show()







