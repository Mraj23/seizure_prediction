import numpy as np
from matplotlib import pyplot as plt
from scipy.fftpack import fft, fftfreq

tedata = np.load('x_test.npy')
trdata = np.load('x_train.npy')


def create_power_spectra(data):
    new_data = []
    for ex in data:
        ex = np.delete(ex, [0,1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17,18], 0)
        new_data.append([])
        for ch in ex:
            yf = fft(ch)
            plt.plot(np.linspace(0,70,70), np.abs(yf[:70]))
            plt.show()
            new_data[-1].append(np.abs(yf)[:70])
        
    new_data = np.array(new_data)
    return new_data

train = create_power_spectra(trdata)
print(train.shape)
test = create_power_spectra(tedata)
print(test.shape)


np.save("x_train_power_bad_removed_2", train)
np.save("x_test_power_bad_removed_2", test)
