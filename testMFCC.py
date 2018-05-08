from MFCC.exactMFCC import *
import scipy.io.wavfile as wav


# 读取文件采样频率和信号数组
(rate, sig) = wav.read("../0000aec1b2c4507a706ad7863eb0968a.wav")
print(rate, sig.shape)
mfcc_feat = calcMFCC_delta_delta(sig, rate)
print(mfcc_feat.shape)
# print(mfcc_feat)
