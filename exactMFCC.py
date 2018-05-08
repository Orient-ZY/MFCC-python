import numpy as np
from MFCC.sigprocess import audio2frame
from MFCC.sigprocess import pre_emphasis
from MFCC.sigprocess import spectrum_power
from scipy.fftpack import dct

try:
    xrange(1)
except:
    xrange = range


def lifter(cepstra, L=22):
    '''
    升倒谱函数
    :param cepstra: MFCC系数
    :param L: 升系数，默认22
    :return:
    '''
    if L > 0:
        nframes, ncoeff = np.shape(cepstra)
        n = np.arange(ncoeff)
        lift = 1 + (L/2) * np.sin(np.pi*n/L)
        return lift*cepstra
    else:
        return cepstra


def get_filter_banks(filters_num=20, NFFT=512, samplerate=16000, low_freq=0, high_freq=None):
    '''
    计算梅尔三角间距滤波器，在第一、三个频率处为0，第二个为1
    :param filters_num: 滤波器个数
    :param NFFT:
    :param samplerate: 采样频率
    :param low_freq:
    :param high_freq:
    :return:
    '''
    low_mel = hz2mel(low_freq)
    high_mel = hz2mel(high_freq)  # 将hz转为梅尔频率
    mel_points = np.linspace(low_mel, high_mel, filters_num+2)  # 在low_mel和high_mel之间等距插入filters_num个点，一个filters_num+2个点
    hz_points = mel2hz(mel_points)  # 再讲梅尔频率转为hz，并找到对应的hz位置
    bin = np.floor((NFFT + 1) * hz_points / samplerate)  # 需知道hz_points对应到fft中的位置
    # 接下来建立滤波器的表达式了，每个滤波器在第一个点处和第三个点处均为0，中间为三角形形状
    fbank = np.zeros([filters_num, int(NFFT / 2 + 1)])
    for j in xrange(0, filters_num):
        for i in xrange(int(bin[j]), int(bin[j+1])):
            fbank[j, i] = (i-bin[j]) / (bin[j+1]-bin[j])
        for i in xrange(int(bin[j+1]), int(bin[j+2])):
            fbank[j, i] = (bin[j+2]-i) / (bin[j+2]-bin[j+1])
    return fbank


def ssc(signal, samplerate=16000, win_length=0.025, win_step=0.01, filters_num=26, NFFT=512, low_freq=0, high_freq=None, pre_emphasis_coeff=0.97):
    high_freq = high_freq or samplerate / 2
    signal = pre_emphasis(signal, pre_emphasis_coeff)
    frames = audio2frame(signal, win_length*samplerate, win_step*samplerate)
    spec_power = spectrum_power(frames, NFFT)
    spec_power = np.where(spec_power==0, np.finfo(float).eps, spec_power)  # 能量谱
    fb = get_filter_banks(filters_num, NFFT, samplerate, low_freq, high_freq)
    feat = np.dot(spec_power, fb.T)  # 计算能量
    R = np.tile(np.linspace(1, samplerate/2, np.size(spec_power, 1)), (np.size(spec_power, 0), 1))
    return np.dot(spec_power*R, fb.T) / feat


def hz2mel(hz):
    '''
    频率转梅尔频率
    :param hz:
    :return:
    '''
    return 2595*np.log10(1+hz/700.0)


def mel2hz(mel):
    '''
    梅尔频率转hz
    :param mel:
    :return:
    '''
    return 700*(10**(mel/2595.0)-1)


def fbank(signal, samplerate=16000, win_length=0.025, win_step=0.01, filters_num=26, NFFT=512, low_freq=0, high_freq=None, pre_emphasis_coeff=0.97):
    '''
    计算音频信号的MFCC
    :param signal:
    :param samplerate: 采样频率
    :param win_length: 窗长度
    :param win_step: 窗间隔
    :param filters_num: 梅尔滤波器个数
    :param NFFT: FFT大小
    :param low_freq: 最低频率
    :param high_freq: 最高频率
    :param pre_emphasis_coeff: 预加重系数
    :return:
    '''
    high_freq = high_freq or samplerate / 2  # 计算音频样本最大频率
    signal = pre_emphasis(signal, pre_emphasis_coeff)  # 对原始信号进行预加重处理
    frames = audio2frame(signal, win_length*samplerate, win_step*samplerate)  # 得到帧数组
    spec_power = spectrum_power(frames, NFFT)  # 得到每一帧FFT后的能量谱
    energy = np.sum(spec_power, 1)  # 对每一帧的能量谱求和
    energy = np.where(energy==0, np.finfo(float).eps, energy)  # 对能量为0的地方调整为eps，方便进行对数处理
    fb = get_filter_banks(filters_num, NFFT, samplerate, low_freq, high_freq)  # 获得每一个滤波器的频率宽度
    feat = np.dot(spec_power, fb.T)  # 对滤波器和能量谱点乘
    feat = np.where(feat==0, np.finfo(float).eps, feat)
    return feat, energy


def log_fbank(signal, samplerate=16000, win_length=0.025, win_step=0.01, filters_num=26, NFFT=512, low_freq=0, high_freq=None, pre_emphasis_coeff=0.97):
    '''
    计算对数值
    :param signal:
    :param samplerate:
    :param win_length:
    :param win_step:
    :param filters_num:
    :param NFFT:
    :param low_freq:
    :param high_freq:
    :param pre_emphasis_coeff:
    :return:
    '''
    feat, energy = fbank(
        signal,
        samplerate,
        win_length,
        win_step,
        filters_num,
        NFFT,
        low_freq,
        high_freq,
        pre_emphasis_coeff
    )
    return np.log(feat)


def derivate(feat, big_theta=2, cep_num=13):
    '''
    计算一阶系数或加速系数的一般变换公式
    :param feat: MFCC数组或一阶系数数组
    :param big_theta: 公式中的大theta，默认取2
    :param cep_num:
    :return:
    '''
    result = np.zeros(feat.shape)
    denominator = 0  # 分母
    for theta in np.linspace(1, big_theta, big_theta):
        denominator = denominator + theta ** 2
    denominator = denominator * 2  # 计算得到分母的值
    for row in np.linspace(0, feat.shape[0]-1, feat.shape[0]):
        tmp = np.zeros((cep_num,))
        numerator = np.zeros((cep_num,))  # 分子
        for t in np.linspace(1, cep_num, cep_num):
            a = 0
            b = 0
            s = 0
            for theta in np.linspace(1, big_theta, big_theta):
                if (t+theta) > cep_num:
                    a = 0
                else:
                    a = feat[int(row)][int(t+theta-1)]
                if (t-theta) < 1:
                    b = 0
                else:
                    b = feat[int(row)][int(t-theta-1)]
                s += theta*(a-b)
            numerator[int(t-1)] = s
        tmp = numerator * 1.0 / denominator
        result[int(row)] = tmp
    return result


def calcMFCC(signal, samplerate=16000, win_length=0.025, win_step=0.01, cep_num=13, filters_num=26, NFFT=512, low_freq=0, high_freq=None, pre_emphasis_coeff=0.97, cep_lifter=22, appendEnergy=True):
    '''
    计算13个MFCC系数
    :param signal:
    :param samplerate:
    :param win_length:
    :param win_step:
    :param cep_num:
    :param filters_num:
    :param NFFT:
    :param low_freq:
    :param high_freq:
    :param pre_emphasis_coeff:
    :param cep_lifter:
    :param appendEnergy:
    :return:
    '''
    feat, energy = fbank(
        signal,
        samplerate,
        win_length,
        win_step,
        filters_num,
        NFFT,
        low_freq,
        high_freq,
        pre_emphasis_coeff
    )
    feat = np.log(feat)
    feat = dct(feat, type=2, axis=1, norm='ortho')[:, :cep_num]  # 高阶离散余弦变换，只取前13个系数
    feat = lifter(feat, cep_lifter)
    if appendEnergy:
        feat[:, 0] = np.log(energy)
    return feat


def calcMFCC_delta(signal, samplerate=16000, win_length=0.025, win_step=0.01, cep_num=13, filters_num=26, NFFT=512, low_freq=0, hight_freq=None, pre_emphasis_coeff=0.97,cep_lifter=22, appendEnergy=True):
    '''
    计算13个MFCC+13个一阶微分系数
    :param signal:
    :param samplerate:
    :param win_length:
    :param win_step:
    :param cep_num:
    :param filters_num:
    :param NFFT:
    :param low_freq:
    :param hight_freq:
    :param pre_emphasis_coeff:
    :param cep_lifter:
    :param appendEnergy:
    :return:
    '''
    feat = calcMFCC(
        signal,
        samplerate,
        win_length,
        win_step,
        cep_num,
        filters_num,
        NFFT,
        low_freq,
        hight_freq,
        pre_emphasis_coeff,
        cep_lifter,
        appendEnergy
    )
    result = derivate(feat)
    result = np.concatenate((feat, result), axis=1)
    return result


def calcMFCC_delta_delta(signal, samplerate=16000, win_length=0.025, win_step=0.01, cep_num=13, filters_num=26, NFFT=512, low_freq=0, high_freq=None, pre_emphasis_coeff=0.97, cep_lifter=22, appendEnergy=True):
    '''
    计算13个MFCC+13个一阶微分系数+13个加速系数。共39个系数
    :param signal:
    :param samplerate:
    :param win_length:
    :param win_step:
    :param cep_num:
    :param filters_num:
    :param NFFT:
    :param low_freq:
    :param high_freq:
    :param pre_emphasis_coeff:
    :param cep_lifter:
    :param appendEnergy:
    :return:
    '''
    feat = calcMFCC(
        signal,
        samplerate,
        win_length,
        win_step,
        cep_num,
        filters_num,
        NFFT,
        low_freq,
        high_freq,
        pre_emphasis_coeff,
        cep_lifter,
        appendEnergy
    )  # 获取13个一般MFCC系数
    result1 = derivate(feat)
    result2 = derivate(result1)
    result3 = np.concatenate((feat, result1), axis=1)
    result = np.concatenate((result3, result2), axis=1)
    return result




