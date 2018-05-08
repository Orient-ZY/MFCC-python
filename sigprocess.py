import numpy as np
import math


# 音频转换为帧矩阵
def audio2frame(signal, frame_length, frame_step, winfunc=lambda x : np.ones((x,))):
    '''
    将音频转化为帧
    :param signal: 原始音频信号
    :param frame_length: 每一帧的长度（采样点的长度，即采样频率诚意时间间隔）
    :param frame_step: 相邻帧的间隔
    :param winfunc: 生成一个向量
    :return:
    '''
    signal_length = len(signal)  # 信号总长度
    frame_length = int(round(frame_length))  # 一帧帧时间长度
    frame_step = int(round(frame_step))  # 相邻帧之间步长
    if signal_length <= frame_length:  # 若信号长度小于一个帧长度，则帧数定义为1
        frames_num = 1
    else:  # 否则，计算帧总长度
        frames_num = 1 + int(math.ceil((1.0*signal_length-frame_length)/frame_step))
    pad_length = int((frames_num-1)*frame_step+frame_length)  # 所有帧加起来总的铺平后的长度
    zeros = np.zeros((pad_length-signal_length,))  # 不够的长度使用0填补
    pad_signal = np.concatenate((signal, zeros))  # 填补后的信号记为pad_signal
    # 相当于对所有帧的时间点进行抽取，得到frames_num*frame_length长度的矩阵
    indices = np.tile(np.arange(0, frame_length), (frames_num, 1))\
        + np.tile(np.arange(0, frames_num*frame_step, frame_step), (frame_length, 1)).T
    indices = np.array(indices, dtype=np.int32)  # 将indices转为矩阵
    frames = pad_signal[indices]  # 得到帧信号
    win = np.tile(winfunc(frame_length), (frames_num, 1))  # window窗函数，这里默认取1
    return frames * win  # 返回帧信号矩阵


# 对每一帧做一个消除关联的变换
def deframesignal(frames, signal_length, frame_length, frame_step, winfunc=lambda x : np.ones((x,))):
    '''
    定义函数对原信号的每一帧进行变换
    :param frames: audio2frame函数返回的帧矩阵
    :param signal_length: 信号长度
    :param frame_length: 帧长度
    :param freame_step: 帧间隔
    :param winfunc: 对每一帧加window函数进行分析，默认此处不加window
    :return:
    '''
    signal_length = round(signal_length)  # 信号长度，返回浮点数的四舍五入值
    frame_length = round(frame_length)  # 帧长度
    frames_num = np.shape(frames)[0]  # 帧的总数
    assert np.shape(frames)[1] == frame_length, '"frames"矩阵大小不正确，它的列数应该等于一帧长度'  # 判断frames维度
    # 相当于对所有帧的时间点进行抽取，得到frames_num*frame_length长度的矩阵
    indices = np.tile(np.arange(0, frame_length), (frames_num, 1))\
        + np.tile(np.arange(0, frames_num*frame_step), (frame_length, 1)).T
    indices = np.array(indices, dtype=np.int32)
    pad_length = (frames_num - 1) * frame_step + frame_length  # 铺平后的所有信号
    if signal_length <= 0:
        signal_length = pad_length
    recalc_signal = np.zeros((pad_length,))  # 调整后的信号
    window_correction = np.zeros((pad_length, 1))  # 窗关联
    win = winfunc(frame_length)
    for i in range(0, frames_num):
        window_correction[indices[i, :]] = window_correction[indices[i, :]] + win + 1e-15  # 表示信号的重叠程度
        recalc_signal[indices[i, :]] = recalc_signal[indices[i, :]] + frames[i, :]  # 原信号加上重叠程度构成调整后的信号
    recalc_signal = recalc_signal / window_correction  # 新的调整后的信号等于调整信号以每处的重叠程度
    return recalc_signal[0:signal_length]  # 返回该新的调整信号


# 计算每一帧傅里叶变换以后的幅度
def spectrum_magnitude(frames, NFFT):
    '''
    计算每一帧经过FFT变换后频谱的幅度，若frames大小为N*L，则返回矩阵的大小为N*NFFT
    :param frames: audio2frame函数返回的矩阵，帧矩阵
    :param NFFT: FFT变换的数组大小，若帧长度小于NFFT，则帧的其余部分用0填充
    :return:
    '''
    complex_spectrum = np.fft.rfft(frames, NFFT)  # 对frames进行FFT变换
    return np.absolute(complex_spectrum)  # 返回频谱的幅度值


# 计算每一帧傅里叶变换后的功率谱
def spectrum_power(frames, NFFT):
    '''
    :param frames: audio2frame函数返回的帧矩阵
    :param NFFT: FFT大小
    :return:
    '''
    # 功率谱等于每一点的幅度平方/NFFT
    return 1.0 / NFFT * np.square(spectrum_magnitude(frames, NFFT))


# 计算每一帧傅里叶变换后的对数功率谱
def log_spectrum_power(frames, NFFT, norm=1):
    '''
    :param frames: 帧矩阵
    :param NFFT: FFT变换大小
    :param norm: 归一化系数
    :return:
    '''
    spec_power = spectrum_power(frames, NFFT)
    spec_power[spec_power<1e-30] = 1e-30  # 防止出现功率谱等于0
    log_spec_power = 10 * np.log10(spec_power)
    if norm:
        return log_spec_power - np.max(log_spec_power)
    else:
        return log_spec_power


# 对原始信号进行预加重处理
def pre_emphasis(signal, coefficient=0.95):
    '''
    :param signal: 原始信号
    :param coefficient: 加重系数
    :return:
    '''
    return np.append(signal[0], signal[1:]-coefficient*signal[:-1])

