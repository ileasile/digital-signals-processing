import numpy as np
from mpl_backend_workaround import MPL_BACKEND_USED
import matplotlib.pyplot as plt
from config import CONFIG
from scipy import ndimage
import csv as csv

def get_data():
    file_name = CONFIG['file_name']
    with open(file_name, 'r') as f:
        first_line = f.readline()
        h = float(first_line.split(':')[1])
        s_val = f.readlines()
        data = np.ndarray((len(s_val),), dtype=np.float64)
        for i, val in enumerate(s_val):
            data[i] = float(val)
        return h, data

def show_save(x, y, title:str, x_label:str='', y_label:str=''):
    plt.close()
    plt.plot(x, y)
    # plt.title(title)
    plt.tight_layout(2)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(title+'.png')
    #plt.show()

def extract_harmonics(frequencies, amplitudes, threshold = None):
    """
    Extract amplitudes and frequencies of (probably) harmonic components
    :param frequencies: array of frequencies
    :param amplitudes: array of amplitudes
    :param threshold: Threshold of <valuable> component
    :return: arrays of valuable frequencies and amplitudes
    """
    th = threshold if threshold is not None else CONFIG['threshold']
    labels, num_labels = ndimage.label(amplitudes > th)
    unique_labels = np.unique(labels)
    idx = np.array(ndimage.maximum_position(amplitudes, labels, unique_labels[1:])).reshape((num_labels, ))

    return frequencies[idx], amplitudes[idx]

def print_components(csv_title, title, frequencies, amplitudes):
    csv_name = csv_title + '.csv'
    with open(csv_name, 'w') as csv_file:
        wr = csv.writer(csv_file, delimiter=',')

        break_out = '='*60
        c_f = 2 * np.pi * frequencies
        print('\n{}:'.format(title))
        print(break_out)
        wr.writerow(['w', 'nu', 'ampl'])
        print('Circular freq\t\t\tFreq\t\t\t\t\tA')
        for i in range(frequencies.shape[0]):
            l = [c_f[i], frequencies[i], amplitudes[i]]
            wr.writerow(l)
            print('{}\t\t{}\t\t{}'.format(*l))
        print(break_out)
        print('')

def modified_signal(time, s1, s2, a = CONFIG['add_amplitude']):
    return a * (np.cos(s1 * 2 * np.pi * time) + 2 * np.cos(s2 * 2 * np.pi * time))

def main():
    freq_word = 'Частота, Гц'
    ampl_word = 'Амплитуда'

    print("Matplotlib backend: {}".format(MPL_BACKEND_USED))

    h, data = get_data()
    n = data.shape[0]
    n_2 = n // 2
    t_all = n * h

    print('Nyquist frequency: {}'.format(1 / (2 * h)))
    print('     ... circular: {}'.format(np.pi / h))

    time_line = np.linspace(0, t_all, n)
    # show_save(time_line, data, 'Figure0')

    freq_line = np.arange(n) / t_all
    freq_line_2 = freq_line[:n_2]
    data_t = np.fft.fft(data, n)
    data_t_amplitudes = (np.abs(data_t) / n * 2)[:n_2]


    show_save(freq_line_2, data_t_amplitudes, 'Figure1', freq_word, ampl_word)
    harmonics_1 = extract_harmonics(freq_line_2, data_t_amplitudes)
    print_components('clear', 'Clear signal components', *harmonics_1)

    noise_configs = [
        (1.5, 'Figure2', 'OK noise'),
        # (2, 'Figure3', 'Not-OK noise'),
    ]

    for noise_a, figure_title, log_title in noise_configs:
        noise = np.random.rand(n) * (noise_a * 2) - noise_a

        noised_signal = data + noise
        noised_t = np.fft.fft(noised_signal, n)
        noised_t_amplitudes = (np.abs(noised_t) / n * 2)[:n_2]

        show_save(freq_line_2, noised_t_amplitudes, figure_title, freq_word, ampl_word)
        harmonics_2 = extract_harmonics(freq_line_2, noised_t_amplitudes)
        print_components('csv'+figure_title, log_title, *harmonics_2)

    add_amplitude = CONFIG['add_amplitude']
    add_configs = [
        (25, 27, 'Figure4', 'Modified signal (no aliasing)'),
        (25, 33, 'Figure5', 'Modified signal (aliasing + strange behavior)'),
    ]

    for s1, s2, figure_title, log_title in add_configs:
        add_signal = modified_signal(time_line, s1, s2, add_amplitude)
        mod_sig = data + add_signal
        mod_t = np.fft.fft(mod_sig, n)
        mod_t_a = (np.abs(mod_t) / n * 2)[:n_2]

        show_save(freq_line_2, mod_t_a, figure_title, freq_word, ampl_word)
        harmonics_4 = extract_harmonics(freq_line_2, mod_t_a)
        print_components('csv'+figure_title, log_title, *harmonics_4)

    add_signal = modified_signal(time_line, 20, 25)
    mod_sig2 = add_signal[::2] + data[::2]
    freq_line_4 = (np.arange(n//2) / (n * h))[:n//4]
    mod_t2 = np.fft.fft(mod_sig2)
    mod_t_a2 = (np.abs(mod_t2) / (n//2) * 2)[:n_2//2]

    show_save(freq_line_4, mod_t_a2, 'Figure6', freq_word, ampl_word)
    harmonics_6 = extract_harmonics(freq_line_4, mod_t_a2)
    print_components('reduced','Modified signal components (only even samples)', *harmonics_6)


if __name__ == "__main__":
    main()