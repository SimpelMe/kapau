import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import argparse
import soundfile as sf
import sys

def parse_arguments():
    """Definiert und analysiert die Kommandozeilenargumente."""
    parser = argparse.ArgumentParser(description="Detect anomalies in a stereo WAV file or two mono WAV files based on spectral differences and correlations.")
    parser.add_argument("input_file", nargs="+", help="Path to the WAV file(s) to analyze. Provide one stereo file or two mono files.")
    parser.add_argument("--threshold", type=float, default=60.0, help="Spectral difference threshold (default: 60.0 dB).")
    parser.add_argument("--threshold_time_gap", type=float, default=0.5, help="Time gap to ignore nearby anomalies (default: 0.5 s).")
    parser.add_argument("--plot", action="store_true", help="Show a plot of the spectral differences.")
    return parser.parse_args()

def load_audio_files(input_files):
    """
    Lädt entweder eine Stereo-Audiodatei oder zwei Mono-Audiodateien.
    Überprüft die Konsistenz der Eingabedateien.
    """
    if len(input_files) == 1:
        file_path = input_files[0]
        y, sr = sf.read(file_path, always_2d=True)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(f"The file '{file_path}' is not a stereo WAV file.")
        return y.T, sr
    elif len(input_files) == 2:
        left_path, right_path = input_files
        y_left, sr_left = sf.read(left_path)
        y_right, sr_right = sf.read(right_path)
        if y_left.ndim != 1 or y_right.ndim != 1:
            raise ValueError("Both files must be mono WAV files.")
        if sr_left != sr_right:
            raise ValueError("The sample rates of the two files do not match.")
        if len(y_left) != len(y_right):
            raise ValueError("The lengths of the two files do not match.")
        return np.array([y_left, y_right]), sr_left
    else:
        raise ValueError("Provide either one stereo WAV file or two mono WAV files.")

def calculate_correlation_formel(signal1, signal2):
    """
    Correlation nach Formel:
    correlation = (sum(signal1 * signal2)) / sqrt(sum(signal1^2) * sum(signal2^2))
    """
    if np.all(signal1 == 0) and np.all(signal2 == 0):
        return 1

    inner_product = np.sum(signal1 * signal2)
    square_product = np.sqrt(np.sum(signal1**2) * np.sum(signal2**2))
    if square_product == 0:
        return 0.0

    return round(inner_product / square_product, 6)

def analyze_audio(input_files, threshold, threshold_time_gap, show_plot):
    """
    Analysiert die Audiodaten und erkennt Anomalien basierend auf spektralen Unterschieden und Korrelationen.
    """
    y, sr = load_audio_files(input_files)
    left, right = y

    S_left = librosa.amplitude_to_db(np.abs(librosa.stft(left)), ref=np.max)
    S_right = librosa.amplitude_to_db(np.abs(librosa.stft(right)), ref=np.max)

    spectral_diff = np.abs(S_left - S_right)

    anomalies = []
    window_size = int(sr * 0.1)  # 100ms Fenster für Korrelation

    for t in range(spectral_diff.shape[1]):
        if spectral_diff[:, t].max() > threshold:
            time_point = librosa.frames_to_time(t, sr=sr)
            formatted_time = format_time(time_point)

            start_sample = int(librosa.frames_to_samples(t))
            end_sample = start_sample + window_size
            if end_sample > len(left):
                end_sample = len(left)

            corr_value = calculate_correlation_formel(left[start_sample:end_sample], right[start_sample:end_sample])

            channel = "Left" if S_left[:, t].max() > S_right[:, t].max() else "Right"
            diff_value = spectral_diff[:, t].max()
            anomalies.append((formatted_time, time_point, channel, diff_value, corr_value))

    if anomalies:
        anomalies = filter_nearby_anomalies(anomalies, threshold_time_gap)

    if anomalies:
        print("Anomalies detected!")
        print(f"{'h:mm:ss.xx':<15}{'Channel':<10}{'Spectral Diff (dB)':<20}{'Correlation'}")
        for anomaly in anomalies:
            print(f"{anomaly[0]:<15}{anomaly[2]:<10}{anomaly[3]:<20.2f}{anomaly[4]:.6f}")
    else:
        print("No significant anomalies detected.")

    if show_plot:
        plt.figure(figsize=(10, 12))

        plt.subplot(2, 1, 1)
        librosa.display.specshow(spectral_diff, sr=sr, x_axis="time", y_axis="log")
        plt.colorbar(format="%+2.0f dB")
        plt.title("Spectral Difference (Left vs Right)")
        plt.xlabel("Time")
        plt.ylabel("Frequency")

        plt.subplot(2, 1, 2)
        correlation_values = [anomaly[4] for anomaly in anomalies]
        times = [anomaly[1] for anomaly in anomalies]
        plt.plot(times, correlation_values, label="Correlation", color="green")
        plt.xlabel("Time (s)")
        plt.ylabel("Correlation")
        plt.title("Correlation Over Time")
        plt.legend()

        plt.tight_layout()
        plt.show()

def filter_nearby_anomalies(anomalies, threshold_time_gap=0.5):
    filtered_anomalies = []
    last_time = -float('inf')
    current_window = []
    for anomaly in anomalies:
        if anomaly[1] - last_time <= threshold_time_gap:
            current_window.append(anomaly)
        else:
            if current_window:
                max_diff_anomaly = max(current_window, key=lambda x: x[3])
                filtered_anomalies.append(max_diff_anomaly)
            current_window = [anomaly]
        last_time = anomaly[1]

    if current_window:
        max_diff_anomaly = max(current_window, key=lambda x: x[3])
        filtered_anomalies.append(max_diff_anomaly)

    return filtered_anomalies

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = round(seconds % 60, 2)
    return f"{hours:01}:{minutes:02}:{seconds:04.2f}"

if __name__ == "__main__":
    try:
        args = parse_arguments()
        analyze_audio(args.input_file, args.threshold, args.threshold_time_gap, args.plot)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
