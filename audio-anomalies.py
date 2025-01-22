import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import argparse
import soundfile as sf

def parse_arguments():
    parser = argparse.ArgumentParser(description="Detect anomalies in stereo WAV files based on spectral differences.")
    parser.add_argument("input_file", type=str, help="Path to the WAV file to analyze.")
    parser.add_argument("--threshold", type=float, default=15.0, help="Spectral difference threshold (default: 15.0 dB).")
    parser.add_argument("--hop_time", type=float, default=0.2, help="Time between analysis windows in seconds (default: 0.2).")
    parser.add_argument("--threshold_time_gap", type=float, default=0.5, help="Threshold time gap (default: 0.5 s) for merging nearby anomalies.")
    parser.add_argument("--plot", action="store_true", help="Show a plot of the spectral differences.")
    return parser.parse_args()

def analyze_audio(input_file, threshold, hop_time, threshold_time_gap, show_plot):
    # WAV-Datei mit soundfile laden
    y, sr = sf.read(input_file, always_2d=True)
    y = y.T  # Kanäle transponieren, damit die Struktur mit librosa übereinstimmt   
    if y.ndim != 2 or y.shape[0] != 2:
        raise ValueError("Input file is not a stereo WAV file.")

    left, right = y  # Linken und rechten Kanal extrahieren
    
    # Spektralanalyse mit Fast Fourier Transform (FFT)
    S_left = librosa.amplitude_to_db(np.abs(librosa.stft(left)), ref=np.max)
    S_right = librosa.amplitude_to_db(np.abs(librosa.stft(right)), ref=np.max)

    # Spektrale Unterschiede berechnen
    spectral_diff = np.abs(S_left - S_right)
    max_diff = np.max(spectral_diff)

    print(f"Max spectral difference: {max_diff:.2f} dB")

    # Anomalien erkennen und filtern
    anomalies = []
    hop_length = int(hop_time * sr)  # Berechne hop_length korrekt

    for t in range(spectral_diff.shape[1]):
        if spectral_diff[:, t].max() > threshold:
            time_point = librosa.frames_to_time(t, sr=sr)  # Entferne hop_length
            formatted_time = format_time(time_point)  # Formatierte Zeit im h:mm:ss-Format
            channel = "Left" if S_left[:, t].max() > S_right[:, t].max() else "Right"
            diff_value = spectral_diff[:, t].max()  # Maximaler Spektraldifferenzwert für das aktuelle Frame
            anomalies.append((formatted_time, time_point, channel, diff_value))

    # Duplikate oder nahe beieinander liegende Anomalien zusammenfassen
    if anomalies:
        anomalies = filter_nearby_anomalies(anomalies, threshold_time_gap)

    if anomalies:
        print("Anomalies detected!")
        print(f"{'Time (h:mm:ss)':<15}{'Channel':<10}{'Spectral Difference (dB)'}")  # Spaltenüberschrift
        for anomaly in anomalies:
            print(f"{anomaly[0]:<15}{anomaly[2]:<10}{anomaly[3]:.2f}")  # Ausgabe der Anomalien mit Spektraldifferenz
    else:
        print("No significant anomalies detected.")

    # Optional: Plot anzeigen
    if show_plot:
        plt.figure(figsize=(10, 6))
        librosa.display.specshow(spectral_diff, sr=sr, x_axis="time", y_axis="log")
        plt.colorbar(format="%+2.0f dB")
        plt.title("Spectral Difference (Left vs Right)")
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.show()

def filter_nearby_anomalies(anomalies, threshold_time_gap=0.5):
    """Fasse nahe beieinander liegende Anomalien zusammen."""
    filtered_anomalies = []
    last_time = -float('inf')
    for anomaly in anomalies:
        if anomaly[1] - last_time > threshold_time_gap:
            filtered_anomalies.append(anomaly)
            last_time = anomaly[1]
    return filtered_anomalies

def format_time(seconds):
    """Formatierte Ausgabe der Zeit in h:mm:ss.x."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = round(seconds % 60, 1)  # Runden auf 1 Nachkommastelle
    return f"{hours:01}:{minutes:02}:{seconds:04.1f}"  # Format mit 1 Nachkommastelle

# Hauptprogramm starten
if __name__ == "__main__":
    args = parse_arguments()
    analyze_audio(args.input_file, args.threshold, args.hop_time, args.threshold_time_gap, args.plot)