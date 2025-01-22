import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import argparse
import soundfile as sf
import sys

def parse_arguments():
    """Definiert und analysiert die Kommandozeilenargumente."""
    parser = argparse.ArgumentParser(description="Detect anomalies in a stereo WAV file or two mono WAV files based on spectral differences.")
    parser.add_argument("input_file", nargs="+", help="Path to the WAV file(s) to analyze. Provide one stereo file or two mono files.")
    parser.add_argument("--threshold", type=float, default=60.0, help="Spectral difference threshold (default: 60.0 dB).")
    parser.add_argument("--hop_time", type=float, default=0.2, help="Time between analysis windows (default: 0.2 s).")
    parser.add_argument("--threshold_time_gap", type=float, default=0.5, help="Time gap to ignore nearby anomalies (default: 0.5 s).")
    parser.add_argument("--plot", action="store_true", help="Show a plot of the spectral differences.")
    return parser.parse_args()

def load_audio_files(input_files):
    """
    Lädt entweder eine Stereo-Audiodatei oder zwei Mono-Audiodateien.
    Überprüft die Konsistenz der Eingabedateien.
    """
    if len(input_files) == 1:
        # Einzelne Datei bereitgestellt, prüfen, ob sie stereo ist
        file_path = input_files[0]
        y, sr = sf.read(file_path, always_2d=True)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(f"The file '{file_path}' is not a stereo WAV file.")
        return y.T, sr  # Rückgabe der Stereo-Kanäle und der Abtastrate
    elif len(input_files) == 2:
        # Zwei Dateien bereitgestellt, prüfen, ob sie mono und kompatibel sind
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

def analyze_audio(input_files, threshold, hop_time, threshold_time_gap, show_plot):
    """
    Analysiert die Audiodaten und erkennt Anomalien basierend auf spektralen Unterschieden.
    """
    # Lade die Audiodateien
    y, sr = load_audio_files(input_files)
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
    hop_length = int(hop_time * sr)  # Berechne hop_length aus der Zeitauflösung

    for t in range(spectral_diff.shape[1]):
        if spectral_diff[:, t].max() > threshold:
            # Zeitpunkt berechnen
            time_point = librosa.frames_to_time(t, sr=sr)
            formatted_time = format_time(time_point)  # Zeit formatieren
            # Kanal mit der größeren Differenz bestimmen
            channel = "Left" if S_left[:, t].max() > S_right[:, t].max() else "Right"
            diff_value = spectral_diff[:, t].max()  # Maximaler Spektraldifferenzwert
            anomalies.append((formatted_time, time_point, channel, diff_value))

    # Nahe beieinander liegende Anomalien zusammenfassen
    if anomalies:
        anomalies = filter_nearby_anomalies(anomalies, threshold_time_gap)

    # Ergebnisse anzeigen
    if anomalies:
        print("Anomalies detected!")
        print(f"{'Time (h:mm:ss.x)':<15}{'Channel':<10}{'Spectral Difference (dB)'}")
        for anomaly in anomalies:
            print(f"{anomaly[0]:<15}{anomaly[2]:<10}{anomaly[3]:.2f}")
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
    """
    Identifiziert die größten Sprünge innerhalb benachbarter Anomalien und gibt diese als Anomalien aus.
    """
    filtered_anomalies = []
    last_time = -float('inf')  # Initialer Vergleichszeitpunkt
    current_window = []  # Liste von Anomalien im aktuellen Zeitfenster
    for anomaly in anomalies:
        if anomaly[1] - last_time <= threshold_time_gap:
            # Anomalie gehört zum aktuellen Zeitfenster
            current_window.append(anomaly)
        else:
            # Wenn das Zeitfenster überschritten wurde, den größten Sprung im aktuellen Fenster ermitteln
            if current_window:
                max_diff_anomaly = max(current_window, key=lambda x: x[3])  # Anomalie mit dem größten Differenzwert
                filtered_anomalies.append(max_diff_anomaly)
            # Reset für das nächste Zeitfenster
            current_window = [anomaly]
        last_time = anomaly[1]

    # Den letzten "Fenster"-Sprung auch hinzufügen, falls es noch Anomalien gibt
    if current_window:
        max_diff_anomaly = max(current_window, key=lambda x: x[3])
        filtered_anomalies.append(max_diff_anomaly)

    return filtered_anomalies

def format_time(seconds):
    """
    Formatierte Ausgabe der Zeit in h:mm:ss.x.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = round(seconds % 60, 1)  # Runden auf 1 Nachkommastelle
    return f"{hours:01}:{minutes:02}:{seconds:04.1f}"  # Format mit 1 Nachkommastelle

# Hauptprogramm starten
if __name__ == "__main__":
    try:
        # Argumente parsen und Analyse starten
        args = parse_arguments()
        analyze_audio(args.input_file, args.threshold, args.hop_time, args.threshold_time_gap, args.plot)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)