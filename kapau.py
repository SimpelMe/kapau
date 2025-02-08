import numpy as np
import os
import librosa
import argparse
import soundfile as sf
import sys
import traceback

# DO NOT TOUCH
# Die folgenden Zeilen sind notwendig für den Harvester
# Pfad zu harvester.app/Contents/MacOS
bundle_path = os.path.dirname(os.path.abspath(__file__))
# Ressourcen-Ordner
lib_path = os.path.join(bundle_path, '../Resources/_internal')
# DYLD_LIBRARY_PATH setzen
os.environ['DYLD_LIBRARY_PATH'] = lib_path + ':' + os.environ.get('DYLD_LIBRARY_PATH', '')

def parse_arguments():
    """Definiert und analysiert die Kommandozeilenargumente."""
    parser = argparse.ArgumentParser(description="Detect anomalies in a stereo WAV file or two mono WAV files based on spectral differences and true peak.")
    parser.add_argument("input_file", nargs="+", help="Path to the WAV file(s) to analyze. Provide one stereo file or two mono files.")
    parser.add_argument("--threshold", type=float, default=60.0, help="Spectral difference threshold (default: 60.0 dB).")
    parser.add_argument("--same_error_gap", type=float, default=5.0, help="Time gap to ignore nearby anomalies (default: 5.0 s).")
    parser.add_argument('--verbose', action='store_true', default=False, help="More detailed output (default: false).")
    parser.add_argument('--harvester', action='store_true', default=False, help="Output pure timestamps for harvester (default: false).")
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
        return 0.0  # Return 0 if the denominator is zero to avoid division by zero
    
    orth_factor = inner_product / square_product
    return round(orth_factor, 2)

def rms_dbfs(signal):
    rms = np.sqrt(np.mean(signal**2))
    return 20 * np.log10(rms) if rms > 0 else -np.inf

def true_peak_dbfs(signal):
    return 20 * np.log10(np.max(np.abs(signal))) if np.max(np.abs(signal)) > 0 else -np.inf

def analyze_audio(input_files, threshold, same_error_gap, verbose, harvester):
    """
    Analysiert die Audiodaten und erkennt Anomalien basierend auf großen spektralen Unterschieden.
    """
    # Lade die Audiodateien
    y, sr = load_audio_files(input_files)
    left, right = y
    # Fenstergröße zur Analyse festlegen in Samples
    hop_length = 512
    S_left = librosa.amplitude_to_db(np.abs(librosa.stft(left, hop_length=hop_length)), ref=1.0)
    S_right = librosa.amplitude_to_db(np.abs(librosa.stft(right, hop_length=hop_length)), ref=1.0)
    spectral_diff = np.abs(S_left - S_right)

    if verbose:
        max_diff = np.max(spectral_diff)

        # Korrelation berechnen
        correlation = [
            calculate_correlation_formel(left[i : i + sr], right[i : i + sr])
            for i in range(0, len(left), sr)
        ]

    # Anomalien erkennen und filtern
    anomalies = []

    for t in range(spectral_diff.shape[1]):
        diff_value = spectral_diff[:, t].max()  # Maximaler Spektraldifferenzwert
        start = t * hop_length
        end = start + hop_length
        if len(left[start:end]) > 0:
            peak_left = true_peak_dbfs(left[start:end])
        else:
            peak_left = -np.inf  # Kein Signal -> niedrigster Wert

        if len(right[start:end]) > 0:
            peak_right = true_peak_dbfs(right[start:end])
        else:
            peak_right = -np.inf  # Kein Signal -> niedrigster Wert

        # Überprüfung, ob eine Anomalie vorliegt wenn:
        # 1. die spektrale Differenz mindestens 20 dB beträgt
        #    → unbedeutende Abweichungen werden grundsätzlich ingnoriert
        # 2. mindestens eine der folgenden Bedingungen erfüllt ist:
        #    a) die spektrale Differenz überschreitet den Schwellwert
        #    b) True Peak für den linken oder rechten Kanal ist sehr groß
        #       → erkennt Bursts
        #    c) True Peak für den linken oder rechten Kanal ist sehr klein
        #       → erkennt Silence
        # 3. beide True Peaks sind nicht zeitgleich sehr hoch oder sehr klein
        #    → vor allem Abblenden werden ignoriert
        if (diff_value >= 20 
            and (diff_value > threshold 
            or peak_left >= -4 
            or peak_right >= -4 
            or peak_left <= -80 
            or peak_right <= -80) 
            and not (peak_left >= -4 and peak_right >= -4) 
            and not (peak_left <= -80 and peak_right <= -80)):
            time_point = librosa.frames_to_time(t, sr=sr, hop_length=hop_length)
            formatted_time = format_time(time_point)

            if verbose:
                channel = "Left" if S_left[:, t].max() > S_right[:, t].max() else "Right"
                corr_value = correlation[int(time_point)] if int(time_point) < len(correlation) else 0
                rms_left = rms_dbfs(left[start:end])
                rms_right = rms_dbfs(right[start:end])

                anomaly_type = ""
                if peak_left < -80:
                    anomaly_type = "Silence"
                    channel = "Left"
                elif peak_right < -80:
                    anomaly_type = "Silence"
                    channel = "Right"
                elif peak_left >= -4:
                    anomaly_type = "Burst"
                    channel = "Left"
                elif peak_right >= -4:
                    anomaly_type = "Burst"
                    channel = "Right"

            if verbose:
                anomalies.append((time_point, formatted_time, diff_value, anomaly_type, channel, corr_value, peak_left, peak_right, rms_left, rms_right))
            else:
                anomalies.append((time_point, formatted_time, diff_value))

    if anomalies:
        anomalies = filter_nearby_anomalies(anomalies, same_error_gap)

    if anomalies:
        if harvester:
            print("Fehler im Audio detektiert bei:")
        else:
            if verbose:
                print(f"Max spectral difference: {max_diff:.2f} dB")
                print(f"{'h:mm:ss':<9}{'Anomaly':<9}{'Ch':<7}{'Diff':<7}{'Corr':<7}{'Peak L':<8}{'Peak R':<8}{'RMS L':<8}{'RMS R'}")

        for anomaly in anomalies:
            if harvester:
                print(f"{anomaly[1]}")
            else:
                if verbose:
                    print(f"{anomaly[1]:<9}{anomaly[3]:<9}{anomaly[4]:<7}{anomaly[2]:<7.2f}{anomaly[5]:<7.2f}{anomaly[6]:<8.2f}{anomaly[7]:<8.2f}{anomaly[8]:<8.2f}{anomaly[9]:.2f}")
                else:
                    print(f"{anomaly[1]}")
        sys.exit(23)
    else:
        if harvester:
            print(0)
        else:
            if verbose:
                print(f"Max spectral difference: {max_diff:.2f} dB")
            print("No significant anomalies detected.")

def filter_nearby_anomalies(anomalies, same_error_gap):
    """
    Identifiziert die größte Spektral-Differenz innerhalb benachbarter Anomalien und gibt diese als Anomalie aus.
    """
    filtered_anomalies = []
    last_time = -float('inf')  # Initialer Vergleichszeitpunkt
    current_window = []  # Liste von Anomalien im aktuellen Zeitfenster
    for anomaly in anomalies:
        if anomaly[0] - last_time <= same_error_gap:
            # Anomalie gehört zum aktuellen Zeitfenster
            current_window.append(anomaly)
        else:
            # Wenn das Zeitfenster überschritten wird, die größte Differenz im aktuellen Fenster ermitteln
            if current_window:
                # Anomalie mit dem größten Differenzwert
                max_diff_anomaly = max(current_window, key=lambda x: x[2])
                filtered_anomalies.append(max_diff_anomaly)
            # Reset für das nächste Zeitfenster
            current_window = [anomaly]
        last_time = anomaly[0]

    # Die letzte Differenz hinzufügen, falls es noch Anomalien gibt
    if current_window:
        max_diff_anomaly = max(current_window, key=lambda x: x[2])
        filtered_anomalies.append(max_diff_anomaly)

    return filtered_anomalies

def format_time(seconds):
    """
    Formatierte Ausgabe der Zeit in h:mm:ss
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:01}:{minutes:02}:{seconds:02}"

# Hauptprogramm starten
if __name__ == "__main__":
    try:
        # Argumente parsen und Analyse starten
        args = parse_arguments()
        if args.verbose and not args.harvester:
            pathwofilename, filename = os.path.split(args.input_file[0])
            print(f"{filename}")
        analyze_audio(args.input_file, args.threshold, args.same_error_gap, args.verbose, args.harvester)
        sys.exit(0)
    except Exception as e:
        # Ausgabe Fehler-Stacktrace inklusive der fehlerhaften Zeile
        print(traceback.format_exc())
        sys.exit(1)
