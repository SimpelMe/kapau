# kapau

Detect anomalies in wav file(s) based on spectral differences and true peak. It detects anomalies that are on one side only.

## Why

We have problems with Pro Tools when bouncing stereo files.
Sometimes the bounce contains silence, bursts (almost fullscale noise) or completely unknown audio (from outside the project).
However, this error only occurs on one side.

It seems to be a combination of Pro Tools, the Trellix virus scanner and our DDP audio file server.
If we completely disable Trellix, this error does not occur.
If we work locally, this error does not occur.

So this tool is to detect those issues.

## How

```shell
usage: kapau.py [OPTIONS] FILES

positional arguments:
  FILES              path to the wav file(s) to analyze
                     provide one stereo or two mono (split) files

options:
  -h, --help         show the help message and exit
  --threshold        spectral difference threshold (dB) (default: 60.0)
  --min_threshold    minimal spectral difference threshold (dB) (default: 20.0)
  --scan_size        size (samples) of analysis window (default: 512)
  --same_error_gap   time (s) ignoring nearby anomalies (default: 5.0)
  --peak_burst       level (dBFS) where burst is detected (default: -4.0)
  --burst_diff       left right peak difference (dB) for burst (default: 6.0)
  --peak_silence     level (dBFS) where silence is detected (default: -80.0)
  -v, --verbose      enable detailed output (default: False)
  -H, --harvester    result report for harvester (default: False)
```

## License

MIT License (c) 2025 Simpel
