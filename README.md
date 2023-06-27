# Wave I/O
自身の研究で使用するための、自作のwavファイル読み込みツール。

# Installation
1. git clone する
2. `pip install (/path/to/repository)`

# Usage
## Import
```python
import waveio as wio
```

## ファイル読み込み
```python
wav = wio.read("sample.wav")
```

## ファイル書き出し
```python
wav.write("sample.wav")
```

## 一部切り出し
```python
wav2 = wav.cut_sec(5, 10) # 5秒から10秒を切り出し
```

# License

"waveio" is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).

