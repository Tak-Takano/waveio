import numpy as np
import pandas as pd
import cmath
from pathlib import Path, PosixPath
from box import Box
from dataclasses import dataclass, field, asdict

import warnings

from typing import NewType, Union, Optional, Any

# __version__ = "5.0.0"

### Utils
    
get_amp = np.frompyfunc(abs, 1, 1)
get_phase = np.frompyfunc(cmath.phase, 1, 1)
get_real = np.frompyfunc(lambda x:x.real, 1, 1)
get_imag = np.frompyfunc(lambda x:x.imag, 1, 1)

### binary file operations

def to_int(file, nbytes, byteorder="little", signed=True):
    return int.from_bytes(file.read(nbytes), byteorder=byteorder, signed=signed)


def to_str(file, nbytes):
    return file.read(nbytes).decode('utf-8', errors="replace")


def from_int(val, nbytes, byteorder="little", signed=True):
    try:
        val = int(val)
        return int.to_bytes(val, length=nbytes, byteorder=byteorder, signed=signed)
    except Exception as e:
        print(e)
        return None


def from_str(val):
    val = str(val)
    return val.encode('utf-8')


# misc

def is_WavStructure(wav_in: Any) -> bool:
    return type(wav_in) == WavStructure


def is_dict_or_box(in_dict:Union[dict, Box]) -> bool:
    return (type(in_dict) is dict) or (type(in_dict) is Box)

def is_wavdict(wavdict: dict, only_header=False) -> bool:
    if not is_dict_or_box(wavdict):
        return False

    if not 'fmt ' in wavdict.keys():
        return False

    if not is_dict_or_box(wavdict['fmt ']):
        return False

    if not 'sample_rate' in wavdict['fmt '].keys():
        return False

    if not 'channels' in wavdict['fmt '].keys():
        return False

    if not 'block_size' in wavdict['fmt '].keys():
        return False
    
    if not only_header:
        if not 'data' in wavdict.keys():
            return False

        if not is_dict_or_box(wavdict.data):
            return False
        
        if not 'data_chunk_pos_in_byte' in wavdict.data.keys():
            return False
    return True


### WavStructure Class
@dataclass
class WavStructure:
    """wave形式のmeta, data, method一括管理クラス

    .. TODO: Google docstringsのクラスの書き方にしたがって追記する

    """
    @dataclass
    class Parameters:
        """メタデータを保管するクラス

        .. TODO: Google docstringsのクラスの書き方にしたがって追記する
        """
        channels: int = field(default=1)
        sample_rate: float = field(default=11025.)
        data_length: int = field(default=0)
        duration: float = field(default=0) #そのうちtimedelta
        
        def to_dict(self):
            return asdict(self)

        def from_dict(self, wavdict):
            if not is_wavdict(wavdict):
                raise ValueError('Invalid format')
            fmt = wavdict['fmt ']
            self.channels = fmt.channels
            self.sample_rate = fmt.sample_rate
            self.data_length = wavdict.data.size // fmt.block_size # だったかな？
            try:
                self.duration = self.data_length / self.sample_rate
            except ZeroDivisionError as e:
                warnings.warn(f'File\'s Sample rate is {self.sample_rate}')
                self.duation = 0 # sample_rateが欠損したファイルがあるため

    
    params: Parameters = field(default_factory=Parameters)
    data: Box = field(default_factory=Box)
        
    def to_pandas(self) -> pd.core.frame.DataFrame:
        """convert .data to pandas

        Returns:
            pd.core.frame.DataFrame: dict型であるデータをデータフレームに変換する．
        """
        return pd.DataFrame(self.data)

    def to_header(self, bytes_per_sample: int, is_special:bool=False) -> bytes:
        """ヘッダーをbytesに変換する

        Case
        =======
        
        blocksize: 1 & is_special: False => 8 bit Linear PCM (unsupported)
        blocksize: 1 & is_special: True => μ-law PCM
        blocksize: 2 & is_special: False => 16 bit Linear PCM
        blocksize: 3 & is_special: False => 24 bit Linear PCM
        blocksize: 4 & is_special: False => 32 bit-integer Linear PCM
        blocksize: 4 & is_special: True => 32 bit-float Linear PCM

        Args:
            bytes_per_sample (int): bytes per sample
            is_special (bool, optional): required for mu-law or 32-float. Defaults to False.

        Returns:
            bytes: header in bytes
        """
        ### 必要な情報
        channels = self.params.channels
        sample_rate = int(self.params.sample_rate)
        block_size = channels * bytes_per_sample
        bits_per_sample = 8 * bytes_per_sample
        bytes_per_sec = block_size * sample_rate
        data_chunk_size = self.params.data_length * block_size


        if bytes_per_sample == 1:
            raise ValueError('Specified format is not supported')
            if is_special:
                format_tag = 7
                fmt_size = 16
            else:
                format_tag = 1
                fmt_size = 16


        elif bytes_per_sample in [2, 3]:
            if channels > 2:
                raise ValueError('Specified format is not supported')
                # 4ch は format_tag=-2, fmt_size=40 ?
            format_tag = 1
            fmt_size = 16

        elif bytes_per_sample == 4:
            raise ValueError('Specified format is not supported')
            if is_special:
                format_tag = 3
                fmt_size = 18

            else:
                format_tag = -2
                fmt_size = 40
        else:
            format_tag = -1
            fmt_size = 16

            
        riff_size = data_chunk_size + fmt_size + 20 # WAVE + fmt  + fmt _size + data + data_size で 20 byte

        bytestream = b''
        bytestream += from_str('RIFF')
        bytestream += from_int(riff_size, 4)
        bytestream += from_str('WAVE')
        bytestream += from_str('fmt ')
        bytestream += from_int(fmt_size, 4)
        bytestream += from_int(format_tag, 2)
        bytestream += from_int(channels, 2)
        bytestream += from_int(sample_rate, 4)
        bytestream += from_int(bytes_per_sec, 4)
        bytestream += from_int(block_size, 2)
        bytestream += from_int(bits_per_sample, 2)
        bytestream += from_str('data')
        bytestream += from_int(data_chunk_size, 4)
        
        return bytestream

    def to_datachunk(self, bytes_per_sample: int)-> bytes:
        """データをバイト列に変換

        Args:
            bytes_per_sample (int): bytes_per_sample

        Returns:
            bytes: so-called raw data
        """

        df = self.to_pandas()
        array = df.drop(columns='time', errors='ignore').values.flatten()
        array = 2**(8*bytes_per_sample - 1)*array
        array = array.astype(int)
        u_from_int = np.frompyfunc(lambda x:from_int(x, bytes_per_sample), 1, 1)
        bin_array = u_from_int(array)
        return b''.join(bin_array)

    def cut_sec(self, st, ed):
        return cut_sec(self, st, ed)

    def cut(self, st, ed):
        return cut(self, st, ed)

    def write(self, fname, bytes_per_sample=2):
        write(self, fname, bytes_per_sample=bytes_per_sample)


### I/O

## Private
from io import BufferedReader


def _switch(f: BufferedReader, pos: int, wavdict: dict) -> tuple[BufferedReader, int, dict]:
    # f.seek(pos)
    cid = to_str(f, 4)
    f.seek(pos)

    if cid == "fmt ":
        _fmt_routine(f, pos, wavdict)

    elif cid == "fact":
        _fact_routine(f, pos, wavdict)

    elif cid == "data":
        _data_routine(f, pos, wavdict)

    elif cid == "JUNK":
        _misc_routine(f, pos, wavdict)

    elif cid == "FLLR":
        _misc_routine(f, pos, wavdict)

    elif cid == "PEAK":
        _misc_routine(f, pos, wavdict)

    else:
        pass
    return f, pos, wavdict


def _fmt_routine(f: BufferedReader, pos: int, wavdict: dict) -> tuple[BufferedReader, int, dict]:
    # 共通処理
    cid = to_str(f, 4)
    pos += 4
    wavdict[cid] = Box({})
    size = to_int(f, 4)
    wavdict[cid].size = size
    pos += 4
    size0 = pos
    
    # 独自処理
    wavdict[cid].format_tag = to_int(f, 2)
    pos += 2
    wavdict[cid].channels = to_int(f, 2)
    pos += 2
    wavdict[cid].sample_rate = to_int(f, 4)
    pos += 4
    wavdict[cid].bytes_per_sec = to_int(f, 4)
    pos += 4
    wavdict[cid].block_size = to_int(f, 2)
    pos += 2
    wavdict[cid].bits_per_sample = to_int(f, 2)
    pos += 2
    
    # 共通処理
    if pos < size + size0:
        f.seek(size + size0)
        pos = size + size0
    return _switch(f, pos, wavdict)


def _fact_routine(f: BufferedReader, pos: int, wavdict: dict) -> tuple[BufferedReader, int, dict]:
    # 共通処理
    cid = to_str(f, 4)
    pos += 4
    wavdict[cid] = Box({})
    size = to_int(f, 4)
    wavdict[cid].size = size
    pos += 4
    size0 = pos
    
    # 独自処理
    wavdict[cid].length_per_channel = to_int(f, 4)
    pos += 4
    
    # 共通処理
    if pos < size + size0:
        f.seek(size + size0)
        pos = size + size0
    return _switch(f, pos, wavdict)


def _data_routine(f: BufferedReader, pos: int, wavdict: dict) -> tuple[BufferedReader, int, dict]:
    # 共通処理
    cid = to_str(f, 4)
    pos += 4
    wavdict[cid] = Box({})
    size = to_int(f, 4)
    wavdict[cid].size = size
    pos += 4
    wavdict[cid].data_chunk_pos_in_byte = pos
    size0 = pos

    
    # 共通処理
    if pos < size + size0:
        f.seek(size + size0)
        pos = size + size0
    return _switch(f, pos, wavdict)


def _misc_routine(f: BufferedReader, pos: int, wavdict: dict) -> tuple[BufferedReader, int, dict]:
    # 共通処理
    cid = to_str(f, 4)
    pos += 4
    wavdict[cid] = Box({})
    size = to_int(f, 4)
    wavdict[cid].size = size
    pos += 4
    size0 = pos
    
    # 共通処理
    if pos < size + size0:
        f.seek(size + size0)
        pos = size + size0
    return _switch(f, pos, wavdict)


def _read_header(file: Union[str, PosixPath]) -> dict:

    file = Path(file)
    if not file.exists():
        raise FileNotFoundError(f'{file} not found')
    
    wavdict = Box({})
    with open(file, "rb") as f:
        pos = 0
        wavdict.riff_id = to_str(f, 4)
        pos += 4
        wavdict.size = to_int(f, 4)
        pos += 4
        wavdict.format = to_str(f, 4)
        pos += 4
        *_, wavdict = _switch(f, pos, wavdict)
    return wavdict

def _chunk_framer(chunk: bytes, nbyte: int):
    n_chunk = len(chunk)
    for i in range(n_chunk//nbyte):
        yield chunk[i * nbyte : (i + 1) * nbyte]


def _make_data_dict(chunk: bytes, nbyte: int, n_ch: int) -> dict:
    data_array = np.array([int.from_bytes(c, byteorder='little', signed=True) for c in _chunk_framer(chunk, nbyte)])
    # n byte で正規化
    gain = 1 / 2**(8*nbyte - 1)
    data_array = gain * data_array
    
    n_data = len(data_array)
    n_col = n_data // n_ch

    # 端数が発生した時は切り捨てる
    data_array = data_array[: n_data // n_col * n_col].reshape((n_col, -1))
    ret = Box({})
    for i in range(n_ch):
        ret[f'ch{i+1}'] = data_array[:, i]
    return ret


def _read_data(file: Union[str, PosixPath], pos: int, read_size: int, block_size: int, channels: int, sample_rate: float, *args, **kwargs) -> dict:
    with open(file, "rb") as f:
        f.seek(pos)
        chunk = f.read(read_size)
    # blocksize は bits per sample * channels
    ret = _make_data_dict(chunk, block_size // channels, channels)
    try:
        delta_t = 1 / sample_rate
    except ZeroDivisionError as e:
        warnings.warn(f'File\'s Sample rate is {sample_rate}')
        delta_t = 0
    ret['time'] = np.array([delta_t * i for i in range(len(ret['ch1']))])
    return ret


## Public

# read
def read(file: Union[str, PosixPath]) -> WavStructure:
    """ファイル全体を読み込んでWavStructureを返す

    Args:
        file (Union[str, PosixPath]): ファイル名

    Returns:
        WavStructure: 結果
    """
    file = str(file) # pathlib対応
    return read_core(file)


def read_params(file: Union[str, PosixPath]) -> WavStructure:
    """ヘッダーだけを読み込んでWavStructureを返す

    Args:
        file (Union[str, PosixPath]): ファイル名

    Returns:
        WavStructure: 結果
    """
    file = str(file) # pathlib対応
    return read_core(file, read_data=False)


def read_partial_sec(file: Union[str, PosixPath], start: float, end: float) -> WavStructure:
    """部分的に読み込んでWavStructureを返す

    Args:
        file (Union[str, PosixPath]): ファイル名
        start (float): 開始位置（秒）
        end (float): 終了位置（秒）

    Returns:
        WavStructure: 結果
    """
    file = str(file) # pathlib対応
    tmp_wav = read_params(file)
    sample_rate = tmp_wav.params.sample_rate
    n_st = int(start * sample_rate)
    n_ed = int(end * sample_rate)
    return read_core(file, n_st, n_ed)


def read_core(file: Union[str, PosixPath],
                 n_st: int=None,
                 n_ed: int=None,
                 read_data: bool=True) -> WavStructure:
    """読み込みの共通処理: 開始，終了位置を指定してWavStructreを返す

    Args:
        file (Union[str, PosixPath]): ファイル名
        n_st (int, optional): 開始位置（サンプル）. Defaults to None.
        n_ed (int, optional): 開始位置（サンプル）. Defaults to None.
        read_data (bool, optional): データを読み込むかどうか. Defaults to True.

    Raises:
        FileNotFoundError: ファイルが存在しないときに発生

    Returns:
        WavStructure: 結果
    """

    # ファイルの存在チェック
    if not Path(file).exists():
        raise FileNotFoundError(f'{file} not found')

    file = str(file) # pathlib対応

    wavdict = _read_header(file)

    # WavStructureへデータ渡し
    ret = WavStructure()

    # paramsの読み込み
    ret.params.from_dict(wavdict)

    if read_data:

        if (n_st is None) and (n_ed is None):
            pos_in_byte = wavdict.data.data_chunk_pos_in_byte
            read_size = wavdict.data.size
        else:
            if n_st > n_ed:
                n_st, n_ed = n_ed, n_st
            pos_in_byte = wavdict['fmt '].block_size * n_st + wavdict.data.data_chunk_pos_in_byte
            read_size = wavdict['fmt '].block_size * (n_ed - n_st)

        wavdata = _read_data(file, pos_in_byte, read_size, **wavdict['fmt '])
        ret.data = wavdata

    else:
        ret.data = Box({'ch1': np.array([]), 'time': np.array([])})

    # validation due to 山ちゃんwave
    if ret.params.data_length != len(ret.data.ch1):
        ret.params.data_length = len(ret.data.ch1)
        ret.params.duration = ret.params.data_length / ret.params.sample_rate
    
    return ret

# convert

def cut_sec(wav: WavStructure, start: float, end: float) -> WavStructure:
    """WavStructureを部分的に切り出す

    Args:
        wav (WavStructure): WavStructureオブジェクト
        start (float): 開始位置（秒）
        end (float): 終了位置（秒）

    Returns:
        WavStructure: 結果
    """

    sample_rate = wav.params.sample_rate
    n_st = int(start * sample_rate)
    n_ed = int(end * sample_rate)
    return cut(wav, n_st, n_ed)


import copy
def cut(wav_in: WavStructure, n_st: int, n_ed: int) -> WavStructure:
    """WavStructureを部分的に切り出す

    Args:
        wav_in (WavStructure): WavStructureオブジェクト
        n_st (float): 開始位置（秒）
        n_ed (float): 終了位置（秒）

    Raises:
        ValueError: 入力がWavStructureオブジェクトではないときに発生

    Returns:
        WavStructure: 結果
    """
    if not is_WavStructure(wav_in):
        raise ValueError('Invalid Input')

    if n_st > n_ed: n_st, n_ed = n_ed, n_st
    wav_out = WavStructure()
    wav_out.params = copy.deepcopy(wav_in.params)
    for k, v in wav_in.data.items():
        wav_out.data[k] = wav_in.data[k][n_st:n_ed]
        
    wav_out.params.data_length = len(wav_out.data[k])
    wav_out.params.duration = wav_out.params.data_length / wav_out.params.sample_rate
    return wav_out

# write

def write(wav: WavStructure,
            file: Union[str, PosixPath],
            bytes_per_sample: int=2):
    file = str(file) # pathlib対応
    write_core(wav, file, bytes_per_sample=bytes_per_sample)


def write_partial_sec(wav: WavStructure,
                        file: Union[str, PosixPath],
                        start:float,
                        end:float,
                        bytes_per_sample: int=2):
    file = str(file) # pathlib対応
    sample_rate = wav.params.sample_rate
    n_st = int(start * sample_rate)
    n_ed = int(end * sample_rate)
    write_core(wav, file, n_st, n_ed, bytes_per_sample=bytes_per_sample)


def write_core(wav: Optional[WavStructure],
                  file: Union[str, PosixPath],
                  n_st: int=None,
                  n_ed: int=None,
                  bytes_per_sample: int=2):
    if not is_WavStructure(wav): 
        raise ValueError('Invalid Input')
    file = str(file) # pathlib対応

    if (n_st is not None) & (n_ed is not None):
        if n_st > n_ed: n_st, n_ed = n_ed, n_st
        wav = cut(wav, n_st, n_ed)
        
    with open(file, "wb") as f:
        f.write(wav.to_header(bytes_per_sample))
        f.write(wav.to_datachunk(bytes_per_sample))
