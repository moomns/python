#!/c/Python34/python
# coding: utf-8

from pandas import DataFrame, Series
import numpy as np
from struct import unpack, pack
import binascii
import matplotlib.pyplot as plt


class DataHeader:

    def __init__(self, f):
        #read file as binary form
        #4byte
        self.nblocks = unpack(">L", f.read(4))[0]
        self.ntraces = unpack(">L", f.read(4))[0]
        self.np = unpack(">L", f.read(4))[0]
        self.ebytes = unpack(">L", f.read(4))[0]
        self.tbytes = unpack(">L", f.read(4))[0]
        self.bbytes = unpack(">L", f.read(4))[0]
        #2byte
        self.vers_id = binascii.hexlify(f.read(2))
        self.status = binascii.hexlify(f.read(2))
        #4byte
        self.nbheaders = unpack(">L", f.read(4))[0]

    def showinfo(self):
        print("<FID Header Information>")
        print("nblocks = {0}\nntraces = {1}\nnp = {2}".format(self.nblocks, self.ntraces, self.np))
        print("type={}".format(type(self.nblocks)))
        print("ebytes = {0}\ntbytes = {1}\nbbytes = {2}".format(self.ebytes, self.tbytes, self.bbytes))
        print("vers_id = {0}\nstatus = {1}\ntype={2}\nnbheaders = {3}\n".format(self.vers_id, self.status, type(self.vers_id), self.nbheaders))

    def __del__(self):
        self.nblocks = 0
        self.ntraces = 0
        self.np = 0
        self.ebytes = 0
        self.tbytes = 0
        self.bbytes = 0
        self.vers_id = 0
        self.status = 0
        self.nbheaders = 0

    def multi_to_single(self):
        self.ntraces = 1
        self.bbytes = self.ntraces * self.tbytes + self.nbheaders * 28

    def write_data_header(self, g):
        g.write(pack(">6I", self.nblocks, self.ntraces, self.np, self.ebytes, self.tbytes, self.bbytes))
        g.write(binascii.unhexlify(self.vers_id))
        g.write(binascii.unhexlify(self.status))
        g.write(pack(">I", self.nbheaders))


class BlockHeader:

    def __init__(self, f):
        self.scale = unpack(">H", f.read(2))[0]
        self.status = unpack(">H", f.read(2))[0]
        self.index = unpack(">H", f.read(2))[0]
        self.mode = unpack(">H", f.read(2))[0]
        self.ctcount = unpack(">L", f.read(4))[0]
        self.lpval = unpack(">f", f.read(4))[0]
        self.rpval = unpack(">f", f.read(4))[0]
        self.lvl = unpack(">f", f.read(4))[0]
        self.tlt = unpack(">f", f.read(4))[0]

    def showinfo(self):
        print("<Block Header Information>")
        print("scale = {0}\nstatus = {1}\nindex = {2}".format(self.scale, self.status, self.index))
        print("mode = {0}\nctcount = {1}\nlpval = {2}".format(self.mode, self.ctcount, self.lpval))
        print("rpval = {0}\nlvl = {1}\ntlt = {2}\n".format(self.rpval, self.lvl, self.tlt))

    def __del__(self):
        self.scale = 0
        self.status = 0
        self.index = 0
        self.mode = 0
        self.ctcount = 0
        self.lpval = 0
        self.rpval = 0
        self.lvl = 0
        self.tlt = 0

    def write_block_header(self, g):
        g.write(pack(">4H", self.scale, self.status, self.index, self.mode))
        g.write(pack(">I", self.ctcount))
        g.write(pack(">4f", self.lpval, self.rpval, self.lvl, self.tlt))

    def multi_to_single(self):
        self.ctcount = 1


def fid_decode(header, f, data, read=1):
    #readはどのトレースを読むのかを指定
    #存在しないトレースを指定していないか確認
    if header.ntraces < read:
        raise ValueError("Trace {} doesn't exist. Please select under ntraces.".format(read))

    #指定トレースより前のトレースを読み捨て
    f.read(header.tbytes * (read - 1))
    for i in range(0, header.np):
        tmp = unpack(">l", f.read(4))[0]
        #np.append(data, tmp)
        data.append(tmp)
        #data = Series([data, tmp])

    #指定したトレースより後のトレースを読み捨て
    f.read(header.tbytes * (header.ntraces - read))
    return data


def fid_to_csv():
    #スピンエコー法のFIDデータの処理のみ対応
    data = r"tachibanafid"
    f = open(data, "rb")
    #データヘッダ取得
    header = DataHeader(f)
    header.showinfo()

    #変換されたバイナリを格納する配列
    sdata = []
    #sdata = np.array([])
    #sdata = Series([])

    #データブロックを読んでバイナリを数値化
    for i in range(0, header.nblocks):
        bheader = BlockHeader(f)
        bheader.showinfo()
        #変換された数値データを一次元配列に保存
        sdata = fid_decode(header, f, sdata)
        del bheader

    np.savetxt("fid-series.csv", sdata, delimiter=",")
    sdata = Series(sdata)
    #reshapeの後にTを入れるともらったプログラムの出力と同じに(列が１つのFIDに)
    ddata = DataFrame((sdata.reshape(256, 512)))
    #フーリエ変換
    fftdata = ddata.apply(np.fft.fft)
    print(fftdata)
    plt.plot(fftdata)
    ddata.to_csv("fid-dataframe.csv", header=None, index=None)
    fftdata.to_csv("postFFT.csv", header=None, index=None)
    fftdata.apply(np.real).to_csv("FFT-real.csv", header=None, index=None)
    fftdata.apply(np.abs).to_csv("FFT-abs.csv", header=None, index=None)

    f.close()


def multi_to_single():
    data = r"tachibanafid"
    write = r"fid-single"
    f = open(data, "rb")
    g = open(write, "bw+")
    #データヘッダ取得
    header = DataHeader(f)
    header.multi_to_single()
    header.write_data_header(g)

    #データブロックを読んでバイナリを数値化
    for i in range(0, header.nblocks):
        #バイナリを格納する配列
        sdata = []
        bheader = BlockHeader(f)
        bheader.multi_to_single()
        bheader.write_block_header(g)
        #変換された数値データを一次元配列に保存
        sdata = fid_decode(header, f, sdata)
        #g.write(pack(">i", sdata))
        
        for j in sdata:
            g.write(pack(">l", j))
        
        del bheader

    f.close()
    g.close()


def multi_to_single_merge():
    spin = r"spinfid"
    data = r"tachibanafid"
    write = r"fid-single-merge"
    f = open(spin, "rb")
    g = open(data, "rb")
    h = open(write, "bw+")
    #シングルスライスのデータヘッダ取得・書き込み
    header_single = DataHeader(f)
    header_single.write_data_header(h)

    #マルチスライスのデータヘッダ取得
    #シングルスライスのデータヘッダを書き込んでいるから，ここでは読み飛ばすだけ
    header = DataHeader(g)

    #データブロックを読んでバイナリを数値化
    for i in range(0, header.nblocks):
        #バイナリを格納する配列
        sdata = []
        trash = []
        #ここではブロックヘッダもシングルスライスのものを流用
        bheader_single = BlockHeader(f)
        bheader_multi = BlockHeader(g)
        bheader_multi.write_block_header(h)
        #変換された数値データを一次元配列に保存
        sdata = fid_decode(header, g, sdata)
        #single-fidと同じだけmulti-fidも読み進める
        trash = fid_decode(header_single, f, trash)
        #g.write(pack(">i", sdata))
        
        for j in sdata:
            h.write(pack(">l", j))
        
        del bheader_single
        del bheader_multi

    f.close()
    g.close()
    h.close()


def main():
    fid_to_csv()
    #multi_to_single()
    #multi_to_single_merge()

if __name__ == '__main__':
    main()
