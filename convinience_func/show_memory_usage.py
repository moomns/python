# -*- coding: utf-8 -*-

import sys
import os

from numbers import Number
from collections import Set, Mapping, deque
import inspect

from logging import getLogger, DEBUG, NullHandler
import psutil
logger = getLogger(__name__)
logger.addHandler(NullHandler())
logger.setLevel(DEBUG)


def get_size(obj, seen=None):
    """Recursively finds size of objects in bytes
    https://github.com/bosswissam/pysize/blob/master/pysize.py
    単体では動くが、locals().keys()を渡して回すとエラー
    """
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if hasattr(obj, '__dict__'):
        for cls in obj.__class__.__mro__:
            if '__dict__' in cls.__dict__:
                d = cls.__dict__['__dict__']
                if inspect.isgetsetdescriptor(d) or inspect.ismemberdescriptor(d):
                    size += get_size(obj.__dict__, seen)
                break
    #以下でtype型を検出したときにTypeError, その他にUnsupportedOperationがRaiseされること有り
    if isinstance(obj, dict):
        size += sum((get_size(v, seen) for v in obj.values()))
        size += sum((get_size(k, seen) for k in obj.keys()))
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum((get_size(i, seen) for i in obj))
    return size


def getsize(obj_0):
    """
    Recursively iterate to sum size of object & members.
    https://stackoverflow.com/questions/449560/how-do-i-determine-the-size-of-an-object-in-python
    これもlocals().keysをで回すとTypeErrorで事故り気味
    現在のshow_memory_usageではこれを利用
    to do 他の関数での値と比較
    """

    zero_depth_bases = (str, bytes, Number, range, bytearray)
    iteritems = 'items'
    def inner(obj, _seen_ids = set()):
        obj_id = id(obj)
        if obj_id in _seen_ids:
            return 0
        _seen_ids.add(obj_id)
        size = sys.getsizeof(obj)
        if isinstance(obj, zero_depth_bases):
            pass # bypass remaining control flow and return
        elif isinstance(obj, (tuple, list, Set, deque)):
            size += sum(inner(i) for i in obj)
        elif isinstance(obj, Mapping) or hasattr(obj, iteritems):
            size += sum(inner(k) + inner(v) for k, v in getattr(obj, iteritems)())
        # Check for custom object instances - may subclass above too
        if hasattr(obj, '__dict__'):
            size += inner(vars(obj))
        if hasattr(obj, '__slots__'): # can have __slots__ with __dict__
            size += sum(inner(getattr(obj, s)) for s in obj.__slots__ if hasattr(obj, s))
        return size
    return inner(obj_0)


def show_memory_usage(var_keys, global_dict, count_size=get_size, usage_in_bytes=10000, hide=False):
    """
    指定したバイト数以上のメモリ(ex.10000)を使用している変数を表示
    
    Argument
    var_keys:           検査したい変数名のリスト
                        現在のスコープについて表示する->locals().keys()
                        グローバルについて調べる->globals().keys()
                        =>この方法だと、どのgetsizeof系を使ってもTypeError/Exceptionに遭遇

                        ひとまずユーザ定義変数のみを対象に解析したいなら
                        起動時にstartvals = set(dir())
                        var_keys->set(dir())-startvals
                        とすれば、例外に遭遇せず所望の動作をする

    global_dict:        変数の辞書 globals()/locals()

    Keyword argument
    count_size:         オブジェクトの容量を正しく計算するための関数 default get_size()
    usage_in_bytes:     表示したい専有メモリの下限バイト数
    hide:               print文を用いた表示を行うか否か

    Usage
    スクリプト初めに
    startvals=set(dir())
    その後
    show_memory_usage(set(dir())-startvals, globals())
    show_memory_usage(set(dir())-startvals, globals(), usage_in_bytes=1024*50)

    ×show_memory_usage(locals.keys(), globals())
    ×show_memory_usage(locals.keys(), globals(), usage_in_bytes=1024*50)

    """

    logger.debug("{}{: <50}{}{: >10}{}".format('|','Variable Name','|','Memory[KB]','|'))
    logger.debug(" ------------------------------------------------------------- ")
    for var in var_keys:
        #sys.getsizeofではメンバを有するオブジェクトのバイト数を正しく取れない
        if var.startswith("_") == 0 and count_size(eval(var, global_dict)) > usage_in_bytes:
            logger.debug("{}{: <50}{}{: >10}{}".format(
                '|',var,'|',count_size(eval(var, global_dict))//1024,'|'))
    logger.debug("\n")

    if hide:
        pass

    else:
        print("{}{: <50}{}{: >10}{}".format('|','Variable Name','|','Memory[KB]','|'))
        print(" ------------------------------------------------------------- ")
        for var in var_keys:
            if var.startswith("_") == 0 and count_size(eval(var, global_dict)) > usage_in_bytes:
                print("{}{: <50}{}{: >10}{}".format(
                    '|',var,'|',count_size(eval(var, global_dict))//1024,'|'))
        print("")


def show_system_resource(hide=False):
    """
    現在のシステムリソースを表示
    """

    if hide:
        pass

    else:
        print("System info")
        print("CPU info\n", psutil.cpu_times())
        print("Memory info\n", psutil.virtual_memory(), "\n")


    logger.debug("\nSystem info\nCPU info\n{}\nMemory info\n{}\n".format(
        psutil.cpu_times(), psutil.virtual_memory()))


def get_now_python_process():
    """
    現在のpythonプロセスを返す
    """

    logger.debug("now process id: {}\n".format(os.getpid()))
    return psutil.Process(os.getpid())


def show_process_info(p, hide=False):
    """
    指定されたプロセスオブジェクトに関し、そのプロセスのCPU/メモリ利用率を表示する
    """
    if hide:
        pass

    else:
        print("Process info")
        print("CPU[%]:", p.cpu_percent())
        print("Memory[%]:", p.memory_percent(), "\n")

    logger.debug("Process info\nCPU[%]: {}\nMemory[%]: {}\n".format(p.cpu_percent(), p.memory_percent()))