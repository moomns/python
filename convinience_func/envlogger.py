# -*- coding: utf-8 -*-

import sys
import pprint
import subprocess

from logging import getLogger, DEBUG, NullHandler
logger = getLogger(__name__)
logger.addHandler(NullHandler())
logger.setLevel(DEBUG)

import wmi

def show_machine_info(hide=False):
    """
    使用しているWindows機のマシンスペックを表示し、ログに記録する

    Keyword Argument

    hide:   printを用いて標準出力に出力するか否か
    """

    computer = wmi.WMI()
    computer_info = computer.Win32_ComputerSystem()[0]
    os_info = computer.Win32_OperatingSystem()[0]
    proc_info = computer.Win32_Processor()[0]
    gpu_info = computer.Win32_VideoController()[0]

    os_name = os_info.Name.split('|')[0]
    os_version = ' '.join([os_info.Version, os_info.BuildNumber])
    system_ram = float(os_info.TotalVisibleMemorySize) / 1048576  # KB to GB

    if hide:
        pass

    else:
        print('OS Name: {0}'.format(os_name))
        print('OS Version: {0}'.format(os_version))
        print('Machine Manufacturer: {0}'.format(computer_info.Manufacturer))
        print('Machine Model: {0}'.format(computer_info.Model))
        print('Machine Name: {0}\n'.format(computer_info.DNSHostName))

        print('CPU: {0}'.format(proc_info.Name))
        print('RAM: {0:.2f} GB'.format(system_ram))
        print('Graphics Card: {0}'.format(gpu_info.Name))

    logger.debug('OS Name: {0}'.format(os_name))
    logger.debug('OS Version: {0}'.format(os_version))
    logger.debug('Machine Manufacturer: {0}'.format(computer_info.Manufacturer))
    logger.debug('Machine Model: {0}'.format(computer_info.Model))
    logger.debug('Machine Name: {0}\n'.format(computer_info.DNSHostName))

    logger.debug('CPU: {0}'.format(proc_info.Name))
    logger.debug('RAM: {0:.2f} GB'.format(system_ram))
    logger.debug('Graphics Card: {0}'.format(gpu_info.Name))


def show_python_env(hide=False):
    """
    使用しているpython環境の詳細を表示する

    Keyword argument
    hide:   printを用いて標準出力に出力するか否か
    """

    packages = subprocess.check_output("conda list").decode("cp932").split("\n")

    if hide:
        pass

    else:
        pprint.pprint(packages)

    for package in packages:
        logger.debug(package)


def record_exec_env():
    """
    スクリプト実行環境をロガーへ記録するためのラッパー関数
    """
    show_machine_info(hide=True)
    show_python_env(hide=True)