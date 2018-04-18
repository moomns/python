# -*- coding: utf-8 -*-

from logging import getLogger
import logging, logging.handlers

# ルートロガーの設定
#rootロガーのログレベルは、ハンドラーの中で一番低いものを指定しておく
#ここでの設定(上位ロガーの設定)が各ライブラリ等の子ハンドラに伝搬する
#rootロガーの取得時にはgetLoggerに引数を渡さない
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# ファイルハンドラ
# ログのファイル出力先を設定
start = datetime.datetime.today()
now = "{}{:02d}{:02d}{:02d}{:02d}{:02d}".format(start.year%2000,start.month, start.day, start.hour, start.minute, start.second)
fh = logging.FileHandler("GridSearch"+now+".log")
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

# ストリームハンドラ
# ログのコンソール出力の設定
sh = logging.StreamHandler()
sh.setLevel(logging.INFO)
logger.addHandler(sh)

# ログの出力形式の設定
# ex)2017-11-01 14:19:18,813:L136   [INFO   ]   [Jupyter.<module>] 
formatter = logging.Formatter('%(asctime)s:L%(lineno)d \t[%(levelname)-7s] \t[%(name)s.%(funcName)s] %(message)s')
fh.setFormatter(formatter)
sh.setFormatter(formatter)

# メールハンドラ
# メール出力の設定
MAIL = 60 #must be larger > logging.CRITICAL
logging.addLevelName(MAIL, "MAIL") #jlogger.log("MAIL", msg)
mailhost =  ('smtp.hines.hokudai.ac.jp', 587)
fromaddr = "FROM"
toaddrs = ["TO"]
subject = "MAIL SUBJECT"
credentials = ("Active Mail ID", "Active Mail Password")
secure = () #smtplib.SMTP.starttls()への引数
mh = logging.handlers.SMTPHandler(mailhost, fromaddr, toaddrs, subject, credentials, secure)
mh.setLevel(logging.CRITICAL) #メール出力条件はCRITICAL(プログラム停止) or 自作レベルMAIL
logger.addHandler(mh)
mh.setFormatter(formatter)

# スクリプト内で用いるロガー
# なるべくルートロガーは汚染しない
#jupyterでの実行内容を記録するロガー
jlogger = logging.getLogger("Jupyter")
jlogger.setLevel(logging.DEBUG)

# ライブラリ用のロガー
# ライブラリを作る際には、ロガーにNullHandlerを登録しておく
#from logging import getLogger, DEBUG, NullHandler
#logger = getLogger(__name__)
#logger.addHandler(NullHandler())
#logger.setLevel(DEBUG)