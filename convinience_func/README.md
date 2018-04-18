# Convinience_func
Pythonにおける便利な小モジュール群をまとめた  

## 各ファイルの概要
### 重要モジュール
* envlogger.py  
pythonや使用機器の実行環境をロギングするための関数を実装したモジュール  
* GraphtecAnalysis.py  
熱電対のロガーから吐き出されたcsvデータを処理し、画像化するための関数を実装したモジュール  
* oscilloscope_data_read.py  
オシロスコープMemoryPrime GDS-1000 SeriesのSave Allで保存されたデータを読み込むための関数を実装したモジュール
* send_mail.py  
メールアラートを投げるための関数を実装したモジュール
* show_memory_usage.py  
各変数のメモリ使用量を把握するための関数を実装したモジュール
* time_series_class.py  
時系列データを読み込んで適切に周波数解析するクラスを実装したモジュール

### モジュール
* accurate_sleep.py  
正しく時間を計時できる(らしい)関数をまとめたモジュール  
* automation.py  
指定フォルダ以下に一括処理を行うための関数を実装したモジュール  
* baseline_als.py  
ベースラインを補正するための関数を実装したモジュール  
* downsampling.py  
エイリアシングを防いでダウンサンプリングする関数を実装したモジュール  
* ICA.py  
独立成分分析のための関数を実装したモジュール  
* show_object_member.py  
オブジェクトが有するメンバを表示するための関数を実装したモジュール

### スクリプト
* logger_config.py  
プログラムに記述するロガーの一般的記法をまとめたスクリプト  
* notch_filter.py  
ノッチフィルタ処理をまとめたスクリプト  
* schedule_scrayping.py  
サイボウズからスクレイピングしてスケジュールを抽出するスクリプト

