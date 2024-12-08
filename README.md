# Wisteria

`※ work ディレクトリであることを確認！`

## Usage

- バッチジョブスクリプトの作成

```sh
#!/bin/sh

#PJM -L rscgrp=debug-a
#PJM -L node=1
#PJM -L elapse=0:15:00
#PJM -g group1
#PJM -j

module load cuda/11.1
module load pytorch/1.8.1

source $PYTORCH_DIR/bin/activate
python3 train.py
```

- ジョブを投入する

```bash
$ pjsub run.sh  # ジョブを投入
[INFO] PJM 0000 pjsub Job 12345 submitted.
```

| オプション | 内容         |
| ---------- | ------------ |
| `pjsub`    | ジョブの投入 |
| `pjdel`    | ジョブの削除 |
| `pjstat`   | ジョブの状態 |

| 項目       | 内容                                                                        |
| ---------- | --------------------------------------------------------------------------- |
| JOB_ID     | ジョブの一意の識別子<br>ジョブ状態の確認や削除時に使用                      |
| JOB_NAME   | 投入されたジョブスクリプトの名前                                            |
| STATUS     | 現在のジョブの状態<br>`RUNNING`（実行中）、`PENDING`（待機中）など          |
| PROJECT    | ジョブが紐づいているプロジェクト名                                          |
| RSCGROUP   | 使用しているリソースグループ<br>例: `share-debug`（デバッグ用共有リソース） |
| START_DATE | ジョブが開始された日時                                                      |
| ELAPSE     | ジョブの経過時間（開始から現在まで）                                        |
| TOKEN      | 割り当てられた計算リソースの一部を示す値                                    |
| NODE       | 使用している計算ノードの詳細                                                |
| GPU        | ジョブで使用している GPU の数                                               |

## module

- module load の設定はセルセッションに限定されるため、ssh 接続が切れると再設定が必要になる。
  | サブコマンド | 説明 |
  |-------------------|----------------------------------------------------------------------|
  | `list` | 設定中のアプリケーション、ライブラリの一覧を表示 |
  | `avail` | 設定可能なアプリケーション、ライブラリの一覧を表示 |
  | `load <module>` | アプリケーション、ライブラリ向けの環境変数を設定 |
  | `unload <module>` | アプリケーション、ライブラリ向けの環境変数を解除 |
  | `switch <moduleA> <moduleB>` | アプリケーション、ライブラリ向けの環境変数を moduleA から moduleB へ入れ替える |
  | `purge` | 設定中のすべてのアプリケーション、ライブラリを解除 |
  | `help <module>` | アプリケーション、ライブラリの使用方法を表示<br>また、構築に使用したコンパイラのバージョンを表示します。 |

```bash
$ module list
$ module avail
```

## ディスク使用状況と上限の表示

```bash
$ show_quota
```

| 項目                    | 説明                                 |
| ----------------------- | ------------------------------------ |
| Directory               | 割り当てディレクトリ                 |
| /home                   | ログインノード用共有ファイルシステム |
| /work/groupname         | 共有ファイルシステム                 |
| /data/scratch/groupname | 高速ファイルシステム(一時領域)       |
| /data/perm/groupname    | 高速ファイルシステム(恒久領域)       |
| used(MB)                | ディスク使用量                       |
| limit(MB)               | ディスク使用上限                     |
| nfiles                  | ファイル数                           |
| nlimit                  | ファイル数上限                       |

## venv

- これはうまくいかない

```bash
$ module load python/3.10.13
$ python -mvenv venv
$ source venv/bin/activate
$ python -V
> Python 3.6.8
```

- これはうまくいく

```bash
$ module load python/3.10.13
$ python3 -mvenv venv
$ source venv/bin/activate
$ python -V
> Python 3.10.13
```

## Reference

[スーパーコンピュータでサンプルプログラムを動かす（PyTorch）](https://aobatogou.com/supercomputer-pytorch)
