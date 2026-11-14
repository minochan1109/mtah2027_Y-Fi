# 提出用：MTAH 2027 Phase 1 ボール保持推定（v3）

**提出ファイルを作るのに必要なものだけ**を集めたフォルダです。ここだけで提出ファイルが作れます。
親フォルダ（`mtah_possession/`）にある v1・v2・実験用ツールは使いません。

## 1. フォルダの置き方

次のように、このリポジトリと同じ階層に公式リポジトリとサンプル試合を置きます。

```
どこか/
├── more-than-a-hack-2027/      公式リポジトリ。配布データは data/ に入れる（公式の手順どおり）
├── sample-data/                Metrica の公開サンプル試合（学習用。下のコマンドで取得）
└── mtah2027_Y-Fi/              このリポジトリ（clone したときの名前は何でもよい）
    └── submission/             ← ここ
```

別の場所に置いている場合は、`--repo` と `--sample-data` で場所を指定します（「4. オプション」）。

## 2. 準備（最初に1回）

```bash
cd submission
git clone https://github.com/metrica-sports/sample-data.git ../../sample-data
pip install -r requirements.txt
```

- Python 3.11 を使います（3.11.18 で確認）。
- Windows で `python` と打つと Microsoft Store が開く場合は、conda などで入れた Python を使ってください。
  例：`C:\Users\<名前>\anaconda\envs\mtah\python.exe make_submission.py`

## 3. 提出ファイルを作る（これだけ）

```bash
cd submission
python make_submission.py
```

できるもの：

| ファイル | 中身 |
|---|---|
| `results/submission.csv` | **提出ファイル**（199,319 行、`frame,label`） |
| `results/report.txt` | 各段階で何をしたかの記録 |

最後に `Valid.` と出れば形式は正しいです（ラベルが正しいかどうかは分かりません）。
時間の目安：初回は約5分（学習用ファイル作りに約2分）、2回目からは約3分です。

### 中で何をしているか

`make_submission.py` は次の3つを順に実行するだけです。1つずつ実行しても同じ結果になります。

```bash
# (1) 学習用ファイルを作る：公開サンプル試合 → external/EVAL-G1〜G3 → results/team_feat/*.npz（すでにあれば飛ばす）
python tools/build_team_feat.py --repo ../../more-than-a-hack-2027 --sample-data ../../sample-data/data

# (2) 配布された試合の全フレームにラベルを付ける
python -m possession.v3.pipeline_v3 --repo ../../more-than-a-hack-2027 \
    --out results/submission.csv --report results/report.txt \
    --team-train results/team_feat/EVAL-G1.npz results/team_feat/EVAL-G2.npz

# (3) 形式の確認（公式スクリプト）
python ../../more-than-a-hack-2027/evaluation/validate_submission.py results/submission.csv
```

## 4. オプション

```bash
python make_submission.py --repo <公式リポジトリの場所> --sample-data <sample-data/data の場所> --out <出力先.csv>
```

配布データが公式リポジトリの `data/` 以外にある場合は、環境変数 `MTAH_DATA_DIR` で場所を指定します。

## 5. 精度を確かめる（他の試合で）

配布された試合の正解は使えないので、精度は Metrica の公開サンプル試合から作った評価用の試合で測ります。

```bash
python tools/score_eval3.py
```

| 試合 | spellF1 | 学習との関係 |
|---|---|---|
| EVAL-G1 | 0.6677 | G2 だけで学習したモデルで測る |
| EVAL-G2 | 0.6671 | G1 だけで学習したモデルで測る |
| EVAL-G3 | 0.5935 | 学習には一切使っていない |

このフォルダを新しく clone した状態から実行して、上の数字がそのまま出ることを確認済みです（2026-11-14）。

## 6. 処理の流れ

```
[A] 選手の誤検出を除く          possession/v2/clean_players.py
[B] ボール検出の信頼度          possession/v2/ball_quality.py
[C] Smart Tagging の整理        possession/v2/tag_policy.py、キックオフ検出 possession/v3/kickoff.py
[D] 特徴量とモデル              possession/features.py、possession/v2/features_v2.py、possession/v3/team_external.py
      P(デッドボール) … ロジスティック回帰（他の試合の正解で学習）
      P(ホーム保持)   … LightGBM 5 シードの平均（他の試合の正解で学習）
[D2] プレー停止（D）の判定      possession/v3/dead_situation.py（設定は params/dead_hybrid.txt）
[E] プレー中の H/A を時系列で決める  possession/v3/hmm_v3.py
[F] サッカーの当たり前の補正    possession/v3/pipeline_v3.py の後半
→ results/submission.csv（試合の外のフレームは X）
```

設定値はすべて `possession/v3/config_v3.py`（v3 で増えたもの）と `possession/v2/config_v2.py`、
`possession/config.py` にあります。初期値が提出版なので、何も指定しなくてよいです。

## 7. ルールとの関係

- **配布された試合には手作業のラベルを使っていません。** モデルの学習に使うのは、Metrica の公開サンプル試合
  （他の試合）の正解だけです。公開5分の正解も使っていません。
- **配布データはこのリポジトリに入れていません。** 学習用ファイル（`results/team_feat/*.npz`）も入れず、
  `tools/build_team_feat.py` で公開サンプル試合から毎回同じものを作ります。
- `external/`・`results/`・`cache/`・`*.npz` は `.gitignore` で除外しています。
- **同じ提出ファイルが作り直せます。** 乱数のシードは固定しています。

## 8. フォルダの中身

| ファイル | 内容 |
|---|---|
| `make_submission.py` | 提出ファイルを作るコマンド（上の3段階をまとめたもの） |
| `requirements.txt` | 必要なライブラリ（バージョン固定） |
| `possession/` | 推定の本体（v3 と、v3 が使う v1・v2 の部品だけ） |
| `params/dead_hybrid.txt` | プレー停止（D）判定の設定 |
| `tools/build_team_feat.py` | 学習用ファイルを公開サンプル試合から作る |
| `tools/make_eval_data.py`、`convert_metrica.py`、`epts_to_metrica_csv.py` | サンプル試合を本番と同じ形式（欠損・誤検出入り）に変換する |
| `tools/score_eval3.py` | 評価用の3試合で spellF1 を測る |

親フォルダのコードを改良したときは、このフォルダにもコピーしてから、もう一度
`python tools/score_eval3.py` と `python make_submission.py` を実行して確かめてください。
