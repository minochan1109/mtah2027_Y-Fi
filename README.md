# MTAH 2027 Phase 1：ボール保持推定

研究計画の ①〜④ を、機能ごとのモジュールとして実装したものです。
配布データ（追跡データ・ホモグラフィ・Smart Tagging）から、提出用の `frame,label` CSV を作ります。

## 重要：ルール変更（2026-10-22〜23）への対応

公式リポジトリの RULES.md が更新され、次の点が明確になりました。このコードはこれに従っています。

- **配布された試合には、どんな目的でも手作業のラベルを付けてはいけない**（学習用・検証用も不可）。
- **公開の5分間の正解は、パイプラインが正しく動くかの確認専用**。学習データにも、パラメータ調整にも使えない。
- 試合の特定フレームに結びついたルールや定数も不可。
- アルゴリズムが作った疑似ラベルでの学習、教師なし学習は可。**他の試合**の手作業ラベルは何件でも可。

そのため、学習には **アルゴリズムで作った疑似ラベルだけ** を使います（`pseudo_labels.py`）。
`--check` は公開5分で採点しますが、表示される点数で設定を比べたり調整したりしないでください。

## 構成

| ファイル | 研究計画との対応 | 内容 |
|---|---|---|
| `possession/config.py` | 全体 | 調整できる数値をすべてここに集約（秒・メートル・確率で表現） |
| `possession/data.py` | 入力・出力 | 追跡データ（ATD）、ホモグラフィ、Smart Tagging の読み込み、CSV の書き出し |
| `possession/preprocess.py` | 前処理 | メートル換算、速度、短いボール欠損の補間、カメラの向き、攻撃方向 |
| `possession/features.py` | ①②③ の特徴量 | `dead_features`（①）、`team_features`（②は `b_*`、③は `s_*` の列） |
| `possession/pseudo_labels.py` | 学習ラベル | ボールのコントロール、デッドボールの疑似ラベル（人は関与しない） |
| `possession/models.py` | ①②③ のモデル | ① ロジスティック回帰、②③ LightGBM、時間ブロック交差検証、自己学習 |
| `possession/hmm.py` | ④ | H/A/D の3状態 HMM＋ビタビ、デッドボールの最短時間、Smart Tagging の事前情報 |
| `possession/postprocess.py` | 後処理 | ラベル化、ハーフ外は X、保持回数などの集計 |
| `possession/pipeline.py` | 全体 | コマンドラインの入口 |
| `tools/convert_metrica.py` | 検証 | Metrica の公開サンプル試合（他の試合・ラベル付き）を同じ形式に変換し、正解 `gt.csv` も作る |
| `tools/overlay_video.py` | 確認 | 戦術カメラ映像に選手の枠・ボール・推定ラベル（外枠の色）を重ねた動画を作る |
| `tools/compare_external.py` | 検証 | v1 / v2 / 設定ファイルごとに、他の試合の spellF1 を表にする |
| `tools/plot_timeline.py` | 確認 | 指定した時間帯の推定結果・確率・疑似ラベルを図にする（映像と見比べる用） |
| `tests/synthetic.py` | テスト | 本物と同じ形式の合成データを作る（実データなしで動作確認するため） |
| `tests/test_smoke.py` | テスト | 合成データでパイプライン全体が動くかを確認 |

### 処理の流れ

```
data.load_*  →  preprocess.prepare  →  features.dead_features / team_features
                                          │
                     pseudo_labels（コントロール・デッドボールの疑似ラベル）
                                          │
              ① DeadModel → P(D)    ② TeamModel(b_*) → P(H)  ボールが見えるフレーム
                                    ③ TeamModel(s_*) → P(H)  ボールが見えないフレーム
                                          │
              ④ hmm: Smart Tagging の事前情報を混ぜる → ビタビ（ハーフごと）
                                          │
              postprocess → submission.csv（採点対象外は X）
```

## 準備

```bash
pip install -r requirements.txt
git clone https://github.com/metrica-sports/more-than-a-hack-2027.git
# データは公式の手順どおり more-than-a-hack-2027/data/ に置く（または MTAH_DATA_DIR）
```

## 実行

```bash
# 学習あり（疑似ラベル＋ロジスティック回帰／LightGBM＋HMM）
python -m possession.pipeline --repo ../more-than-a-hack-2027 --out results/team.csv

# 学習なし（ルールだけ＋HMM）。比較の基準として
python -m possession.pipeline --repo ../more-than-a-hack-2027 --mode rules --out results/rules.csv

# 公開5分で動作確認（点数で設定を選ばないこと）
python -m possession.pipeline --repo ../more-than-a-hack-2027 --check

# 提出前に必ず
python ../more-than-a-hack-2027/evaluation/validate_submission.py results/team.csv
```

- 初回は追跡データの読み込みに数分かかります。2回目以降は `cache/tracking.npz` を使います。
- 合成データ（実データと同じ 199,319 フレーム）では、全体で約3分でした。
- パラメータは `--params my_params.json`（`config.Params` の項目名と値）で上書きできます。

## バージョン2（試作）：ボールなしを基本にして、信頼できるボールを足す

```bash
python -m possession.v2.pipeline_v2 --repo ../more-than-a-hack-2027 --out results/v2.csv --report results/v2_report.txt --dump results/v2_detail.csv
```

| ファイル | 担当の目安 | 内容 |
|---|---|---|
| `possession/v2/clean_players.py` | [A] | 選手の誤検出の除去（ピッチ外・静止物）、キーパーのチームの修正、1チーム11人まで |
| `possession/v2/ball_quality.py` | [B] | ボール検出の信頼度（0〜1）、飛び・孤立・静止の誤検出の除去、信頼できる検出の間だけ補間、浮き球の判定 |
| `possession/v2/tag_policy.py` | [C] | タグの使い方の表（attack / defend / restart / goal / ignore と重み） |
| `possession/v2/features_v2.py` ほか | [D] | 基本の特徴量（陣形・内包・自陣への戻り）＋ボールの追加特徴量、疑似ラベル、モデル |
| `possession/v2/hmm_v2.py` | [E] | 切り替わりの兆候で切り替えコストを下げるビタビ |
| `possession/v2/config_v2.py` | 全員 | 各ファイルの数値（自分の担当の節だけ変える） |

良し悪しは他の試合で決める：

```bash
python tools/convert_metrica.py --src ../sample-data/data/Sample_Game_1 --out external/MET-G1F --stem MET-G1F --fake
python tools/compare_external.py --repo ../more-than-a-hack-2027 --games external/MET-G1 external/MET-G2 external/MET-G1F external/MET-G2F
```

`--fake` は本番に似た誤り（静止した偽ボール、ボールの飛び、ゴールポストの静止物、キーパーのチームの入れ替わり）を人工的に入れる。

## 評価用データを作る（他の試合。ルール上、調整に使ってよい）

Metrica の公開サンプル試合から、「本番に似せた使用データ」と「正解データ」を作る。

```bash
git clone https://github.com/metrica-sports/sample-data.git ../sample-data
python tools/make_eval_data.py --src ../sample-data/data/Sample_Game_1 --out external/EVAL-G1
python tools/make_eval_data.py --src ../sample-data/data/Sample_Game_2 --out external/EVAL-G2 --seed 1
```

| フォルダ | 中身 |
|---|---|
| `input/`（使用データ） | 全フレームの 41% でボール欠損、残りの検出の約 10% を誤検出に置き換え。選手もカメラの範囲外を欠損させ、検出の約 10% を誤り（チームの入れ替わり・キーパーの入れ替わり・静止物）にしたもの。タグは使用データからタグ付けルールで作ったもの |
| `truth/`（正解データ） | 本物の選手・ボール（`_atd`、`ball.csv`）、保持の正解（`gt.csv`）、同じタグ付けルールを正解データに当てたタグ、入れた誤りの記録（`errors.npz`） |

割合は `--ball-missing`・`--ball-false`・`--player-false` で変えられる。

```bash
# 推定全体（spellF1）
python tools/compare_external.py --repo ../more-than-a-hack-2027 --games external/EVAL-G1 external/EVAL-G2
# 選手の掃除とボールの判定だけ（再現率・除去率など）
python tools/eval_cleaning.py --games external/EVAL-G1 external/EVAL-G2 --params params/my_try.json
```

## 他の試合で点数を測る（旧：convert_metrica.py）

```bash
git clone https://github.com/metrica-sports/sample-data.git ../sample-data
python tools/convert_metrica.py --src ../sample-data/data/Sample_Game_1 --out external/MET-G1
python -m possession.pipeline --repo ../more-than-a-hack-2027 --external external/MET-G1 --out results/met_g1.csv
# 最後に「EXTERNAL GAME score (allowed for tuning): ... spellF1=...」が出る
```

- 正解はMetricaの人手のイベントデータから作っています（作り方は `convert_metrica.py` の先頭に記載）。技術文書の「学習データ」に出典として書いてください。
- サンプル試合は全員が映っている追跡データなので、本番に近づけるため、ボールと遠くの選手を一部わざと消しています（`--no-degrade` で無効）。
- 参考：Sample_Game_1 で spellF1 は約0.45でした（本番の点数とは別物です）。

## 推定結果を図で見る

```bash
python -m possession.pipeline --repo ../more-than-a-hack-2027 --out results/team.csv --dump results/detail.csv
python tools/plot_timeline.py --detail results/detail.csv --start 12:30 --seconds 60 --out results/12m30.png
```

横軸は映像の時刻（分:秒）なので、戦術カメラの映像を同じ時刻で開いて見比べられます。`--gt` は**他の試合の正解だけ**に使ってください。

## 映像に重ねて見る

```bash
python tools/overlay_video.py --repo ../more-than-a-hack-2027 --detail results/v0_detail.csv --start 20:00 --seconds 60 --out results/review_2000.mp4
```

外枠の色＝推定（H 青・A オレンジ・D 灰・X 黒）、選手の枠＝ホーム青・アウェイ オレンジ（位置から作った目安の大きさ）、黄色の輪＝ボール。
下の帯は前後10秒の推定。`--seconds 0` で最後まで（時間がかかる）。再生は VLC がおすすめ。
見て分かった「傾向」や「バグ」を直すのはよいが、「この区間は本当は H」と記録して学習・評価・設定選びに使うのはルール違反。

## 実データが届いたら最初に確かめること

1. ログに出る **Smart Tagging のチーム名の対応**（例：`{'Atletic Club': 1, 'Barcelona': 0}`）が正しいか。
2. `ball visible` の割合が DATA.md の値（約57%）に近いか。
3. 生の ATD の各グループが `x,y,speed` の順か（`data.load_tracking` はこの前提）。ボールの3番目の値は使っていません。
4. ログの「module 1 top coefficients」「module 2/3 top features」が常識に合うか。
5. 最後に出る集計（保持の回数・平均時間・D の割合）がサッカーとして自然か。正解と比べるのではなく、常識と比べてください。

## 検証のしかた（ルール上できること）

- 公開5分：フレームの対応、CSV の形式、ラベルの意味の確認だけ。
- 合成データ：`tests/synthetic.py` で、コードが壊れていないかを確認。
- **他の試合のデータ**：ラベル付きなら、ルール上は学習にも検証にも使えます（例：Metrica が公開しているサンプルデータ。形式の変換が必要）。
- 映像を見て、間違えた場面の傾向を調べるのは可。ただし、見た結果を特定フレームのラベルや定数にしてはいけません。

## 技術文書に書くこと

SUBMISSION.md の要件により、使った学習データをすべて書く必要があります。このコードでは「配布された試合に対する、アルゴリズムによる疑似ラベルのみ」です。他の試合のデータを足した場合は、その出典・誰がどれだけラベル付けしたかも書いてください。

## 注意

- 配布データはクラウドの LLM（ChatGPT、Claude など）に渡さないでください。コードの相談はデータ抜きで。
- `cache/`・`results/`・`data/`・`external/` はリポジトリに含めないでください（`.gitignore` 済み）。
