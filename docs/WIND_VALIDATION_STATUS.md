# Wind validation status / 風解析の妥当性監査

最終更新: 2026-07-19

## English summary

The public residential-v1 dataset and both derived FNO checkpoints are
quarantined. One XLB sample is numerically catastrophic, the old height and
pedestrian-level coordinate contract is physically inconsistent, and the
corrected KBC setup fails the current grid-independence gate. The assets remain
downloadable only with an explicit audit override; they must not be used for
design evaluation or cited performance claims. Dataset-v2 generation and
training are intentionally blocked until the physical protocol passes.

## 現在の結論

街区生成、接道、築年、共同建替え、火災・費用のスクリーニング実装は継続して
利用できる。一方、風に依存する順位、Pareto案、Houdini heatmap、既報のXLB改善率は、
新しいデータセットとcheckpointが成立するまで研究結果として利用しない。

models-residential-v1 Releaseは再現・監査用に残すが、models/manifest.jsonを
quarantinedとし、通常のdownload_models.pyは取得を拒否する。過去状態を調べる場合だけ
allow-quarantinedを明示する。

## v1で見つかった問題

1. residential_xlb.npzのsample 374はfinite値のまま発散している。流入速度0.05に
   対し最大速度335.951であり、他sampleの代表的最大値約0.089と桁が異なる。
2. この1件を含めた学習用ystdは約0.1914、除外時は約0.0210で、約9.1倍に膨張する。
   v1 checkpointはこの汚染統計を保持している。
3. 旧入力は建物高さを水平100 mで割っており、鉛直物理領域を独立に定義していない。
   例えば256 × 256 × 64格子を100 m角として解釈すると、xyセル0.391 mに対し
   zセル1.563 mとなる。
4. 旧pedestrian_z=4は同じ解釈で約6.25 mとなり、歩行者高さ1.5 mではない。
5. 旧pipelineは低解像度形状をbackend内部でXLB格子へ拡大していたため、解像度変更が
   形状表現と流体格子を同時に変えていた。
6. 全sampleから正規化統計を計算し、test splitでcheckpointを選んでいたため、
   学習・validation・最終testの分離も不十分だった。

このためv1から得たFNO/XLBの絶対値、改善率、点ごとのRMSは撤回する。

## 修正後の物理・データ契約

- 建物高さはbuilding_height_m / domain_height_mで符号化する。
- 既定領域は100 m × 100 m × 60 m、Reynolds基準高さ10 m、評価高さ1.5 m。
- x・y・zセルをほぼ立方体にし、1.5 m面は隣接z sliceを線形補間する。
- 各解像度で元のベクトル形状をXLB格子へ直接ラスタライズする。
- backendは入力shapeとXLB xy格子の完全一致を要求し、暗黙の拡大縮小をしない。
- 非finite、負値、流入の8倍を超えるpeakをrejectし、破損cacheも再利用しない。
- dataset contract v2はsolver設定、backend署名、格子検証report hashを保存する。
- validatorはshape、物理metadata、重複、外れ値、空領域sample、共同化sampleを検査する。
- 正規化統計はtrain splitだけから計算し、validationでcheckpointを選び、
  untouched testは最後に一度だけ報告する。

## 格子感度の実測結果

同じseed 1の街区、同じ100 m × 100 m × 60 m領域、1.5 m補間面、Re=8000、
流入0.05、1 flow-through、末尾25%平均を用いた。形状被覆率は解像度間で
約31.45–31.57%とほぼ一定であり、単純な建築面積差だけでは説明できない。

| XLB格子 | step | 平均通風指標 | 弱風率 |
|---|---:|---:|---:|
| 128 × 128 × 77 | 2560 | 0.3228 | 0.6517 |
| 160 × 160 × 96 | 3200 | 0.3499 | 0.6269 |
| 200 × 200 × 120 | 4000 | 0.4353 | 0.4857 |

平均指標driftは0.1125、弱風率driftは0.1660で、既定許容値0.03を大きく超える。
さらに256格子の1 flow-throughと、200格子の2 flow-throughでは非finite発散を
確認した。したがって現在のKBC条件は格子収束しておらず、v2学習データ生成は停止する。

## 安全ゲート

次を先に実行する。

    python scripts/verify_residential_grid.py

reportがpassし、dataset生成に使う完全なXLB configとbackend署名が一致した場合だけ、
次の生成が進む。

    python scripts/gen_residential_dataset.py
    python scripts/validate_residential_dataset.py data/residential_xlb_v2.npz
    python scripts/train_residential_fno.py --data data/residential_xlb_v2.npz

allow-unconvergedは小規模な実装smoke専用であり、学習、公開、設計比較には使わない。

## 次に必要な物理検証

1. KBCの格子・時間安定性を整理し、Smagorinsky LES等を別protocolとして比較する。
   collision modelは結果を確認せず暗黙に切り替えない。
2. 現行の床・上面・側面no-slip境界を、外部風に適した上空・側方境界、
   粗度・大気境界層流入と比較する。
3. 街区周囲の上流・下流・側方paddingを増やし、100 m街区が計算領域を埋める影響を除く。
4. flow-through数、統計平均窓、風向、seedの感度を分離する。
5. 格子収束後に実測または風洞・独立CFDとの照合を行う。
6. その後に500件以上を生成し、FNOを再学習してPareto候補をXLBで再順位付けする。

これらが終わるまで、風は実装研究中の層であり、Mokumitsuの完成済み機能とは扱わない。
