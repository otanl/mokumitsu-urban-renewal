# Mokumitsu

[English](README.md)

木造住宅密集地域を、不規則な道路・敷地・建物の集合として生成し、
防災上の更新必要性と、接道上の実現可能性を分けて評価したうえで、
火災・夏季風・床面積・オープンスペース・権利者数を比較する研究パッケージです。
Houdiniは可視化・形状編集の任意アダプタであり、研究コアの依存ではありません。

v0.1は、合成街区を対象とする研究プロトタイプとしてはいったん完成しています。
さらにv0.2の最初の段階として、Houdini内で共同建替え形状を編集し、同じ研究コアで
風・火災・床面積・空地を自動再評価するサンプルを追加しました。ただし、実地区で
較正・検証された計画支援ツールではありません。完成範囲と次段階は
[現状とロードマップ](docs/ROADMAP.md)、詳細な実験手順は
[研究ノート](docs/RESEARCH.md)に記録しています。

![Mokumitsuの処理概要：街区生成、建替え方針、火災スクリーニング、統合優先順位](docs/assets/mokumitsu_overview.png)

*`scripts/render_readme_overview.py`でseed 0から再生成できる実装概要です。
有機的街区の生成、接道・築年による建替え方針、グラフ火災スクリーニング、
統合した更新優先順位を示します。風／CFD順位は格子検証gateを通過するまで
意図的に含めていません。*

**風モデル監査（2026-07-19）:** 公開residential-v1データセットとFNOは隔離しました。
XLB sample 374の数値発散、旧鉛直座標契約の誤り、修正後KBC条件の格子非収束が
確認されたためです。既報の風改善率と同梱heatmapは過去実装の表示例であり、研究結果や
設計判断には使えません。v2データ生成は物理検証gateで停止しています。詳細は
[風解析の妥当性監査](docs/WIND_VALIDATION_STATUS.md)を参照してください。

## インストール

    uv venv --python 3.12
    uv pip install -e ".[dev,viz]"

Houdiniを使わない生成・接道・火災・更新テストは、このプロジェクト単独で
実行できます。XLB検証を行う場合だけ、次を追加します。

    uv pip install -e ".[verify]"

現在、設計評価に使用できる公開住宅地FNOはありません。通常のdownloaderは隔離済み
v1 Releaseを拒否します。過去状態の再現・監査だけを目的とする場合は、次の明示optionを
使用できます。取得したv1 assetを設計比較、再学習、数値引用に使わないでください。

    .venv\Scripts\python.exe scripts\download_models.py --allow-quarantined --profile all --include-dataset

街区生成、接道、築年、火災、更新順位、権利、住戸、費用、空地の機能はcheckpointなしで
利用できます。新しい風pipelineは格子検証reportがpassするまでdataset生成を拒否します。

別の保存先を使う場合は、取得前にMOKUMITSU_CHECKPOINT_DIR環境変数で指定します。
チェックポイントがなくても、街区生成・接道・火災・更新順位の機能とテストは実行できます。

## 研究フロー

    有機的な道路網・敷地生成
        → 築年／構造／耐火性能の条件付き付与
        → 接道・42条2項道路セットバックの代理評価
        → 個別更新の必要性と実現可能性
        → 連担2–4敷地の共同建替え候補
        → 道路拡幅用地・風向通路・共有中庭／ポケットパークの土地配分
        → 火災スクリーニング + 夏季風評価（風はv2 gate通過後）
        → 多目的Pareto候補
        → 期別の権利変換・仮移転・住戸容量・事業費スクリーニング
        → 高忠実度CFD・火災物理モデルで別途検証（現在は未通過／未実装）

築年は一様乱数ではありません。街区ごとの継承度、接道リスク、敷地の小ささを
組み合わせて年代コホートを選び、その年代と接道条件から木造率・耐火性能・
建蔽傾向を条件付きで付与します。これは実測台帳の代替ではなく、
仮説を明示した合成データ生成器です。

## CLI

有機的な木密を生成:

    .venv\Scripts\mokumitsu.exe generate --seed 0 --out outputs\mokumitsu.json

建替え必要性を順位付け:

    .venv\Scripts\mokumitsu.exe prioritize --district outputs\mokumitsu.json --limit 20 --out outputs\mokumitsu_priorities.json

共同建替え案をPareto評価:

    .venv\Scripts\mokumitsu.exe pareto --district outputs\mokumitsu.json --wind-model residential --wind-rose "0:0.7,90:0.3" --out outputs\mokumitsu_pareto.json

複数期の共同建替えを事業面からスクリーニング:

    .venv\Scripts\mokumitsu.exe feasibility --district outputs\mokumitsu.json --projects 3 --wind-model residential --out outputs\mokumitsu_feasibility.json

評価図も含む再現用スクリプト:

    .venv\Scripts\python.exe scripts\evaluate_joint_feasibility.py

既定では幅6 mまでの道路拡幅用地、幅2 mの風向通路、統合敷地の10%を目安とする
共有中庭またはポケットパークを建物と重ならない実ポリゴンとして生成します。
`--target-road-width`、`--corridor-width`、`--open-space-fraction`で変更できます。

事業評価では、既存住宅棟1棟を1住戸とする仮定、新築延べ床の住宅有効率82%、
目標住戸面積55 m²を既定値とします。権利床は敷地面積70%・従前床面積30%の
重みで配分します。工事・解体・外構・仮移転・権利調整の単価もすべて
`JointRenewalFeasibilityPolicy`と結果JSONへ保存されます。これらは市場価格や
鑑定評価ではなく、感度分析のための未較正シナリオ値です。

モデル名は厳密に解決されます。別系統のチェックポイントへ暗黙に
フォールバックしません。研究結果JSONには解決済みモデルの由来を記録できます
（--hash-model）。

## Houdiniサンプル

共同建替え形状を直接スタディするシーンは
[examples/houdini_joint_design.hip](examples/houdini_joint_design.hip)です。
`/obj/mokumitsu_joint_design/LIVE_JOINT_DESIGN`で、共同化する2～4敷地、
候補順位、建物位置、建蔽率、縦横比、回転、高さ、道路提供幅、通風路幅、
共有空地率、夏季風配を変更できます。互換v2 checkpointがある場合だけFNO風速比を、
常に確率的火災、従前床面積の維持率、接道解消、連続空地を自動更新します。
解析用のRunボタンはありません。Resetは既定の実行可能形状へ戻すだけです。

建物が統合敷地、道路提供地、通風路または共有空地へはみ出す入力は、暗黙に
移動・縮小せずINVALIDとして表示します。表示レイヤだけの切替は
`VIEW_OPTIONS`に分離しているため、風表示のオン／オフでは評価を再実行しません。
各設計はgeometry、風配、方針、火災条件、モデルhashから決定論的keyを作り、
`$HIP/cache/joint_design`へ保存します。この動的cacheはGitへ含めません。

対話用workerとportable TorchScriptの実装経路自体は残していますが、現在の評価コードは
v2物理metadataを持たないcheckpointを拒否します。residential-v1 Releaseは監査目的で
保存する隔離assetです。互換モデルがなくても形状編集、火災、床面積、空地の検討は継続できます。
同梱HIPに保存されたFNO風速値も同じv1由来であり、表示機能の履歴確認に限定します。

共同建替えの事業順序を確認する専用シーンは
[examples/houdini_joint_feasibility.hip](examples/houdini_joint_feasibility.hip)です。
CAM_JOINT_TIMELINEを選び、フレーム1～4を再生すると、現状から2036・2039・2042年の
3事業へ進みます。橙は次に共同化する4棟、桃はその期に完成した共同建物、紫は既完成建物、
水色は通風路、緑は公園、黄橙は道路提供用地です。画面下のHUDに仮移転、戻り住戸、
恒久転出、住戸容量、期別・累積費用、風指標、空地面積を表示します。

/obj/mokumitsu_joint_feasibility/CACHE_JOINT_TIMELINEは4期を
$HIP/cache/joint_feasibility_timeline.$F4.bgeo.scへ保存済みです。
通常再生ではJSON、FNO、火災、最適化を再計算しません。

風の分布を見るには、`/obj/mokumitsu_joint_feasibility/WIND_DISPLAY_TOGGLE`を選び、
`Show cached FNO wind field`をオンにします（サンプルHIPの既定はオン）。
0°風70%・90°風30%を世界座標へ戻して合成した歩行者高さのスカラー風速比
`U/U0`を表示し、青は弱風（0.3未満）、緑付近は基準風速、黄～赤は強風側
（1.3超）を示します。オフにすると通常の地面・道路表示へ戻ります。
各期の128×128 FNO結果は32×32へ面積平均して同じFile Cacheへ保存済みなので、
切替やタイムライン再生は推論を再実行しません。ただし値は隔離済みv1由来の履歴表示であり、
設計比較、速度ベクトル表示、最終候補のXLB検証には使用できません。

個別更新・火災・風を比較する既存シーンは
[examples/houdini_mokumitsu.hip](examples/houdini_mokumitsu.hip)です。
`/obj/mokumitsu_resilience/district_fire_screen`で、街区seed、更新期、表示指標、
夏季風向、火災条件を変更できます。築年、接道、建替え順位、更新状態、延焼指標を
同じ街区形状上で比較する研究ビューです。保存済みFNO歩行者風は隔離済みv1の履歴表示です。

HIPを開いてタイムラインを再生すると、フレーム1～7が更新期0～6に対応し、
既定の5年間隔で`2026 → 2056`年の個別建替えが進みます。初期状態、更新済み棟、
接道問題により共同化待ちとなる棟、延焼・風指標を比較できます。
`/obj/mokumitsu_resilience/CACHE_TIMELINE`は標準のFile Cache SOPで、7期分の
計算済みgeometryを`$HIP/cache/mokumitsu_timeline.$F4.bgeo.sc`へ保存します。
`Load from Disk`が既定で有効なため、通常の再生では火災・FNOを再計算しません。
`Drive renewal from timeline`をオフにすると、`Manual renewal phase`で任意期を
直接比較できます。非風条件の編集は可能ですが、現在は有効なv2 checkpointがないため、
風を含むFile Cacheを再生成して設計評価へ使わないでください。

同梱HIPのPython SOPは生成PCの絶対パスを持たず、標準cloneではHIP位置から
`src`と`houdini`を実行時に解決します。保存済み履歴を確認するだけなら
再生成は不要です。標準外のworker環境だけ`MOKUMITSU_PYTHON`で指定します。

    $HYTHON = "C:\Program Files\Side Effects Software\Houdini 20.5.xxx\bin\hython.exe"
    .venv\Scripts\python.exe scripts\evaluate_joint_feasibility.py --include-districts
    & $HYTHON houdini\build_joint_design_hip.py
    & $HYTHON houdini\build_joint_feasibility_hip.py
    & $HYTHON houdini\build_mokumitsu_hip.py

既定の生成先はexamples\houdini_joint_design.hip、
examples\houdini_joint_feasibility.hip、examples\houdini_mokumitsu.hipです。
タイムライン2種の再生成はFile Cache SOPを通して各期を一度計算し、HIP隣接の
`cache`フォルダへ保存するため時間がかかりますが、生成済みキャッシュの再生は即時です。
このHIPの風表示はFNOスクリーニングであり、最終候補のXLB検証とは分離しています。

## 現在の位置づけ

以下の完成判定から風性能層を一時的に除外する。風に依存するPareto順位とHoudini表示は、
v2 checkpointが格子・境界・時間平均のgateを通過した後に再評価する。

「研究コードとしてのv0.1完成」と「実務ツールとしての完成」を区別します。
前者は、決定論的な合成街区、テスト、CLI、共同建替え評価、Houdiniの
キャッシュ済みサンプルまでを再現可能な単位として公開することで達成しています。
後者には実在地区GIS、風・火災・費用・制度条件の較正と不確実性評価が必要です。

実装済み:

- グリッドではない有機的道路・袋路・旗竿／裏敷地
- 接道代理評価と更新順位
- グラフ型の確率的火災延焼スクリーニング
- 段階的／共同建替え
- 共同化に伴う道路用地・風向連結通路・共有空地の幾何配分
- 共同建替えの期別事業費・権利床配分・仮移転・住戸容量の累積評価
- 複数風向のFNO評価
- 2–4敷地候補のPareto選別
- Houdini内のパラメトリック共同建替えと常駐preview worker

検証が必要:

- 実在地区GIS・建物台帳による生成分布の較正
- 市街地火災モデルとの比較
- XLBの格子独立性、collision model、流入・上空・側方境界、padding、統計平均の感度
- 実在地区の世帯・権利・費用資料による事業評価係数の較正
- 日射、温熱、避難、資金調達・補助制度の導入
- 配布FNOの複数seed・複数風向XLB検証と誤差範囲の公開

接道評価は建築確認判断ではなく計画初期の代理指標です。火災モデルとFNOも
候補絞り込み用であり、Pareto採用案をそのまま設計解とは扱いません。

## 公開境界

このリポジトリに含める対象:

- src/mokumitsu
- package CLI と更新・XLB検証スクリプト
- 木密関連テスト
- 生成仮定、評価式、再現可能な実験設定
- Houdini用の薄い可視化アダプタと最小サンプルHIP／キャッシュ

汎用CFD接続は独立したhoudini-xlbに置き、Mokumitsuから一方向に利用します。
学習済み大容量checkpointとデータセットはGit外のアセットです。公開v1 Releaseは
SHA-256を維持したまま隔離し、通常のdownloaderでは取得できません。v2向けの生成、
厳格検証、train/validation/test分離学習、TorchScript exportをリポジトリ内に置きます。

Houdini上のパラメトリック編集と常駐worker、決定論的cacheまでは実装しました。
次はXLBの格子・境界・時間平均gateを通すことを優先します。有効なv2 checkpointの後に、
Pareto候補ブラウザ、previewとXLB検証のprovenance表示、明示的なXLB実行、
より重い評価に対する非同期化またはdebounceを再開します。
汎用な環境多目的最適化エンジンは、少なくとも二つの独立用途で同じ評価器APIが
実証されるまで第三のリポジトリへ切り出しません。
