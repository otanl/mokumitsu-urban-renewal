# Mokumitsu

[English](README.md)

木造住宅密集地域を、不規則な道路・敷地・建物の集合として生成し、
防災上の更新必要性と、接道上の実現可能性を分けて評価したうえで、
火災・夏季風・床面積・オープンスペース・権利者数を比較する研究パッケージです。
Houdiniは可視化・形状編集の任意アダプタであり、研究コアの依存ではありません。

v0.1は、合成街区を対象とする研究プロトタイプとHoudiniサンプルとしてはいったん完成です。
ただし、実地区で較正・検証された計画支援ツールではありません。完成範囲と次段階は
[現状とロードマップ](docs/ROADMAP.md)、詳細な実験手順は
[研究ノート](docs/RESEARCH.md)に記録しています。

## インストール

    uv venv --python 3.12
    uv pip install -e ".[dev,viz]"

Houdiniを使わない生成・接道・火災・更新テストは、このプロジェクト単独で
実行できます。XLB検証を行う場合だけ、次を追加します。

    uv pip install -e ".[verify]"

風評価にはGitへ含めないTorchScript FNOとJSONメタデータが必要です。

    checkpoints\fno_residential_ts.pt
    checkpoints\fno_residential_ts.json

別の保存先を使う場合はMOKUMITSU_CHECKPOINT_DIR環境変数で指定します。
チェックポイントがなくても、街区生成・接道・火災・更新順位の機能とテストは実行できます。

## 研究フロー

    有機的な道路網・敷地生成
        → 築年／構造／耐火性能の条件付き付与
        → 接道・42条2項道路セットバックの代理評価
        → 個別更新の必要性と実現可能性
        → 連担2–4敷地の共同建替え候補
        → 道路拡幅用地・風向通路・共有中庭／ポケットパークの土地配分
        → 火災スクリーニング + 夏季風評価
        → 多目的Pareto候補
        → 期別の権利変換・仮移転・住戸容量・事業費スクリーニング
        → XLB・妥当性確認済み火災モデルで最終検証

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
切替やタイムライン再生は推論を再実行しません。これは設計比較用FNOスクリーニングであり、
速度ベクトル表示や最終候補のXLB検証を代替するものではありません。

個別更新・火災・風を比較する既存シーンは
[examples/houdini_mokumitsu.hip](examples/houdini_mokumitsu.hip)です。
`/obj/mokumitsu_resilience/district_fire_screen`で、街区seed、更新期、表示指標、
夏季風向、火災条件を変更できます。築年、接道、建替え順位、更新状態、延焼指標、
FNO歩行者風を同じ街区形状上で比較する研究ビューです。

HIPを開いてタイムラインを再生すると、フレーム1～7が更新期0～6に対応し、
既定の5年間隔で`2026 → 2056`年の個別建替えが進みます。初期状態、更新済み棟、
接道問題により共同化待ちとなる棟、延焼・風指標を比較できます。
`/obj/mokumitsu_resilience/CACHE_TIMELINE`は標準のFile Cache SOPで、7期分の
計算済みgeometryを`$HIP/cache/mokumitsu_timeline.$F4.bgeo.sc`へ保存します。
`Load from Disk`が既定で有効なため、通常の再生では火災・FNOを再計算しません。
`Drive renewal from timeline`をオフにすると、`Manual renewal phase`で任意期を
直接比較できます。条件や表示を変更する場合は、File Cache SOPの`Load from Disk`を
オフにして編集し、`Save to Disk`を押すとフレーム1～7を再計算してキャッシュを更新します。

clone後は、ローカルのHoudiniパスから再生成してください。

    $HYTHON = "C:\Program Files\Side Effects Software\Houdini 20.5.xxx\bin\hython.exe"
    .venv\Scripts\python.exe scripts\evaluate_joint_feasibility.py --include-districts
    & $HYTHON houdini\build_joint_feasibility_hip.py
    & $HYTHON houdini\build_mokumitsu_hip.py

既定の生成先はexamples\houdini_joint_feasibility.hipと
examples\houdini_mokumitsu.hipです。
再生成コマンドはFile Cache SOPを通して7期を一度計算し、HIP隣接の`cache`フォルダへ
保存するため時間がかかりますが、生成済みキャッシュの再生は即時です。
このHIPの風表示はFNOスクリーニングであり、最終候補のXLB検証とは分離しています。

## 現在の位置づけ

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

検証が必要:

- 実在地区GIS・建物台帳による生成分布の較正
- 市街地火災モデルとの比較
- XLBの解像度・流入境界・統計平均の感度
- 実在地区の世帯・権利・費用資料による事業評価係数の較正
- 日射、温熱、避難、資金調達・補助制度の導入
- 公開可能な学習済みFNOまたは再学習用データセットの配布

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
学習済み大容量checkpointとデータセットはGit外のアセットです。

次はHoudini上のパラメトリック編集、非同期評価、Pareto比較を改善します。
汎用な環境多目的最適化エンジンは、少なくとも二つの独立用途で同じ評価器APIが
実証されるまで第三のリポジトリへ切り出しません。
