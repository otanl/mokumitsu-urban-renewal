# 木密地区の段階的更新・風・防災モデル

この文書は、日本の制度・木密文脈を含む研究方法の日本語正本である。
公開範囲と依存関係は[Architecture](ARCHITECTURE.md)、完成条件と優先順位は
[現状とロードマップ](ROADMAP.md)を参照する。

> **風解析監査（2026-07-19）:** residential-v1 dataset/checkpointと、そこから得た
> FNO/XLB改善率を撤回した。旧高さ・歩行者座標契約とsample 374の発散に加え、
> 修正後のKBC条件も格子独立性gateを通過していない。以下の風機能は実装仕様であり、
> 現在有効な性能結果ではない。詳細は[風解析の妥当性監査](WIND_VALIDATION_STATUS.md)。

## 目的

既存の密集市街地を、完成形だけでなく建替え順序まで含めて評価するための共通データ層。
同じ `MokumitsuDistrict` を、Houdini表示、延焼スクリーニング、FNO/XLB風評価へ渡す。

## 街区生成

```powershell
.venv\Scripts\python.exe scripts\gen_mokumitsu.py --seed 0
```

既定値は直交グリッドではない。100 m角の敷地内に、まず屈曲した旧道を通し、その道路から
方向持続性を持つ路地を枝状に成長させる。道路中心線を交差位置でノード化し、幅員分をバッファ
した残地をポリゴン化して不整形街区を得る。その後、各街区を主軸に沿って再帰分割し、街路沿い
敷地、背後地、狭い竿部分を持つ旗竿状敷地を生成する。建物も敷地主軸に沿う回転矩形であり、
軸平行の外接矩形へ戻していない。幾何演算には `Shapely>=2.0` を用いる。

生成結果には、非直交道路率、道路方位エントロピー、袋小路率、不整形敷地率、敷地面積変動係数、
旗竿状敷地率、背後地率を持つ `morphology_summary()` を適用できる。これらは「見た目だけを少し
歪めたグリッド」が既定値へ戻ることを防ぐ回帰テストにも使っている。

明示的な比較対照が必要な場合だけ、旧直交生成器を使用する。

```powershell
.venv\Scripts\python.exe scripts\gen_mokumitsu.py --grid --seed 0
```

通常の有機的生成器は次を再現可能な乱数シード付きで保持する。

- 法42条1項道路相当、42条2項道路相当、法上道路ではない路地
- 接道2 mを満たす敷地、狭あい道路でセットバックを要する敷地、未接道敷地
- 道路・敷地・建物のIDと接続関係
- 非直交の道路方位、屈曲、三叉路、袋小路、旗竿状敷地、背後地
- 建築年、構造、防火性能、階数
- FNO/XLBへ渡せる正規化高さマップ

建物密度は `target_net_building_coverage`（既定0.54）を目標にする。ただし、建物は敷地外へ
はみ出させず、セットバック後の不整形敷地に内包できる最大回転矩形を上限とするため、実現値は
seedによりおおむね宅地内40--46%、地区全体28--33%となる。既定100 m角では84--102棟/haである。
東京都の木造住宅密集地域抽出は住宅戸数密度55世帯/ha以上等を使うが、本モデルの棟数は世帯数
そのものではないため、これは形態校正の参考値としてのみ使う。

築年は独立一様乱数ではない。街区単位の更新履歴を共有し、接道不良・狭小敷地ほど古い建物が
残りやすい条件付き分布から、`1981年以前`、`1981--2000年`、`2001年以降`を生成する。1981年と
2000年は耐震・木造接合部規定の更新時期を表す区分であり、火災性能そのものは構造・防火性能で
別に保持する。

接道判定は計画比較用の近似であり、許可判定ではない。法43条の例外、自治体条例、道路指定の
個別事情は扱わない。実地区では行政の道路台帳等で置き換える。

## 建替え優先度

`renewal_priorities()` は、更新必要度と事業可能性を分けて返す。

- 必要度: 築年、構造、防火性能、接道リスク、隣棟間隔、敷地内建蔽
- 可能性: 連続2 m以上の法上道路接道、路地接道、未接道
- 推奨行為: 早期個別建替え、共同建替え/接道改善、計画的個別建替え、将来共同化

## 延焼スクリーニング

```powershell
.venv\Scripts\python.exe scripts\simulate_mokumitsu_fire.py --wind-dir 0 --wind-speed 5
```

建物間を有向辺とする確率的first-passageモデル。回転建物の実輪郭間距離を使い、辺の平均延焼時間は、隣棟間隔、出火側の
可燃性、着火側の感受性、建物間を横断する道路幅、風向との整合で変化する。各試行では延焼時間を
対数正規変動させ、指定時間内の焼失棟数・面積・各棟の被災確率を集計する。

これは更新案を高速比較するためのスクリーニングモデルであり、燃焼CFDでも、公的な市街地延焼
シミュレーションの代替でもない。絶対被害量として校正せず、案同士の相対比較に用いる。選定案は
消防研究センター等の確立したモデルで再検証する。

## 段階的建替え

```powershell
.venv\Scripts\python.exe scripts\simulate_mokumitsu_renewal.py --seed 0 --phases 6
```

`simulate_renewal_trajectory()` は、各期に同じ処理能力を割り当て、延焼影響を重ねた優先順位に
従って個別建替え可能な敷地を更新する。更新棟は耐火建築物・最低3階という明示的なシナリオ値へ
置換し、建築面積を変えずに床面積、非耐火率、延焼スクリーニング結果を時系列で記録する。

未接道・接道不足の敷地を勝手に合法化して建て替えることはしない。これらは
`joint_rebuild_or_access_improvement / deferred` として記録される。したがって現在のモデルは
「個別更新だけではどこで停滞するか」を測るベースラインであり、共同化後の敷地形状・権利変換・
道路拡幅は次の設計変数である。

## 夏季の歩行者風・通風スクリーニング

```powershell
.venv\Scripts\python.exe scripts\simulate_mokumitsu_renewal.py --seed 0 --phases 6 --summer-wind-rose "0:0.7,90:0.3"
```

`add_wind_to_trajectory()` は各更新期の高さマップを同じFNOへ渡し、空領域の平均風速
`u0` に対する `U/u0` を集計する。既定の相対閾値は、0.30未満を夏季の弱風・換気不足、
1.30超を強風として扱う。これは法令上の風環境評価尺度ではなく、同じ外部風条件における案同士の
相対比較用である。

全屋外に加えて、道路、敷地内空地、建物外周3 m帯を別々に評価する。JSONと時系列図には、
全屋外の平均風速比・弱風率・強風率、道路と敷地内空地、建物際の指標を保存する。
現在、上記指標を計算できる科学的に有効な公開residential modelはない。v1 Releaseは、
XLB sample 374の発散、建物高さを水平100 mで割る旧encoding、約6.25 m相当となる
旧pedestrian slice、全sample正規化統計のため隔離した。通常のdownloaderは取得を拒否し、
model loaderもv2物理metadataを持たないcheckpointを拒否する。

修正後は100 m × 100 m × 60 m、基準高さ10 m、評価高さ1.5 m、ほぼ立方体の
x/y/zセル、各XLB解像度への直接rasterizationを契約とした。dataset contract v2は
grid reportと完全なXLB configを保存し、generatorとtrainerは格子独立性がpassするまで
停止する。したがって、この節に以前記載していたseed 0の弱風率、v1の500件学習、
FNO/XLB誤差はすべて撤回し、再生成後に置き換える。

## 接道不良敷地の共同建替え

~~~powershell
.venv\Scripts\python.exe scripts\simulate_joint_renewal.py
$env:PYTHONUTF8=1
.venv\Scripts\python.exe scripts\verify_joint_renewal_xlb.py
~~~

`joint_renewal_candidates()` は、個別建替え不能な敷地と、同一街区内で実際に境界を共有する
個別建替え可能敷地のペアだけを候補にする。法的接道を新造せず、統合敷地は既存の接道を継承する。
2棟分の従前延べ床面積を1棟へ正確に移し、3～6階の範囲で建築面積を減らした複数配置を生成し、
住宅地FNOで宅地内弱風・全屋外弱風・道路弱風・建物際強風を評価して逐次選択する。

> **旧統合結果の撤回:** ここに記載していたFNO弱風率、256格子XLB改善率、
> およびそれで選んだ3件の共同建替え案はv1物理契約に依存するため、定量結果として
> 使用しない。床面積保存、接道解消、候補生成、簡易延焼のコード経路はテスト対象として
> 残すが、風を含む案の順位と統合効果はv2 model成立後に再計算する。

### 2～4敷地の連鎖共同化とPareto比較

~~~powershell
$env:PYTHONUTF8=1
.venv\Scripts\python.exe scripts\evaluate_cluster_renewal.py --include-recommended-district
.venv\Scripts\python.exe scripts\verify_cluster_renewal_xlb.py
~~~

ペア探索を一般化し、同一街区で実際の共有境界グラフが連結する2～4敷地を列挙する。候補は
単純な一体敷地へ統合でき、少なくとも接道困難敷地1件と既存接道を継承できる敷地1件を含むものに
限る。規模ごとに候補数を層別化してから配置案を作るため、権利者数の多い案が候補総数だけで有利に
ならない。

各案は統合後の空地を単なる「建築面積の残り」とせず、次の土地配分ポリゴンとして保持する。

- 既存接道が6 m未満の場合、道路中心線から6 m幅まで拡幅するための敷地控除
- 夏季主風向と平行で、統合敷地の投影長の55%以上を連続する幅2 mの通風空地
- 2敷地なら共有中庭、3～4敷地ならポケットパーク（統合後敷地の10%、最低8 m²）

道路用地は統合敷地から実際に差し引いて道路幅へ反映し、建物は全空地ポリゴンとの交差を禁止する。
通風空地と中庭/公園が既存接道へつながる面積を独立目的とする。全案で従前延べ床面積を正確に
維持し、次の6目的を明示して非支配解を残す。

- 夏季風配に対する弱風・建物際強風の合成ペナルティ
- 簡易延焼モデルの期待焼失建築面積
- 街区建築面積率
- 接道へ連結する通風空地・共有中庭/ポケットパーク面積
- 共同化に関与する権利数
- 解消できる接道困難敷地数

均衡重みは非支配集合から代表案を1件選ぶためだけに使い、Pareto集合そのものは重みで削らない。
> **旧Pareto結果の撤回:** 28案中16案、旧均衡案、FNO風ペナルティ、
> 256 × 256 × 64 XLBの改善率とRMSはv1風契約に依存するため撤回する。
> 非支配集合の計算法、床面積保存、空地polygon、接道困難敷地数、権利者数という
> 評価軸は残すが、代表案の選択と風・火災を含む数値はv2で再実験する。

### 共同建替えの期別事業・権利・住戸評価

~~~powershell
$env:PYTHONUTF8=1
.venv\Scripts\python.exe scripts\evaluate_joint_feasibility.py --include-districts
~~~

`optimize_joint_renewal()`が選んだ重複しない共同化を、既定3年間隔の完成事業として
順に再適用する。`JointRenewalFeasibilityPolicy`は、次の仮定を結果JSONへ明示する。

- 現行スキーマには世帯・住戸レコードがないため、既存住宅棟1棟を居住1住戸とする
- 新築住戸容量は、共同建物の延べ床面積 × 住宅有効率82% ÷ 目標住戸面積55 m²の切り捨て
- 各敷地の権利評価割合は敷地面積70%・従前延べ床30%の加重比で、権利床を連続量として保存
- 新築住戸は従前居住者の戻り住戸を先に確保し、残りを最大剰余法で権利割合に配分する
- 工事中は全従前住戸を仮移転とし、住戸数 × 工期月数を住戸月として累積する
- 解体、建設、道路・空地整備、仮移転、権利調整、設計監理、予備費を独立費目とする

新築容量が従前住戸を下回る場合は、不足を恒久転出住戸として明示し、住戸維持と扱わない。
土地取得、補助金、借入金利、割引現在価値はまだ含めない。既定の円単価は計算経路と
感度分析を成立させるための未較正シナリオ値であり、現在市場の見積りではない。

seed 0、0°:70%・90°:30%では、2036・2039・2042年に4敷地ずつ3事業が成立した。
12権利を変換し、各期4住戸を仮移転させながら全12住戸を戻し、街区住戸容量は
94から94で維持された。仮移転は最大4住戸、累積216住戸月である。接道連結空地は
累積173.03 m²、道路用地は15.08 m²となり、FNO風ペナルティは
0.33945から0.30481へ段階的に低下した。未較正の既定単価による費用は各期
2.1295、2.2165、1.9024億円、累積6.2485億円である。この絶対額ではなく、
代替案間の構成差と単価感度を初期研究の対象とする。

結果は `outputs/cluster_renewal_pareto.json/.png` と
`outputs/cluster_renewal_xlb.json/.png` に保存する。前者へ
`--include-recommended-district` を付けると、推奨街区を既存の `MokumitsuDistrict` schemaで
同梱する。土地配分は同じ案の`placement.open_spaces`へ実座標ポリゴンとして保存されるため、
Houdini可視化へも渡せる。

## Houdini

共同建替えの事業順序は、個別更新HIPとは分けた4フレームの専用シーンで確認する。

    $env:PYTHONUTF8=1
    $HYTHON = "C:\Program Files\Side Effects Software\Houdini 20.5.xxx\bin\hython.exe"
    .venv\Scripts\python.exe scripts\evaluate_joint_feasibility.py --include-districts
    & $HYTHON houdini\build_joint_feasibility_hip.py

examples/houdini_joint_feasibility.hipを開き、ビューポートカメラを
/obj/CAM_JOINT_TIMELINEへ切り替えてフレーム1～4を再生する。フレーム1は2026年現状、
フレーム2～4は2036・2039・2042年の完成状態である。橙色の既存棟・敷地輪郭が次の共同化、
桃色がその期の共同建物、紫色が既完成共同建物、水色が通風路、緑色が公園/中庭、
黄橙色が道路提供用地を表す。HUDには仮移転・戻り・恒久転出住戸、街区住戸容量、
未較正シナリオ費用、風ペナルティ、空地・道路用地を表示する。

CACHE_JOINT_TIMELINEは4期を
$HIP/cache/joint_feasibility_timeline.$F4.bgeo.scへ保存し、Load from Diskを
既定で有効にする。したがって通常の再生では結果JSONもFNOも火災計算も最適化も実行しない。
別結果へ差し替えるときだけLoad from Diskを無効にし、Feasibility result JSONを
変更してSave to Diskを実行する。

個別更新・火災・風の研究ビューは従来の別シーンとして生成する。

```powershell
$env:PYTHONUTF8=1
$HYTHON = "C:\Program Files\Side Effects Software\Houdini 20.5.xxx\bin\hython.exe"
& $HYTHON houdini\build_mokumitsu_hip.py
```

`/obj/mokumitsu_resilience/district_fire_screen` のパラメータで、シード、更新期、期当たり更新率、
夏季風向、火災時風向・風速、評価時間、モンテカルロ回数を変更できる。表示は築年、接道/
建替え可能性、総合優先度、更新済み/共同化待ち、被災確率、出火時延焼影響に加え、連続風速比と
弱風/快適域/強風の3区分を切り替える。風マップはFNO評価時に回転した場を元の街区世界座標へ
戻して重ね、道路は風場を隠さない輪郭表示となる。詳細属性にもCLIと同じ風指標を保存する。

既定ではタイムラインのフレーム1～7を更新期0～6へ対応させ、1 fpsで2026～2056年の
段階的個別建替えを再生する。`CACHE_TIMELINE`はHoudini標準のFile Cache SOP 2.0で、
7期を`$HIP/cache/mokumitsu_timeline.$F4.bgeo.sc`へ永続化する。通常再生では
建替え済み/共同化待ちの状態、延焼結果、FNO風指標を再計算せずディスクから読み込む。
`Drive renewal from timeline`を無効にすれば手動の更新期パラメータへ切り替えられる。
seed、表示指標、更新率、風・火災条件を変更する場合は`Load from Disk`を無効にして編集し、
`Save to Disk`でフレーム1～7を再計算する。保存完了後は`Load from Disk`が再び有効になる。

Houdini側でも同じShapely依存を使うため、初回だけHoudiniのPythonへ
`hython.exe -m pip install "shapely>=2.0"` が必要となる。風表示にはHoudini Pythonの
`torch` と `checkpoints/fno_residential_ts.pt/.json` も必要であり、モデルはセッション中だけ
`hou.session` にキャッシュする。

## 次段階

道路拡幅、公園、共有空地、風向に沿う空地連結の幾何操作に加え、事業費、権利床、
仮移転、住戸容量の累積スクリーニングまで初期モデルを実装した。

1. Houdini上のPareto候補比較、preview/XLB provenance、非同期評価を同じ評価器契約で接続する
2. 実在地区のGIS・世帯・権利者・費用資料で生成分布と事業評価を較正する
3. 複数seed・風向のPareto候補をXLBで再評価し、簡易延焼を校正済みモデルと比較する
4. 実気象の風配を入れ、外壁風圧・開口・室内換気回数の評価を後段へ接続する

汎用な環境多目的最適化エンジンは、二つ以上の独立用途が同じAPIを実証するまで
別リポジトリへ抽出しない。詳細な判断基準は[現状とロードマップ](ROADMAP.md)に置く。

## 制度・モデル参照

- [e-Gov 建築基準法（第42・43条）](https://laws.e-gov.go.jp/law/325AC0000000201)
- [国土交通省 密集市街地の改善整備](https://www.mlit.go.jp/jutakukentiku/house/jutakukentiku_house_fr5_000075.html)
- [東京都 防災都市づくり推進計画（木造住宅密集地域の抽出指標）](https://www.funenka.metro.tokyo.lg.jp/assets/pdf/promotion-plan/bosai4_172.pdf)
- [消防研究センター 市街地火災延焼シミュレーション](https://nrifd.fdma.go.jp/research/seika/kasai_ensyou/daikibo/index.html)
- [先行研究と本モデルの位置づけ](MOKUMITSU_RELATED_WORK.md)
