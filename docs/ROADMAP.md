# 現状とロードマップ / Project status and roadmap

## English summary

Version 0.1 is a complete synthetic research prototype, but not a calibrated
planning tool. The first v0.2 slice now provides live parametric joint design in
Houdini, a shared Python evaluator, deterministic cache keys and a persistent
preview worker. Pareto browsing, provenance and explicit XLB actions remain.
A generic environmental multi-objective framework should be
extracted only after a second independent design domain proves the same evaluator
contract.

## 現時点の判断

Mokumitsuは、次の意味ではいったん完成とする。

- 合成木密街区を決定論的に生成できる。
- 築年、構造、接道、防災の仮定と代理評価をコード上で追跡できる。
- 個別建替えと2～4敷地の共同建替えを比較できる。
- 火災、夏季風、床面積、接道解消、連続空地、権利者数を同じ候補上で評価できる。
- 共同建替えを期別の権利変換、仮移転、住戸容量、概算費用へ展開できる。
- Houdiniで更新順序とキャッシュ済みFNO風速比を再生できる。
- Houdiniで共同建替え形状を編集し、FNO・火災・床面積・空地を自動再評価できる。
- 最終候補を別リポジトリのXLBで再計算する経路がある。
- CPUテストで研究コアの主要な不変条件を確認できる。

一方、以下の意味では未完成である。

- 実在地区の再現・予測ツール
- 建築確認、接道判定、権利変換または事業採算の判断ツール
- 認定・検証済みの市街地火災シミュレータ
- 室内換気、温熱快適性、日射を含む総合環境設計ツール
- 不確実性を含む政策評価モデル

したがってv0.1の表現は「synthetic research prototype」とし、
「都市更新最適解」や「実務利用可能」とは表現しない。

## リポジトリ方針

### 維持する二つの公開単位

1. **houdini-xlb**
   - Houdini geometryとXLB/Warpを接続する汎用CFDアダプタ
   - Solver SOP、常駐worker、キャッシュ、解析結果表示
   - 木密固有の権利、事業、火災、更新目的を持たない

2. **mokumitsu-urban-renewal**
   - 木密地区のデータモデル、生成仮定、評価式、更新・共同化ロジック
   - FNO・火災のスクリーニングとXLB検証スクリプト
   - 木密固有のHoudini可視化アダプタと最小サンプル

依存方向はMokumitsuからhoudini-xlbへの一方向とする。

### 今は作らない第三リポジトリ

汎用な「環境要因の多目的最適化エンジン」は現段階では切り出さない。
理由は、共通化対象が実質的に一つの研究用途だけであり、今抽象化すると、
本当に必要な入出力よりも仮想的な汎用性を優先する可能性が高いためである。

次の条件をすべて満たしたときに抽出を再検討する。

- Mokumitsu以外の独立した設計用途が一つ以上ある。
- 両用途が同じ評価器、制約、provenance、cache-key契約を利用する。
- FNO以外の高速評価器とXLB以外の検証器を同じ契約で交換できる。
- Houdiniを使わないバッチ利用とHoudini内の対話利用が同じAPIを共有する。
- 抽出によって重複が減り、利用手順が増えない。

## 優先順位

### v0.1 — 公開可能な研究スナップショット

完了条件:

- 独立したPython packageとしてbuild・install・testできる。
- MITライセンス、日英README、研究ノート、先行研究、既知の限界が揃う。
- モノレポ固有パスを排除する。
- 大容量checkpoint、dataset、生成outputをGitへ含めない。
- サンプルHIPは同梱cacheだけで開いて再生できる。
- FNOとXLBを同一視しない説明を全ドキュメントで維持する。

### v0.2 — Houdini内のインタラクティブ性能設計

進行中。単にスライダーを増やすのではなく、判断過程を見えるようにする。

- 建物位置、高さ、共同化範囲、空地形状をパラメトリックに編集する。
- 編集中はFNO・火災の高速previewを遅延実行または非同期実行する。
- 入力geometry、モデル、風配、目的、制約から決定論的cache keyを作る。
- 全体目的だけでなく、弱風、強風、接道、延焼、住戸、費用を個別表示する。
- 現案、Pareto候補、XLB検証済み候補を明確に区別する。
- タイムラインは更新期または最適化世代のどちらかを明示して使う。
- XLB実行は明示操作とし、previewの自動再cookへ混ぜない。

成果物:

- [x] baselineと設計案を比較するHoudini非依存evaluator
- [x] geometry・scenario・policy・model identityを含む決定論的cache key
- [x] Houdini parameter-to-evaluator adapter
- [x] 常駐project-Python workerによるCUDA/CPU preview
- [x] 同一候補・同一key・cache復帰を確認する単体／headless HIPテスト
- [x] FNO checkpoint・XLB学習データのRelease配布とSHA-256 manifest
- [ ] Pareto candidate browser
- [ ] preview/verification provenance panel
- [ ] 明示的なXLB検証操作と結果の固定
- [ ] 重い評価向けのdebounceまたは非同期cook

現状のworkerは同期cookだが、モデル、街区、基準案、風maskをprocess内に保持し、
編集後のwarm previewを短縮する。初回起動は依然として重く、Houdini UIを完全に
非同期化したわけではない。「real-time CFD」ではなく「interactive screening」
として扱う。

### v0.3 — 研究妥当性

- 公開可能な実地区GISを一地区以上取り込む。
- 道路幅、敷地面積、建物年代・構造分布を実データで較正する。
- 複数seed・複数風向でFNOとXLBを比較し、zone metricの誤差を報告する。
- XLBの格子、流入、平均時間、境界条件の感度を調べる。
- 火災モデルを既存の市街地延焼モデルまたは公開事例と比較する。
- 単価、住戸、権利配分の感度と不確実性を報告する。
- 最適化が代理モデルの誤差を利用していないか確認する。

### v0.4 — 環境要因の拡張

v0.3の評価基盤が安定してから候補にする。

- 日射・日影
- 屋外温熱指標
- 避難・消防アクセス
- 騒音
- 室内換気への圧力係数連携

各評価器は単位、空間mask、方向・季節、信頼度、計算費用を明示する。
目的を増やす前に、意思決定者が比較可能な出力設計を行う。

## 現在の主要課題

| 優先度 | 課題 | 影響 | 対応 |
|---|---|---|---|
| 高 | 実地区較正がない | 形態・火災・費用の外的妥当性が不明 | 公開GISによるcase study |
| 高 | FNO最適化候補のXLB検証数が少ない | surrogate exploitationを否定できない | seed・風向を増やしたvalidation matrix |
| 中 | 公開model assetが合計約583 MB | 初回取得に時間がかかる | portable/accelerated profileを分け、将来は小型化も比較 |
| 中 | 常駐workerとcache後もHoudini cook自体は同期 | 初回起動や重い評価でUIが待つ | 実装済みwarm worker/keyed cacheに加え、debounceまたは非同期結果反映 |
| 中 | 目的値の意味が画面だけでは伝わりにくい | 結果を「最適解」と誤読しやすい | zone overlay、凡例、provenance、比較表 |
| 中 | 火災と風が別のscreening model | 複合災害の相互作用を表していない | まず個別較正後に連成の必要性を評価 |
| 低 | 汎用optimizerの抽出 | 現段階では保守負担が増える | v0.2以降の実績で再判断 |

## 言語方針

- README.md、package metadata、API名、docstring、contribution guideは英語。
- README.ja.md、制度・建築・都市計画の詳細、国内資料の読解は日本語。
- 研究上重要な結論と限界は日英READMEの両方に記載する。
- 機械的な全文二重管理は避け、詳細文書にはどちらが正本かを明示する。

この方針では、国際的なOSS利用と、日本固有の制度・木密文脈の精度を両立できる。
