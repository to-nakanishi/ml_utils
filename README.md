# ml_utils

![CI status](https://github.com/to-nakanishi/ml_utils/actions/workflows/ci.yml/badge.svg)

Kaggleコンペや与信モデリングで再利用可能なPython関数群。
[Home Credit Default Risk](https://github.com/to-nakanishi/home_credit_default_risk) をはじめとするポートフォリオで実際に使用した処理を、再利用可能な形にリファクタリングし、テスト付きで汎用化しています。

## Installation

```bash
git clone https://github.com/to-nakanishi/ml_utils.git
cd ml_utils
pip install -e .
```

modeling モジュール(`run_baseline` など)を使う場合は、LightGBM / CatBoost を含めてインストール:

```bash
pip install -e ".[modeling]"
```

## Development

### Setup

```bash
git clone https://github.com/to-nakanishi/ml_utils.git
cd ml_utils
pip install -e ".[dev,modeling]"
```

### Run tests

```bash
pytest tests/ -v
```

## Structure

### Current

```
ml_utils/
├── ml_utils/                      # パッケージ本体
│   ├── feature_engineering/
│   │   ├── memory.py              # メモリ最適化(ダウンキャスト)
│   │   ├── target_encoding.py     # OOF + Smoothing Target Encoding
│   │   ├── composite.py           # 平均系の合成特徴量
│   │   ├── imputation.py          # デフォルト率ベースの欠損補完
│   │   ├── density.py             # TARGET別密度の交点(リスク反転境界)算出
│   │   └── aggregation.py         # サブテーブルの group_key 単位一括集約
│   ├── modeling/
│   │   └── baseline.py            # LGBM/CatBoost ベースライン試走
│   └── validation/                
│       └── splitters.py           # 時系列CV(スライド窓/拡大窓)
├── tests/                         # pytest テストコード
│   ├── feature_engineering/
│   │   ├── test_memory.py
│   │   ├── test_target_encoding.py
│   │   ├── test_composite.py
│   │   ├── test_imputation.py
│   │   ├── test_density.py
│   │   └── test_aggregation.py
│   ├── modeling/
│   │   └── test_baseline.py
│   └── validation/             
│       └── test_splitters.py
├── pyproject.toml
├── README.md
└── LICENSE
```

### Planned

```
ml_utils/
├── ml_utils/
│   ├── feature_engineering/       
│   ├── modeling/                  # baseline 実装済み、学習ラッパー等を追加予定
│   └── validation/              
└── tests/
└── (各モジュールに対応するテスト)
```

## Modules

### `feature_engineering.memory`
DataFrameのメモリ使用量を削減するダウンキャスト関数。
| Function | Description |
|----------|-------------|
| `downcast_numeric` | 数値列をfloat32 / int8〜int32に最適化 |

### `feature_engineering.target_encoding`
リーク防止と過学習抑制を組み込んだターゲットエンコーディングと、その品質診断。
| Function | Description |
|----------|-------------|
| `target_encode_oof` | OOF + Smoothing でリーク防止しつつカテゴリを target 率にエンコード |
| `diagnose_target_encoding` | エンコード結果の品質を診断(小グループ数、リーク疑い、test カバレッジ等を dict で返す) |
| `bin_target_encode_oof` | 連続値をビン化して OOF + Smoothing で target 率にエンコード(target_encode_oof の数値版)。欠損は独立ビンとして欠損自体のデフォルト率でエンコード。bins は int(等幅)/ 明示境界の両対応 |

### `feature_engineering.composite`
厳選した数値列から平均系の合成特徴量を生成。
| Function | Description |
|----------|-------------|
| `make_composite_features` | 複数列から算術平均・標準偏差・幾何平均(+ 加重平均)を生成 |

### `feature_engineering.imputation`
ターゲット率ベースの欠損補完。
| Function | Description |
|----------|-------------|
| `impute_by_target_rate` | 欠損をデフォルト率が近いビンの代表値で補完(欠損フラグ列も任意で追加) |

### `feature_engineering.density`
TARGET 別の密度分布から、リスクの向きが反転する境界(交点)を算出。
| Function | Description |
|----------|-------------|
| `find_density_crossover` | TARGET=0/1 の KDE が交差するスコアを列ごとに算出し `{列名: 交点}` を返す。返した交点で `(df[col] < 交点)` の閾値フラグを生成できる |

### `feature_engineering.aggregation`
サブテーブル(1対多)を group_key 単位で1行に集約 / 集約前の診断。
| Function | Description |
|----------|-------------|
| `aggregate_table` | bureau等のサブテーブルを group_key 単位で集約(数値: min/max/mean/sum/std、カテゴリ: nunique、直近値、レコード数)。試走用の一括集約 |
| `diagnose_aggregation` | 集約前にキーの素性を診断。repeat_rate(1対多の度合い)、欠損数、直近値が一意か等を dict で返す |

### `modeling.baseline`
LightGBM / CatBoost によるベースライン試走。
| Function | Description |
|----------|-------------|
| `run_baseline` | SKF の OOF AUC と5fold平均の特徴量重要度を算出し、両モデルの比較表を返す。クリーニング後・FE後の節目で性能を確認する2値分類用の試走。SHAP用に全データ学習モデルも返す |

### `validation.splitters`
時系列クロスバリデーション。
| Function | Description |
|----------|-------------|
| `SlidingWindowSplit` | 「直近N期間で学習→次の1期間を検証」を時間順にスライドする時系列CV。sklearn互換で run_baseline の cv に渡せる。expanding でスライド窓/拡大窓を切替。IEEEのような時系列リークを防ぐ |

(以下、関数を追加するたびに更新)

## Requirements

- Python >= 3.11
- pandas >= 2.0
- numpy >= 1.24
- scikit-learn >= 1.3
- lightgbm >= 4.0 / catboost >= 1.2 (modeling のみ)

## License

MIT
