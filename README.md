# ml_utils

Kaggleコンペや与信モデリングで再利用可能なPython関数群。
[Home Credit Default Risk](https://github.com/to-nakanishi/home_credit_default_risk) プロジェクトで実際に使用した関数を切り出し、汎用化しています。

## Installation

```bash
git clone https://github.com/to-nakanishi/ml_utils.git
cd ml_utils
pip install -e .
```

## Structure

### Current
```
ml_utils/
├── ml_utils/                  # パッケージ本体
│   └── feature_engineering/
│       ├── memory.py          # メモリ最適化(ダウンキャスト)
│       ├── target_encoding.py # OOF + Smoothing Target Encoding
│       ├── composite.py       # 平均系の合成特徴量
│       └── imputation.py      # デフォルト率ベースの欠損補完
├── tests/                     # pytest テストコード
│   └── feature_engineering/
│       ├── test_memory.py
│       ├── test_target_encoding.py
│       ├── test_composite.py
│       └── test_imputation.py
├── pyproject.toml
├── README.md
└── LICENSE
```

### Planned

```
ml_utils/
├── ml_utils/
│   ├── feature_engineering/
│   │   ├── memory.py          # 実装済み
│   │   ├── target_encoding.py # 実装済み
│   │   ├── composite.py       # 実装済み
│   │   ├── imputation.py      # 実装済み
│   │   └── flags.py           # 境界値・ラウンド値フラグ
│   ├── validation/            # CV戦略
│   └── modeling/              # モデル学習ラッパー
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

(以下、関数を追加するたびに更新)

## Development

### Setup

```bash
git clone https://github.com/to-nakanishi/ml_utils.git
cd ml_utils
pip install -e ".[dev]"
```

### Run tests

```bash
pytest tests/ -v
```


## Requirements

- Python >= 3.10
- pandas >= 2.0
- numpy >= 1.24

## License

MIT
