# ml_utils

Kaggleコンペや与信モデリングで再利用可能なPython関数群。
[Home Credit Default Risk](https://github.com/to-nakanishi/home_credit_default_risk) プロジェクトで実際に使用した関数を汎用化して切り出しています。

## Installation

```bash
git clone https://github.com/to-nakanishi/ml_utils.git
cd ml_utils
pip install -e .
```

## Structure
```
ml_utils/
├── feature_engineering/   # 特徴量エンジニアリング
│   ├── memory.py          # メモリ最適化(ダウンキャスト)
│   ├── target_encoding.py # OOF + Smoothing Target Encoding
│   ├── composite.py       # 複数列の合成特徴量
│   ├── imputation.py      # 欠損補完
│   └── flags.py           # 境界値・ラウンド値フラグ
├── validation/            # CV戦略
└── modeling/              # モデル学習ラッパー
```
## Modules

### `feature_engineering.memory`

DataFrameのメモリ使用量を削減するダウンキャスト関数。

| Function | Description |
|----------|-------------|
| `downcast_numeric` | 数値列をfloat32 / int8〜int32に最適化 |

(以下、関数を追加するたびに更新)

## Development

### Setup

```bash
git clone https://github.com/to-nakanishi/ml_utils.git
cd ml_utils
pip install -e ".[dev]"
```

### Run tests

``bash
pytest tests/ -v
```


## Requirements

- Python >= 3.10
- pandas >= 2.0
- numpy >= 1.24

## License

MIT
