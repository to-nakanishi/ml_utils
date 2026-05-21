"""Aggregate a child table to one row per group key for baseline features."""

import pandas as pd


def aggregate_table(
    df: pd.DataFrame,
    group_key: str,
    sort_col: str,
    prefix: str,
) -> pd.DataFrame:
    """
    ----------
    summary
    ----------
    サブテーブル(1対多)を group_key 単位で1行に集約する関数。Home Credit の
    bureau / previous_application などの子テーブルを SK_ID_CURR 単位に畳み込む、
    試走(ベースライン)用の一括集約を想定する。

    生成する集約は4種:
      - 数値列: min / max / mean / sum / std
      - カテゴリ列(数値以外): nunique(ユニーク数)
      - 直近値: sort_col が最大の行の各列値(`_latest`)
      - sort_col 自体: min / max / count
        (sort_col は数値集約から除外し、ここで別途集約する。時間軸の列を想定し、
         例: DAYS_CREDIT なら min=最古・max=最新・count=レコード件数。
         時間軸に対して sum/std は意味が薄いため取らない)

    列名は必ず prefix を付けてフラット化する(`{prefix}{列}_{統計}`)。複数の
    サブテーブルを集約すると列名が衝突しうるため、prefix は必須引数とした
    (テーブルごとに 'BUREAU_' のように一意な接頭辞を渡すこと)。

    数値列は select_dtypes(include='number')、カテゴリ列は select_dtypes(
    exclude='number')で判定する。downcast や object→category 変換などの前処理を
    通した後でも、型の状態に依存せず「数値か / それ以外か」で振り分けられる。

    本関数は集約済み DataFrame を返すのみで、application 等への結合(merge)は
    行わない。結合キー名はテーブルにより異なり、結合方法は前処理側の責務である
    ため、呼び出し側で join すること。

    レコードが1件のみの group は std が NaN になるが、これは「ばらつきを評価
    できる情報が無い」というシグナルとして加工せず返す(必要なら後段で
    impute_by_target_rate 等で補完する)。

    ----------
    Parameters
    ----------
    df : pd.DataFrame
        集約対象のサブテーブル。`group_key` と `sort_col` を含む。
    group_key : str
        集約のキー列(例: 'SK_ID_CURR')。
    sort_col : str
        直近値の判定に使う列(例: 'DAYS_CREDIT')。最大値の行を直近とみなす。
    prefix : str
        生成列に付ける接頭辞(必須)。例: 'BUREAU_'。
        複数テーブル集約時の列名衝突を防ぐため、テーブルごとに一意にすること。　

    ----------
    Returns
    ----------
    pd.DataFrame
        group_key 1行ごとに集約した DataFrame(group_key は列として保持)。
        application 等への結合は行わない。

    ----------
    Examples
    ----------
    >>> bureau_agg = aggregate_table(bureau, 'SK_ID_CURR', 'DAYS_CREDIT', 'BUREAU_')
    >>> train = train.merge(bureau_agg, on='SK_ID_CURR', how='left')
    """
    if group_key not in df.columns:
        raise KeyError(f"group_key '{group_key}' not found in df.")
    if sort_col not in df.columns:
        raise KeyError(f"sort_col '{sort_col}' not found in df.")
    if df[group_key].isna().any():
        raise ValueError(f"group_key '{group_key}' contains NaN.")
    if len(df) == 0:
        raise ValueError("df is empty.")

    exclude = [group_key, sort_col]
    num_cols = [c for c in df.select_dtypes(include='number').columns if c not in exclude]
    # 数値以外をカテゴリ扱い(object / category / string などを漏れなく対象にする)
    cat_cols = [c for c in df.select_dtypes(exclude='number').columns if c not in exclude]

    # --- 数値型集約 ---
    num_agg = df.groupby(group_key)[num_cols].agg(['min', 'max', 'mean', 'sum', 'std'])
    num_agg.columns = [f'{prefix}{c}_{s}' for c, s in num_agg.columns]

    # --- カテゴリ型集約 ---
    if cat_cols:
        cat_agg = df.groupby(group_key)[cat_cols].agg(['nunique'])
        cat_agg.columns = [f'{prefix}{c}_{s}' for c, s in cat_agg.columns]
    else:
        cat_agg = pd.DataFrame(index=df[group_key].unique())

    # --- 直近値(sort_col が最大の行) ---
    latest = df.sort_values(sort_col).groupby(group_key).tail(1).set_index(group_key)
    latest_cols = {c: f'{prefix}{c}_latest' for c in num_cols + cat_cols}
    latest = latest[num_cols + cat_cols].rename(columns=latest_cols)

    # --- ソート列自体の集約 ---
    sort_agg = df.groupby(group_key)[sort_col].agg(['min', 'max', 'count'])
    sort_agg.columns = [f'{prefix}{sort_col}_{s}' for s in ['min', 'max', 'count']]

    # --- 結合 ---
    result = num_agg.join(cat_agg).join(latest).join(sort_agg).reset_index()
    return result
