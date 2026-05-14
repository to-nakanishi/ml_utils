import numpy as np
import pandas as pd


def downcast_numeric(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    ----------
    summary
    ----------
    数値列を最小のdtypeにダウンキャストしてメモリ使用量を削減する関数。
    DataFrameの数値列を走査し、値の範囲に応じて最小のdtypeに変換する。
    整数列は int8/int16/int32/int64 のいずれかに、浮動小数列は範囲が
    収まれば float32 に、そうでなければ float64 のままにする。

    ----------
    Parameters
    ----------
    df : pd.DataFrame
        入力DataFrame. ※in-placeで書き換えられる点に注意
    verbose : bool, default True
        Trueの場合、メモリ削減量を標準出力する。 

    ----------
    Returns
    ----------
    pd.DataFrame
        ダウンキャスト後のDataFrame(入力と同一オブジェクト)。

    ----------
    Examples
    ----------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'a': [1, 2, 3], 'b': [1.5, 2.5, 3.5]})
    >>> df_downcasted = downcast_numeric(df)
    >>> df_downcasted.dtypes
    a       int8
    b    float32
    dtype: object
    
    ----------
    Notes
    ----------
    float16は精度劣化(有効桁約3桁)とオーバーフロー(最大値約65,504)の
    リスクがあるためコメントアウト。 LightGBM/CatBoost は内部で
    float32 に変換するため、学習時のメモリ削減効果もない。

    Used in: Home Credit Default Risk (Kaggle)。
    """
    start_mem=df.memory_usage().sum()/1024**2

    numeric_cols = df.select_dtypes(include=['number']).columns

    for col in numeric_cols:
        col_type = df[col].dtype
        cmin = df[col].min()
        cmax = df[col].max()

        if str(col_type)[:3]=='int':
            if   cmin > np.iinfo(np.int8).min and cmax < np.iinfo(np.int8).max:
              df[col]=df[col].astype(np.int8)
            elif cmin > np.iinfo(np.int16).min and cmax < np.iinfo(np.int16).max:
              df[col]=df[col].astype(np.int16)
            elif cmin > np.iinfo(np.int32).min and cmax < np.iinfo(np.int32).max:
              df[col]=df[col].astype(np.int32)
            else:
              df[col]=df[col].astype(np.int64)
        else:
            # float16 は精度劣化(有効桁約3桁)とオーバーフロー(最大値約65,504)のリスクがあり、
            # ML用途では float32 で止めるのが現代の主流。LightGBM/CatBoost も内部で float32 に
            # 戻すため、学習時のメモリ削減効果はない。テーブルデータでは使わない。
            # if cmin > np.finfo(np.float16).min and cmax < np.finfo(np.float16).max:
            #     df[col] = df[col].astype(np.float16)
            # elif cmin > np.finfo(np.float32).min and cmax < np.finfo(np.float32).max:
            if cmin > np.finfo(np.float32).min and cmax < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)

    if verbose:
        end_mem = df.memory_usage().sum() / 1024**2
        reduction_mb = start_mem - end_mem
        reduction_pct = 100 * reduction_mb / start_mem
        print(f'メモリ使用量: {start_mem:.2f}MB -> {end_mem:.2f}MB | '
              f'削減: {reduction_mb:.2f}MB ({reduction_pct:.1f}%)')
    return df
