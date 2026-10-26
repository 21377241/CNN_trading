"""
小样本测试：从 WRDS 拉 OptionMetrics option prices 一行片段，
压缩写成 parquet(zstd) + csv.gz，默认输出到 F:\\optionm\\_sample
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# SPX 一天，字段全开；行数通常几千～几万，足够验证通路
SECID = 108105
SAMPLE_DATE = "2024-06-03"
YEAR = 2024


def resolve_out_dir() -> Path:
    f_drive = Path(r"F:\optionm\_sample")
    if Path("F:/").exists():
        return f_drive
    # 远程/无 F 盘时落到项目下，避免直接失败
    fallback = Path(__file__).resolve().parents[1] / "data" / "optionm_sample"
    print(f"未检测到 F:，改用: {fallback}")
    return fallback


def main() -> int:
    try:
        import wrds
    except ImportError:
        print("缺少 wrds，请先: pip install wrds pyarrow pandas")
        return 1

    OUT_DIR = resolve_out_dir()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {OUT_DIR.resolve()}")
    print(f"样本: optionm.opprcd{YEAR} | secid={SECID} | date={SAMPLE_DATE} | SELECT *")

    t0 = time.time()
    # 首次会提示输入 WRDS 用户名/密码；若已配置 ~/.pgpass 则免交互
    db = wrds.Connection()
    sql = f"""
        SELECT *
        FROM optionm.opprcd{YEAR}
        WHERE secid = {SECID}
          AND date = '{SAMPLE_DATE}'
    """
    df = db.raw_sql(sql, date_cols=["date", "exdate", "last_date"])
    db.close()
    t_query = time.time() - t0

    if df is None or len(df) == 0:
        print("查询结果为空：检查订阅权限、日期是否为交易日、或表名年份是否存在。")
        return 2

    print(f"行数={len(df)}, 列数={df.shape[1]}")
    print("列名:", list(df.columns))
    print(df.head(3).to_string())

    pq_path = OUT_DIR / f"opprcd{YEAR}_secid{SECID}_{SAMPLE_DATE}.parquet"
    gz_path = OUT_DIR / f"opprcd{YEAR}_secid{SECID}_{SAMPLE_DATE}.csv.gz"

    t1 = time.time()
    df.to_parquet(pq_path, index=False, compression="zstd")
    t_pq = time.time() - t1

    t2 = time.time()
    df.to_csv(gz_path, index=False, compression="gzip")
    t_gz = time.time() - t2

    pq_mb = pq_path.stat().st_size / (1024 * 1024)
    gz_mb = gz_path.stat().st_size / (1024 * 1024)

    print("\n=== 小样本结果 ===")
    print(f"查询耗时: {t_query:.1f}s")
    print(f"parquet(zstd): {pq_path}  {pq_mb:.2f} MB  写盘 {t_pq:.2f}s")
    print(f"csv.gz:        {gz_path}  {gz_mb:.2f} MB  写盘 {t_gz:.2f}s")
    print("小样本 OK。确认无误后再按年全量下载。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
