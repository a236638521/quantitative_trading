import os
from pathlib import Path
import pandas as pd
from typing import Optional
from ..config import CACHE_DIR


class DataStorage:
    """本地行情数据持久化与缓存管理 (采用高效 Parquet 格式)"""

    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_file_path(self, symbol: str, period: str = "daily", adjust: str = "qfq") -> Path:
        clean_symbol = symbol.replace(".", "_")
        return self.cache_dir / f"{clean_symbol}_{period}_{adjust}.parquet"

    def save_bars(self, symbol: str, df: pd.DataFrame, period: str = "daily", adjust: str = "qfq"):
        """保存 K 线数据到本地缓存"""
        if df is None or df.empty:
            return
        file_path = self._get_file_path(symbol, period, adjust)
        # 确保时间列格式统一并排序
        df = df.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            df = df.sort_values("date").drop_duplicates(subset=["date"])
        df.to_parquet(file_path, index=False)

    def load_bars(self, symbol: str, period: str = "daily", adjust: str = "qfq") -> Optional[pd.DataFrame]:
        """从本地缓存加载 K 线数据"""
        file_path = self._get_file_path(symbol, period, adjust)
        if file_path.exists():
            try:
                df = pd.read_parquet(file_path)
                return df
            except Exception:
                return None
        return None

    def exists(self, symbol: str, period: str = "daily", adjust: str = "qfq") -> bool:
        return self._get_file_path(symbol, period, adjust).exists()
