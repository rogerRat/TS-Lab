"""Shared deprecated ETT source configuration for explicit compatibility use."""

ETT_REPO_URL = "https://github.com/zhouhaoyi/ETDataset/raw/main/ETT-small"
ETT_FILES = ("ETTh1.csv", "ETTh2.csv", "ETTm1.csv", "ETTm2.csv")
ETT_SUBSETS = [filename[:-4] for filename in ETT_FILES]
