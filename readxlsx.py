## 读取excel文件
import pandas as pd
import os

basepath = os.path.dirname(os.path.abspath(__file__))
excel_path = os.path.join(basepath, "C题", "附件.xlsx")


xls = pd.ExcelFile(excel_path)
print("工作表:", xls.sheet_names)
# 2) 读取整本工作簿：返回 {sheet_name: DataFrame}
all_sheets = pd.read_excel(
    excel_path,
    sheet_name=None,          # None 表示读取所有表
    engine="openpyxl",
    # keep_default_na=False,  # 空单元不变成NaN，留空
    # dtype=str,              # 全部按文本读，避免类型混杂
)
for name, df in all_sheets.items():
    df.to_csv(f"{name}.csv", index=False, encoding="utf-8-sig")
