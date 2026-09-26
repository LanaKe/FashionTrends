import pandas as pd

# ==========================
# SETTINGS
# ==========================

file1 = "similarity_matrix22.csv"
file2 = "similarity_matrix.csv"

# Set to None if you just want row comparison
# Otherwise specify the unique identifier column (e.g. "id")
id_column = None

# ==========================
# LOAD FILES
# ==========================

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

print("=" * 60)
print("BASIC INFORMATION")
print("=" * 60)

print(f"File 1: {len(df1)} rows, {len(df1.columns)} columns")
print(f"File 2: {len(df2)} rows, {len(df2.columns)} columns")
print()

# ==========================
# COLUMN COMPARISON
# ==========================

cols1 = set(df1.columns)
cols2 = set(df2.columns)

print("=" * 60)
print("COLUMN COMPARISON")
print("=" * 60)

print("Columns only in File 1:")
print(cols1 - cols2)

print("\nColumns only in File 2:")
print(cols2 - cols1)

common_cols = list(cols1 & cols2)
print(f"\nShared columns ({len(common_cols)}):")
print(common_cols)

# ==========================
# ROW COMPARISON
# ==========================

print("\n" + "=" * 60)
print("ROW COMPARISON")
print("=" * 60)

if id_column is None:

    # Compare complete rows
    merged = df1.merge(df2, how="outer", indicator=True)

    only_file1 = merged[merged["_merge"] == "left_only"]
    only_file2 = merged[merged["_merge"] == "right_only"]

    print(f"Rows only in File 1: {len(only_file1)}")
    print(f"Rows only in File 2: {len(only_file2)}")

    if len(only_file1):
        print("\nExample rows only in File 1:")
        print(only_file1.head())

    if len(only_file2):
        print("\nExample rows only in File 2:")
        print(only_file2.head())

else:

    df1 = df1.set_index(id_column)
    df2 = df2.set_index(id_column)

    ids1 = set(df1.index)
    ids2 = set(df2.index)

    print(f"IDs only in File 1: {len(ids1 - ids2)}")
    print(f"IDs only in File 2: {len(ids2 - ids1)}")

    common_ids = ids1 & ids2
    print(f"Common IDs: {len(common_ids)}")

    common_columns = list(set(df1.columns) & set(df2.columns))

    differences = []

    for idx in common_ids:
        row1 = df1.loc[idx, common_columns]
        row2 = df2.loc[idx, common_columns]

        if not row1.equals(row2):
            diff_cols = []

            for col in common_columns:
                if pd.isna(row1[col]) and pd.isna(row2[col]):
                    continue
                if row1[col] != row2[col]:
                    diff_cols.append(col)

            differences.append({
                "ID": idx,
                "Different columns": diff_cols
            })

    print(f"\nRows with different values: {len(differences)}")

    if differences:
        print("\nFirst 10 differences:")
        for d in differences[:10]:
            print(d)

print("\nComparison complete.")