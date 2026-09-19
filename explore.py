import pandas as pd

# The file is tab-separated with no header row
df = pd.read_csv('spam.csv', sep='\t', header=None, names=['label', 'message'])

print("=== Shape ===")
print(df.shape)

print("\n=== Class balance ===")
print(df['label'].value_counts())
print(df['label'].value_counts(normalize=True) * 100)

print("\n=== Sample spam messages ===")
print(df[df['label'] == 'spam']['message'].head(5).to_string())

print("\n=== Sample ham messages ===")
print(df[df['label'] == 'ham']['message'].head(5).to_string())

print("\n=== Message length stats ===")
df['length'] = df['message'].apply(len)
print(df.groupby('label')['length'].describe())

print("\n=== Missing values ===")
print(df.isnull().sum())