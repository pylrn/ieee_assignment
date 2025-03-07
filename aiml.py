import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data.csv')

print("Dataset Info:")
print(df.info())

print("\nSample Data:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum().sum())

print("\nSummary Statistics:")
print(df.describe())

plt.figure(figsize=(8, 4))
sns.countplot(x='label', data=df, palette='viridis')
plt.title('Distribution of Clothing Categories')
plt.xlabel('Category')
plt.ylabel('Count')
plt.show()

def display_sample_images(df, num_samples=5):
    plt.figure(figsize=(10, 10))
    for label in sorted(df['label'].unique()):
        samples = df[df['label'] == label].iloc[:num_samples, 1:].values
        for i, sample in enumerate(samples):
            image = sample.reshape(28, 28)
            plt.subplot(len(df['label'].unique()), num_samples, label * num_samples + i + 1)
            plt.imshow(image, cmap='gray')
            plt.title(f"Label: {label}")
            plt.axis('off')
    plt.tight_layout()
    plt.show()

display_sample_images(df)

plt.figure(figsize=(10, 8))
corr = df.corr().iloc[:20, :20]
sns.heatmap(corr, cmap='coolwarm', annot=False)
plt.title('Correlation Heatmap (Partial)')
plt.show()