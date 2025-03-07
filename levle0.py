import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')

print("Dataset Shape:", df.shape)

print(df.head())

plt.figure(figsize=(10, 5))
for i in range(10):
    image = df.iloc[i, 1:].values.reshape(28, 28)
    label = df.iloc[i, 0]
    plt.subplot(2, 5, i + 1)
    plt.imshow(image, cmap='gray')
    plt.title(f'Label: {label}')
    plt.axis('off')
plt.tight_layout()
plt.show()

sample_image = df.iloc[0, 1:].values.reshape(28, 28)
print("Pixel Values Range:", sample_image.min(), "-", sample_image.max())