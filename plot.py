import torch
import matplotlib.pyplot as plt

def plot(dataset, labels_map=None):
  figure = plt.figure(figsize=(8, 8))
  cols, rows = 3, 3
  for i in range(1, cols * rows + 1):
      sample_idx = torch.randint(len(dataset), size=(1,)).item()
      img, label = dataset[sample_idx]
      figure.add_subplot(rows, cols, i)
      plt.title(labels_map.get(label, label) if labels_map else label)
      plt.axis("off")
      plt.imshow(img.squeeze(), cmap="gray")
  plt.show()