from pathlib import Path


def plot_from_static(image_filename: str = 'planned.png'):
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    img = mpimg.imread(Path(__file__).parents[3] / 'docs/source/_static' / image_filename)
    plt.figure()
    plt.imshow(img)
    plt.axis('off')
    plt.show()
