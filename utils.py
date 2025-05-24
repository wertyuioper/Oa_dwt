import matplotlib.pyplot as plt

def show_image(image, title="Image"):
    plt.figure()
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()
