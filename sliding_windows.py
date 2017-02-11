import numpy as np

def sliding_window_list(start:(int, int), end:(int, int), scale=1, pixels_per_step=(32, 32)):
    window_size = tuple(np.array((64, 64)) * scale)

    start_x, start_y = start
    end_x, end_y = end

    span_x = end_x - start_x
    span_y = end_y - start_y

    window_width, window_height = window_size
    pixels_per_step_x, pixels_per_step_y = pixels_per_step

    num_windows_x = np.int(((span_x - window_width) // pixels_per_step_x) + 1)
    num_windows_y = np.int(((span_y - window_height) // pixels_per_step_y) + 1)

    windows = []

    for window_x in range(num_windows_x):
        for window_y in range(num_windows_y):
            window_start_x = window_x * pixels_per_step_x + start_x
            window_start_y = window_y * pixels_per_step_y + start_y

            window_end_x = window_start_x + window_width
            window_end_y = window_start_y + window_height

            windows.append(((window_start_x, window_start_y), (window_end_x, window_end_y), scale))

    return windows

if __name__ == "__main__":
    from draw import draw_bounding_boxes
    from image import rgb_image
    import matplotlib.pyplot as plt

    filepath = "./data/test_images/test1.jpg"
    image = rgb_image(filepath)
    windows = sliding_window_list(image)
    image = draw_bounding_boxes(image, windows)

    plt.imshow(image)
    plt.show()
