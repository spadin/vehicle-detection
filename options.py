image_options = {
    "color_space": "HSV" # Default: "RGB"
}

hog_options = {
    "channel_index": 1, # Default: "ALL"
    "orientations": 9, # Default: 9
    "pixels_per_cell": (16, 16), # Default: (2, 2)
    "cells_per_block": (4, 4) # Default: (2, 2)
}

color_histogram_options = {
    "bins": 16, # Default: 32
    "range": (0., 1.) # Default: (0., 1.)
}

spatial_binning_options = {
    "size": (16, 16) # Default: (32, 32)
}
