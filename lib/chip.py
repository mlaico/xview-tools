"""
"""
import numpy as np

"""
xView image processing helper functions.
"""


def chip_image(img, coords, classes, fixsz=512, minsz=-1):
    """
    Chip an image and get relative coordinates and classes.  Bounding boxes that pass into
        multiple chips are clipped: each portion that is in a chip is labeled. For example,
        half a building will be labeled if it is cut off in a chip. If there are no boxes,
        the boxes array will be [[0,0,0,0]] and classes [0].
        Note: This chip_image method is only tested on xView data-- there are some image manipulations that can mess up different images.
    Args:
        img: the image to be chipped in array format
        coords: an (N,4) array of bounding box coordinates for that image
        classes: an (N,1) array of classes for each bounding box
        shape: an (W,H) tuple indicating width and height of chips
    Output:
        An image array of shape (M,W,H,C), where M is the number of chips,
        W and H are the dimensions of the image, and C is the number of color
        channels.  Also returns boxes and classes dictionaries for each corresponding chip.
    """
    # deal w/ greyscale images
    if len(img.shape) == 2:
        img = np.stack((img,)*3, axis=-1)

    height,width,_ = img.shape

    # w_num,h_num = (int(width/minsz),int(height/minsz)) if minsz else (int(width/fixsz),int(height/fixsz))
    # w_chip,h_chip = (width//w_num,height//h_num) if minsz else (fixsz, fixsz)

    w_num,h_num = (int(width/fixsz),int(height/fixsz))
    w_chip,h_chip = (fixsz, fixsz)

    num_chips = w_num*h_num

    # print([w_num, h_num, w_chip, h_chip, num_chips, fixsz])
    chips = np.zeros((num_chips,h_chip,w_chip,3))
    total_boxes = {}
    total_classes = {}

    k = 0
    for i in range(w_num):
        for j in range(h_num):
            x = np.logical_or(
                np.logical_and((coords[:,0]<((i+1)*w_chip)),(coords[:,0]>(i*w_chip))),
                np.logical_and((coords[:,2]<((i+1)*w_chip)),(coords[:,2]>(i*w_chip)))
            )
            out = coords[x]
            y = np.logical_or(
                np.logical_and((out[:,1]<((j+1)*h_chip)),(out[:,1]>(j*h_chip))),
                np.logical_and((out[:,3]<((j+1)*h_chip)),(out[:,3]>(j*h_chip)))
            )
            outn = out[y]
            out = np.transpose(np.vstack((
                np.clip(outn[:,0]-(w_chip*i),0,w_chip),
                np.clip(outn[:,1]-(h_chip*j),0,h_chip),
                np.clip(outn[:,2]-(w_chip*i),0,w_chip),
                np.clip(outn[:,3]-(h_chip*j),0,h_chip)
            )))
            box_classes = classes[x][y]

            if out.shape[0] != 0:
                total_boxes[k] = out
                total_classes[k] = box_classes
            else:
                total_boxes[k] = np.array([[0,0,0,0]])
                total_classes[k] = np.array([0])

            chip = img[h_chip*j:h_chip*(j+1),w_chip*i:w_chip*(i+1),:3]
            chips[k]=chip

            k = k + 1

    return chips.astype(np.uint8),w_chip,h_chip,total_boxes,total_classes


def chip_image_w_overlap(
    img,
    coords,
    classes,
    fixsz=512,
    overlap=40,
    small_box_min=12,
    med_box_min=24,
    large_box_min=48
):
    """
    Chip an image and get relative coordinates and classes.  Bounding boxes that pass into
        multiple chips are clipped: each portion that is in a chip is labeled. For example,
        half a building will be labeled if it is cut off in a chip. If there are no boxes,
        the boxes array will be [[0,0,0,0]] and classes [0].
        Note: This chip_image method is only tested on xView data-- there are some image manipulations that can mess up different images.
    Args:
        img: the image to be chipped in array format
        coords: an (N,4) array of bounding box coordinates for that image
        classes: an (N,1) array of classes for each bounding box
        shape: an (W,H) tuple indicating width and height of chips
    Output:
        An image array of shape (M,W,H,C), where M is the number of chips,
        W and H are the dimensions of the image, and C is the number of color
        channels.  Also returns boxes and classes dictionaries for each corresponding chip.
    """
    # deal w/ greyscale images
    if len(img.shape) == 2:
        img = np.stack((img,)*3, axis=-1)

    height,width,_ = img.shape

    w_num,h_num = (int((width-overlap/2)/(fixsz-overlap)),int((height-overlap/2)/(fixsz-overlap)))
    w_chip,h_chip = (fixsz, fixsz)

    # check for remainder images
    rem_img_w = (w_num+1)*w_chip - width
    rem_img_h = (h_num+1)*h_chip - height

    if rem_img_w > 3*overlap:
        w_num += 1
    if rem_img_h > 3*overlap:
        h_num += 1

    num_chips = w_num*h_num

    # print([w_num, h_num, w_chip, h_chip, num_chips, fixsz])
    chips = np.zeros((num_chips,h_chip,w_chip,3))
    total_boxes = {}
    total_classes = {}

    k = 0
    for i in range(w_num):
        for j in range(h_num):
            # min/max chip coords
            x_min = i*(w_chip - overlap)
            x_max = min(i*(w_chip - overlap) + w_chip, width)
            y_min = j*(h_chip - overlap)
            y_max = min(j*(h_chip - overlap) + h_chip, height)

            buf = int(overlap/2)

            # take intersection of bboxes and chip boundaries

            # identify small objects

            small_objects = np.full((classes.shape[0]), False)
            med_objects = np.full((classes.shape[0]), False)
            large_objects = np.full((classes.shape[0]), False)
            small_object_ids = [1,2,4,5,6,7,8,9,10,11,12,13,14,15,17,21,23,24,25,26,27,34]
            med_object_ids = [3,16,19,20,22,32,33,35]
            large_object_ids = [18,28,29,30,31]

            # get boolean vals for small, medium, and large object categories
            for cls in small_object_ids:
                small_objects = np.logical_or(small_objects, (classes == cls))
            for cls in med_object_ids:
                med_objects = np.logical_or(med_objects, (classes == cls))
            for cls in large_object_ids:
                large_objects = np.logical_or(large_objects, (classes == cls))

            # Keep boxes within image boundaries that are buffered depending on category size
            # small objects
            buf = small_box_min
            small_x = np.logical_or(
                np.logical_and((coords[:,0] < x_max-buf), (coords[:,0] > x_min+buf)),
                np.logical_and((coords[:,2] < x_max-buf), (coords[:,2] > x_min+buf))
            )
            small_y = np.logical_or(
                np.logical_and((coords[:,1] < y_max-buf), (coords[:,1] > y_min+buf)),
                np.logical_and((coords[:,3] < y_max-buf), (coords[:,3] > y_min+buf))
            )
            # keep boxes that have at least 1 x- and 1 y-coord in img bounds
            in_small_bounds = np.logical_and(small_x, small_y)
            small_objects = np.logical_and(small_objects, in_small_bounds)

            # med objects
            buf = med_box_min
            med_x = np.logical_or(
                np.logical_and((coords[:,0] < x_max-buf), (coords[:,0] > x_min+buf)),
                np.logical_and((coords[:,2] < x_max-buf), (coords[:,2] > x_min+buf))
            )
            med_y = np.logical_or(
                np.logical_and((coords[:,1] < y_max-buf), (coords[:,1] > y_min+buf)),
                np.logical_and((coords[:,3] < y_max-buf), (coords[:,3] > y_min+buf))
            )
            in_med_bounds = np.logical_and(med_x, med_y)
            med_objects = np.logical_and(med_objects, in_med_bounds)

            # large objects
            buf = large_box_min
            large_x = np.logical_or(
                np.logical_and((coords[:,0] < x_max-buf), (coords[:,0] > x_min+buf)),
                np.logical_and((coords[:,2] < x_max-buf), (coords[:,2] > x_min+buf))
            )
            large_y = np.logical_or(
                np.logical_and((coords[:,1] < y_max-buf), (coords[:,1] > y_min+buf)),
                np.logical_and((coords[:,3] < y_max-buf), (coords[:,3] > y_min+buf))
            )
            in_large_bounds = np.logical_and(large_x, large_y)
            large_objects = np.logical_and(large_objects, in_large_bounds)

            all_objects = np.logical_or(
                small_objects, np.logical_or(med_objects, large_objects)
            )

            outn = coords[all_objects]

            out = np.transpose(np.vstack((
                np.clip(outn[:,0] - x_min, 0, x_max-x_min),
                np.clip(outn[:,1] - y_min, 0, y_max-y_min),
                np.clip(outn[:,2] - x_min, 0, x_max-x_min),
                np.clip(outn[:,3] - y_min, 0, y_max-y_min)
            )))
            box_classes = classes[all_objects]

            if out.shape[0] != 0:
                total_boxes[k] = out
                total_classes[k] = box_classes
            else:
                total_boxes[k] = np.array([[0,0,0,0]])
                total_classes[k] = np.array([0])

            chip = img[int(y_min):int(y_max), int(x_min):int(x_max), :3]
            chips[k, 0:int(y_max-y_min), 0:int(x_max-x_min), :3] = chip

            k = k + 1

    return chips.astype(np.uint8), total_boxes, total_classes