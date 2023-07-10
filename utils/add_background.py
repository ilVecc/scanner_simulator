from pathlib import Path
import random as rnd
import cv2

foreground_images_path = Path(r"./output/images")
foreground_masks_path = Path(r"./output/annotations")
background_images_path = Path(r"./backgrounds")
# the image is considered AS-IS to be a portrait
desired_portrait_width = 752
background_height_same = True

bg_paths = list(background_images_path.iterdir())
rnd.shuffle(bg_paths)


for fg_path in foreground_images_path.iterdir():
    fg_mask_path = foreground_masks_path / fg_path.name
    
    fg_img = cv2.imread(str(fg_path), cv2.IMREAD_COLOR)
    fg_msk = cv2.imread(str(fg_mask_path), cv2.IMREAD_GRAYSCALE)
    
    assert fg_img.shape[:-1] == fg_msk.shape

    # resize fg as desired
    fg_h, fg_w, _ = fg_img.shape
    scale = desired_portrait_width / fg_w
    fg_h, fg_w = int(fg_h * scale), int(fg_w * scale)
    fg_img = cv2.resize(fg_img, (fg_w, fg_h))
    fg_msk = cv2.resize(fg_msk, (fg_w, fg_h))

    # sample bg
    bg_path = rnd.choice(bg_paths)
    bg_img = cv2.imread(str(bg_path), cv2.IMREAD_COLOR)
    bg_h, bg_w, _ = bg_img.shape

    print(fg_h, fg_w)
    print(bg_h, bg_w)
    
    # rescale the bg if needed
    if bg_h < fg_h:
        if background_height_same:
            print(f"Background image {bg_path.name} has been upscaled to match height")
            scale = fg_h / bg_h
            bg_h, bg_w = int(bg_h * scale), int(bg_w * scale)
            bg_img = cv2.resize(bg_img, (bg_w, bg_h))
        else:
            raise Exception(f"Cannot reshape background image {bg_path.name} of shape {bg_img.shape} to shape {fg_img.shape}")
    
    # crop the needed region from bg
    x = (bg_w - fg_w)/2
    y = (bg_h - fg_h)/2
    bg_img_cropped = bg_img[int(y):int(y+fg_h), int(x):int(x+fg_w)]

    fg_img_selection = fg_msk == 1
    bg_img_cropped[fg_img_selection, :] = fg_img[fg_img_selection, :]
    cv2.imshow("img", bg_img_cropped)
    cv2.waitKey(0)
