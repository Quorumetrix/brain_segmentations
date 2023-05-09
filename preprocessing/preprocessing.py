#!/usr/bin/env python

# preprocessing.py


def downsample_image(image, downsample_factor=4):
    downsampled_shape = tuple(s // downsample_factor for s in image.shape)
    return transform.resize(image, downsampled_shape, order=3, anti_aliasing=True)

def upscale_slice(image_slice, original_shape):
    return transform.resize(image_slice, original_shape, order=3, anti_aliasing=True)


def save_downsampled_png(img_ds, filepath):
    
    # Normalize the intensity values to the range [0, 255]
    normalized_img = (img_ds - img_ds.min()) / (img_ds.max() - img_ds.min())
    scaled_img = (normalized_img * 255).astype(np.uint8)

    # Save the image as a 24-bit PNG
    imageio.imwrite(filepath+'.png', scaled_img)


def downsample_folder(file_list):
    for identifier in file_list:
        if identifier.endswith('.tif'):
            print(identifier)
            neun_img, cfos_img = load(identifier)

            assert neun_img.shape == cfos_img.shape
            IMG_DIM = neun_img.shape # Temp, to config. Then this assert will make sense. 
            assert neun_img.shape == IMG_DIM
            assert cfos_img.shape == IMG_DIM

            neun_img_ds, cfos_img_ds =  downsample_image(neun_img, downsample_factor=DS_FACTOR),  downsample_image(cfos_img, downsample_factor=DS_FACTOR)

            save_downsampled_png(neun_img_ds, neun_output + identifier[:-4])
            save_downsampled_png(cfos_img_ds, cfos_output + identifier[:-4])

        
    