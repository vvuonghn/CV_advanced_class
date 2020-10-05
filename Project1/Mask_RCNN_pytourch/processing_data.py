import os
import argparse
import shutil
import cv2
import numpy as np
import matplotlib.image as plt
from PIL import Image

def load_chair_attribute(filename):
    # [1, 3, 4]: Arrays of chair instances
    instance_ids = set()
    with open(filename) as f:
        for line in f.readlines():
            infos = line.split(' # ')
            instance_id = int(infos[0].strip())
            category = infos[4].strip()
            if category == "chair":
                instance_ids.add(instance_id)
    return list(instance_ids)

def convert_segmentation(seg_filename, instance_ids):
    seg_img = cv2.imread(seg_filename)
    try:
        unique_instances = np.unique(seg_img[:,:,0])
    except Exception as e:
            print("Error", e)
            return None
    seg_out = np.zeros((seg_img.shape[0], seg_img.shape[1]), dtype=np.uint8)
    for idx, instance_id in enumerate(instance_ids):
        try:
            seg_out[seg_img[:, :, 0] == unique_instances[instance_id]] = idx + 1
        except Exception as e:
            print("Error", e)
    return seg_out

# def main(data_path, out):
#     count = 0
#     for subdir1 in os.listdir(data_path):
#         input_dir1 = os.path.join(data_path, subdir1)
#         if not os.path.isdir(input_dir1): continue
#         for subdir2 in os.listdir(input_dir1):
#             input_dir2 = os.path.join(input_dir1, subdir2)
#             if not os.path.isdir(input_dir2): continue
#             for img_name in os.listdir(input_dir2):
#                 if img_name.startswith(".") or not img_name.endswith(".jpg"): continue
                
#                 # Make output directory
#                 out_dir = os.path.join(out, subdir1, subdir2)
#                 os.makedirs(out_dir, exist_ok=True)
#                 # Process image
#                 ## Copy output
                
#                 ## Process segmentation and save
#                 print("os.path.join(out_dir, img_name) ",os.path.join(out_dir, img_name))

#                 instance_ids = load_chair_attribute(os.path.join(input_dir2, img_name.replace(".jpg", "_atr.txt")))
#                 if len(instance_ids) > -1:
#                     count +=1
#                     print("count ",count)
#                 #     shutil.copy(os.path.join(input_dir2, img_name), os.path.join(out_dir, img_name))
#                 #     converted_seg = convert_segmentation(os.path.join(input_dir2, img_name.replace(".jpg", "_seg.png")), instance_ids)
#                 #     cv2.imwrite(os.path.join(out_dir, img_name.replace(".jpg", "_seg.png")), converted_seg)
#                 #     print("Converted {} with {} chairs".format(img_name, len(instance_ids)))




def main(data_path, out):
    
    out_dir_image = os.path.join(out, "Chair_ADE20K_IMG")
    out_dir_mask = os.path.join(out, "Chair_ADE20K_MASK")
    out_dir_mask_vs = os.path.join(out, "Chair_ADE20K_MASK_VS")
    os.makedirs(out_dir_image, exist_ok=True)
    os.makedirs(out_dir_mask, exist_ok=True)
    os.makedirs(out_dir_mask_vs, exist_ok=True)
    count = 0

    for subdir1 in os.listdir(data_path):
        input_dir1 = os.path.join(data_path, subdir1)
        if not os.path.isdir(input_dir1): continue

        for subdir2 in os.listdir(input_dir1):
            input_dir2 = os.path.join(input_dir1, subdir2)
            if not os.path.isdir(input_dir2):
                if subdir2.startswith(".") or not subdir2.endswith(".jpg"): continue
                
                # Make output directory
                # out_dir = os.path.join(out, subdir1, subdir2)
                # os.makedirs(out_dir, exist_ok=True)
                # Process image
                ## Copy output
                
                ## Process segmentation and save
                # print("os.path.join(out_dir, img_name) ",os.path.join(out_dir, img_name))

                instance_ids = load_chair_attribute(os.path.join(input_dir1, subdir2.replace(".jpg", "_atr.txt")))
                if len(instance_ids) > -1:
                    count +=1
                    converted_seg = convert_segmentation(os.path.join(input_dir2, subdir2.replace(".jpg", "_seg.png")), instance_ids)
                    if converted_seg == None:
                        continue
                    shutil.copy(os.path.join(input_dir1, subdir2), os.path.join(out_dir_image, subdir2))
                    
                    cv2.imwrite(os.path.join(out_dir_mask, subdir2.replace(".jpg", "_seg.png")), converted_seg)
                    plt.imsave(os.path.join(out_dir_mask_vs, subdir2.replace(".jpg", "_seg.png")), converted_seg)
                    print(count, " Converted {} with {} chairs".format(subdir2, len(instance_ids)))

                continue
            for subdir3 in os.listdir(input_dir2):
                input_dir3 = os.path.join(input_dir2, subdir3)
                if not os.path.isdir(input_dir3):
                    if subdir3.startswith(".") or not subdir3.endswith(".jpg"): continue
                    
                    # Make output directory
                    # out_dir = os.path.join(out, subdir1, subdir2)
                    # os.makedirs(out_dir, exist_ok=True)
                    # Process image
                    ## Copy output
                    
                    ## Process segmentation and save
                    # print("os.path.join(out_dir, img_name) ",os.path.join(out_dir, img_name))

                    instance_ids = load_chair_attribute(os.path.join(input_dir2, subdir3.replace(".jpg", "_atr.txt")))
                    if len(instance_ids) > -1:
                        count +=1
 
                        shutil.copy(os.path.join(input_dir2, subdir3), os.path.join(out_dir_image, subdir3))
                        converted_seg = convert_segmentation(os.path.join(input_dir2, subdir3.replace(".jpg", "_seg.png")), instance_ids)
                        cv2.imwrite(os.path.join(out_dir_mask, subdir3.replace(".jpg", "_seg.png")), converted_seg)
                        plt.imsave(os.path.join(out_dir_mask_vs, subdir3.replace(".jpg", "_seg.png")), converted_seg)
                        print(count, " Converted {} with {} chairs".format(subdir3, len(instance_ids)))
                    continue
                for subdir4 in os.listdir(input_dir3):
                    input_dir4 = os.path.join(input_dir3, subdir4)
                    if not os.path.isdir(input_dir4):
                        if subdir4.startswith(".") or not subdir4.endswith(".jpg"): continue
                        
                        # Make output directory
                        # out_dir = os.path.join(out, subdir1, subdir2)
                        # os.makedirs(out_dir, exist_ok=True)
                        # Process image
                        ## Copy output
                        
                        ## Process segmentation and save
                        # print("os.path.join(out_dir, img_name) ",os.path.join(out_dir, img_name))

                        instance_ids = load_chair_attribute(os.path.join(input_dir3, subdir4.replace(".jpg", "_atr.txt")))
                        if len(instance_ids) > -1:
                            count +=1
                            shutil.copy(os.path.join(input_dir3, subdir4), os.path.join(out_dir_image, subdir4))
                            converted_seg = convert_segmentation(os.path.join(input_dir3, subdir4.replace(".jpg", "_seg.png")), instance_ids)
                            cv2.imwrite(os.path.join(out_dir_mask, subdir4.replace(".jpg", "_seg.png")), converted_seg)
                            plt.imsave(os.path.join(out_dir_mask_vs, subdir4.replace(".jpg", "_seg.png")), converted_seg)
                            print(count, " Converted {} with {} chairs".format(subdir4, len(instance_ids)))
                        continue
                    for subdir5 in os.listdir(input_dir4):
                        input_dir5 = os.path.join(input_dir4, subdir5)
                        if not os.path.isdir(input_dir5):
                            if subdir5.startswith(".") or not subdir5.endswith(".jpg"): continue
                            
                            # Make output directory
                            # out_dir = os.path.join(out, subdir1, subdir2)
                            # os.makedirs(out_dir, exist_ok=True)
                            # Process image
                            ## Copy output
                            
                            ## Process segmentation and save
                            # print("os.path.join(out_dir, img_name) ",os.path.join(out_dir, img_name))

                            instance_ids = load_chair_attribute(os.path.join(input_dir4, subdir5.replace(".jpg", "_atr.txt")))
                            if len(instance_ids) > -1:
                                count +=1
                                shutil.copy(os.path.join(input_dir4, subdir5), os.path.join(out_dir_image, subdir5))
                                converted_seg = convert_segmentation(os.path.join(input_dir4, subdir5.replace(".jpg", "_seg.png")), instance_ids)
                                cv2.imwrite(os.path.join(out_dir_mask, subdir5.replace(".jpg", "_seg.png")), converted_seg)
                                plt.imsave(os.path.join(out_dir_mask_vs, subdir5.replace(".jpg", "_seg.png")), converted_seg)
                                print(count, " Converted {} with {} chairs".format(subdir5, len(instance_ids)))
                            continue
                        for subdir6 in os.listdir(input_dir5):
                            input_dir6 = os.path.join(input_dir5, subdir6)
                            if not os.path.isdir(input_dir6):
                                if subdir6.startswith(".") or not subdir6.endswith(".jpg"): continue
                                
                                # Make output directory
                                # out_dir = os.path.join(out, subdir1, subdir2)
                                # os.makedirs(out_dir, exist_ok=True)
                                # Process image
                                ## Copy output
                                
                                ## Process segmentation and save
                                # print("os.path.join(out_dir, img_name) ",os.path.join(out_dir, img_name))

                                instance_ids = load_chair_attribute(os.path.join(input_dir5, subdir6.replace(".jpg", "_atr.txt")))
                                if len(instance_ids) > -1:
                                    count +=1
                                    shutil.copy(os.path.join(input_dir5, subdir6), os.path.join(out_dir_image, subdir6))
                                    converted_seg = convert_segmentation(os.path.join(input_dir5, subdir6.replace(".jpg", "_seg.png")), instance_ids)
                                    cv2.imwrite(os.path.join(out_dir_mask, subdir6.replace(".jpg", "_seg.png")), converted_seg)
                                    plt.imsave(os.path.join(out_dir_mask_vs, subdir6.replace(".jpg", "_seg.png")), converted_seg)
                                    print(count, " Converted {} with {} chairs".format(subdir6, len(instance_ids)))
                                continue
                            for subdir7 in os.listdir(input_dir6):
                                input_dir7 = os.path.join(input_dir6, subdir7)
                                if not os.path.isdir(input_dir6):
                                    if subdir7.startswith(".") or not subdir7.endswith(".jpg"): continue
                                    
                                    # Make output directory
                                    # out_dir = os.path.join(out, subdir1, subdir2)
                                    # os.makedirs(out_dir, exist_ok=True)
                                    # Process image
                                    ## Copy output
                                    
                                    ## Process segmentation and save
                                    # print("os.path.join(out_dir, img_name) ",os.path.join(out_dir, img_name))

                                    instance_ids = load_chair_attribute(os.path.join(input_dir6, subdir7.replace(".jpg", "_atr.txt")))
                                    if len(instance_ids) > -1:
                                        count +=1
                                        shutil.copy(os.path.join(input_dir6, subdir7), os.path.join(out_dir_image, subdir6))
                                        converted_seg = convert_segmentation(os.path.join(input_dir6, subdir7.replace(".jpg", "_seg.png")), instance_ids)
                                        cv2.imwrite(os.path.join(out_dir_mask, subdir7.replace(".jpg", "_seg.png")), converted_seg)
                                        plt.imsave(os.path.join(out_dir_mask_vs, subdir7.replace(".jpg", "_seg.png")), converted_seg)
                                        print(count, " Converted {} with {} chairs".format(subdir7, len(instance_ids)))
                                    continue
                
            # for img_name in os.listdir(input_dir2):
            #     if img_name.startswith(".") or not img_name.endswith(".jpg"): continue
                
            #     # Make output directory
            #     out_dir = os.path.join(out, subdir1, subdir2)
            #     os.makedirs(out_dir, exist_ok=True)
            #     # Process image
            #     ## Copy output
                
            #     ## Process segmentation and save
            #     print("os.path.join(out_dir, img_name) ",os.path.join(out_dir, img_name))

            #     instance_ids = load_chair_attribute(os.path.join(input_dir2, img_name.replace(".jpg", "_atr.txt")))
            #     if len(instance_ids) > -1:
            #         count +=1
            #         print("count ",count)
                #     shutil.copy(os.path.join(input_dir2, img_name), os.path.join(out_dir, img_name))
                #     converted_seg = convert_segmentation(os.path.join(input_dir2, img_name.replace(".jpg", "_seg.png")), instance_ids)
                #     cv2.imwrite(os.path.join(out_dir, img_name.replace(".jpg", "_seg.png")), converted_seg)
                #     print("Converted {} with {} chairs".format(img_name, len(instance_ids)))

if __name__ == "__main__":
    data_path = "/vinai/vuonghn/Research/CV_courses/Dataset/ADE20K_2016_07_26/images/validation/"
    out_path = "/vinai/vuonghn/Research/CV_courses/Dataset/Chair_ADE20K/val/"
    main(data_path,out_path)
