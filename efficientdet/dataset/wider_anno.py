#Function for reading faces bounding box annotations for WIDER Face dataset
import os
import numpy as np




def widerface_annotations():
    """Function for reading faces bounding box annotations for WIDER Face dataset
    
    Parameters:
    ----------
    :param dict annotations: dictionary with annotations files path 
                            { "train" : WIDER_train_annotations_path,
                              "val"   : WIDER_val_annotations_path }

    :return annotation_dict: annotation dictionary

    Face annotations:
    ----------------
    file_name
    number_of_bounding_box
    bbox [x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose]

    Mappings between attribute names and label values:
    -------------------------------------------------
    blur:
      clear->0
      normal blur->1
      heavy blur->2

    expression:
      typical expression->0
      exaggerate expression->1

    illumination:
      normal illumination->0
      extreme illumination->1

    occlusion:
      no occlusion->0
      partial occlusion->1
      heavy occlusion->2

    pose:
      typical pose->0
      atypical pose->1

    invalid:
      false->0 (valid image)
      true->1 (invalid image)
    """

    annotations = dict()
    if os.path.exists('WIDER/WIDER_train'): annotations["train"] = 'WIDER/wider_face_split/wider_face_train_bbx_gt.txt' 
    if os.path.exists('WIDER/WIDER_val'): annotations["val"] = 'WIDER/wider_face_split/wider_face_val_bbx_gt.txt'
    if os.path.exists('WIDER/WIDER_test'): annotations["test"] = 'WIDER/wider_face_split/wider_face_test_filelist.txt'

    PATH_dict = { 
    'train'           : 'WIDER/WIDER_train/images/',      # path to the train directory
    'val'             : 'WIDER/WIDER_val/images/',        # path to the validation directory 
    'test'            : 'WIDER/WIDER_test/images/',
    }

    annotation_dict = dict()
    for key, value in annotations.items():
        with open(value, "r") as file_:
          rows = file_.readlines()

        idx = 0
        annotation_dict[key] = []
        while (idx < len(rows)):
            file_name = rows[idx].replace("\n", "")
            number_of_bounding_box = int(rows[idx+1]) if key !='test' else 0
            bbox = []
            poses = []
            '''
            Attention! there are photos without annotations..
            0--Parade/0_Parade_Parade_0_452.jpg
            0
            0 0 0 0 0 0 0 0 0 0 
            '''
            if key != 'test':
                jump = number_of_bounding_box if number_of_bounding_box != 0 else 1

                for i in range(1, jump+1):
                    row = rows[idx+1+i]
                    row = [int(item) for item in row.split(' ')[:10]]
                    box = np.array(row[:4]) #xmin, ymin, w, h
                    box[2:4] = box[2:4] + box[0:2] #xmin, ymin, xmax, ymax
                    bbox.append(list(box))
                    poses.append(str(row[9]))
           
            annotation_dict[key].append({
                'path'             : PATH_dict[key] + file_name,
                'number_of_bounding_box': number_of_bounding_box, 
                'bbox'                  : bbox if key != 'test' else [[0,0,0,0]],
                'poses'                 : poses if key != 'test' else ['0']
            })

            idx += (jump+2) if key != 'test' else 1
       
    return annotation_dict
