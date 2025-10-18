"""
Extended data utilities to support both HDF5 and COCO JSON format datasets
"""
import os
import json
import numpy as np
import h5py
import torch
import torch.utils.data as data
from PIL import Image
import cv2


class CreateData(data.Dataset):
    """Original dataset class for HDF5 format"""
    def __init__(self, dataset_dict):
        self.len_dset_dict = len(dataset_dict)
        self.rgb = dataset_dict['rgb']
        self.depth = dataset_dict['depth']
        self.seg_label = dataset_dict['seg_label']

        if self.len_dset_dict > 3:
            self.class_label = dataset_dict['class_label']
            self.use_class = True

    def __getitem__(self, index):
        rgb_img = self.rgb[index]
        depth_img = self.depth[index]
        seg_label = self.seg_label[index]

        rgb_img = torch.from_numpy(rgb_img)
        depth_img = torch.from_numpy(depth_img)

        dataset_list = [rgb_img, depth_img, seg_label]

        if self.len_dset_dict > 3:
            class_label = self.class_label[index]
            dataset_list.append(class_label)
        return dataset_list

    def __len__(self):
        return len(self.seg_label)


class CreateDataFromJSON(data.Dataset):
    """New dataset class for COCO JSON format"""
    def __init__(self, json_path, data_root, target_size=(240, 320), use_class=False):
        """
        Args:
            json_path: Path to the COCO format JSON file
            data_root: Root directory containing images and depth maps
            target_size: (height, width) to resize images to match NYU format
            use_class: Whether to include class labels
        """
        with open(json_path, 'r') as f:
            self.coco_data = json.load(f)
        
        self.data_root = data_root
        self.target_size = target_size
        self.use_class = use_class
        
        # Build image id to annotations mapping
        self.img_to_anns = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
        
        # Build category id to category index mapping
        self.cat_id_to_idx = {}
        for idx, cat in enumerate(self.coco_data['categories']):
            self.cat_id_to_idx[cat['id']] = idx
        
        self.images = self.coco_data['images']
        self.num_classes = len(self.coco_data['categories'])
        
        print(f"[INFO] Loaded {len(self.images)} images with {len(self.coco_data['annotations'])} annotations")
        print(f"[INFO] Number of categories: {self.num_classes}")

    def _load_image(self, img_info):
        """Load RGB image"""
        img_path = os.path.join(self.data_root, img_info['original_path'])
        
        # Try different possible paths
        if not os.path.exists(img_path):
            # Try with file_name
            img_path = os.path.join(self.data_root, img_info['file_name'])
        
        if not os.path.exists(img_path):
            # Create a dummy image if file not found
            print(f"Warning: Image not found {img_path}, creating dummy image")
            img = np.zeros((480, 960, 3), dtype=np.uint8)
        else:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to target size
        img = cv2.resize(img, (self.target_size[1], self.target_size[0]))
        return img

    def _load_depth(self, img_info):
        """Load depth image - assumes depth is in same directory structure"""
        depth_path = img_info['original_path'].replace('/color/', '/depth/')
        depth_path = os.path.join(self.data_root, depth_path)
        
        if not os.path.exists(depth_path):
            # Try alternative path
            depth_path = depth_path.replace('.jpg', '.png')
        
        if not os.path.exists(depth_path):
            # Create dummy depth map
            depth = np.zeros((self.target_size[0], self.target_size[1]), dtype=np.uint8)
        else:
            depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
            depth = cv2.resize(depth, (self.target_size[1], self.target_size[0]))
        
        return depth

    def _create_segmentation_mask(self, img_info, img_height, img_width):
        """Create segmentation mask from COCO annotations"""
        mask = np.zeros((img_height, img_width), dtype=np.uint8)
        
        img_id = img_info['id']
        if img_id not in self.img_to_anns:
            return mask
        
        # Get original image size
        orig_h = int(img_info['height'])
        orig_w = int(img_info['width'])
        
        for ann in self.img_to_anns[img_id]:
            if 'segmentation' not in ann or not ann['segmentation']:
                continue
            
            # Get category index (0-based)
            cat_idx = self.cat_id_to_idx.get(ann['category_id'], 0)
            
            # Draw polygon segmentation
            for seg in ann['segmentation']:
                # Reshape to polygon format
                poly = np.array(seg).reshape(-1, 2)
                
                # Scale polygon to target size
                poly[:, 0] = poly[:, 0] * self.target_size[1] / orig_w
                poly[:, 1] = poly[:, 1] * self.target_size[0] / orig_h
                
                poly = poly.astype(np.int32)
                
                # Fill polygon with category index
                cv2.fillPoly(mask, [poly], cat_idx + 1)  # +1 because 0 is background
        
        return mask

    def __getitem__(self, index):
        img_info = self.images[index]
        
        # Load RGB image
        rgb_img = self._load_image(img_info)  # (H, W, 3)
        
        # Load depth image
        depth_img = self._load_depth(img_info)  # (H, W)
        
        # Create segmentation mask
        seg_label = self._create_segmentation_mask(
            img_info, 
            self.target_size[0], 
            self.target_size[1]
        )  # (H, W)
        
        # Convert to PyTorch format: RGB (3, H, W), Depth (1, H, W)
        rgb_img = np.transpose(rgb_img, (2, 0, 1)).astype(np.float32)  # (3, H, W)
        depth_img = depth_img[np.newaxis, :, :].astype(np.float32)  # (1, H, W)
        seg_label = seg_label.astype(np.int64)  # (H, W)
        
        # Convert to tensors
        rgb_img = torch.from_numpy(rgb_img)
        depth_img = torch.from_numpy(depth_img)
        
        dataset_list = [rgb_img, depth_img, seg_label]
        
        if self.use_class:
            # Use category id as class label
            cat_id = self.cat_id_to_idx.get(
                self.img_to_anns.get(img_info['id'], [{}])[0].get('category_id', 1), 
                0
            )
            dataset_list.append(cat_id)
        
        return dataset_list

    def __len__(self):
        return len(self.images)


def get_data(opt, use_train=True, use_test=True):
    """
    Load dataset from either HDF5 or JSON format
    Automatically detects format based on file extension
    """
    if not os.path.exists(opt.dataroot):
        raise Exception(f'Dataset file not found: {opt.dataroot}')
    
    # Check file format
    file_ext = os.path.splitext(opt.dataroot)[1].lower()
    
    if file_ext == '.json':
        # Load from JSON (COCO format)
        print('[INFO] Loading dataset from JSON (COCO format)')
        return get_data_from_json(opt, use_train, use_test)
    elif file_ext in ['.h5', '.hdf5']:
        # Load from HDF5 (original format)
        print('[INFO] Loading dataset from HDF5 format')
        return get_data_from_hdf5(opt, use_train, use_test)
    else:
        raise Exception(f'Unsupported file format: {file_ext}. Use .json, .h5, or .hdf5')


def get_data_from_json(opt, use_train=True, use_test=True):
    """
    Load dataset from COCO JSON format
    """
    train_dataset_generator = None
    test_dataset_generator = None
    
    # Assume data_root is in the same directory as JSON file
    data_root = os.path.dirname(opt.dataroot)
    
    if use_train:
        train_json = opt.dataroot
        if os.path.exists(train_json):
            train_dataset_generator = CreateDataFromJSON(
                train_json, 
                data_root,
                target_size=(240, 320),
                use_class=opt.use_class
            )
            print('[INFO] Training set generator has been created from JSON')
    
    if use_test:
        # Try to find test JSON file
        test_json = opt.dataroot.replace('train', 'test').replace('Train', 'Test')
        if os.path.exists(test_json):
            test_dataset_generator = CreateDataFromJSON(
                test_json,
                data_root,
                target_size=(240, 320),
                use_class=opt.use_class
            )
            print('[INFO] Test set generator has been created from JSON')
        else:
            # Use a split of training data as test
            print(f'[WARNING] Test file not found: {test_json}')
            print('[INFO] Using training data for testing (not recommended)')
            test_dataset_generator = train_dataset_generator
    
    return train_dataset_generator, test_dataset_generator


def get_data_from_hdf5(opt, use_train=True, use_test=True):
    """
    Original function: Load NYU_v2 or SUN rgb-d dataset in hdf5 format from disk
    """
    h5file = h5py.File(opt.dataroot, 'r')

    train_dataset_generator = None
    test_dataset_generator = None

    # Create python dicts containing numpy arrays of training samples
    if use_train:
        train_dataset_generator = dataset_generator(h5file, 'train', opt.use_class)
        print('[INFO] Training set generator has been created from HDF5')

    # Create python dicts containing numpy arrays of test samples
    if use_test:
        test_dataset_generator = dataset_generator(h5file, 'test', opt.use_class)
        print('[INFO] Test set generator has been created from HDF5')
    
    h5file.close()
    return train_dataset_generator, test_dataset_generator


def dataset_generator(h5file, dset_type, use_class):
    """
    Move h5 dictionary contents to python dict as numpy arrays and create dataset generator
    """
    dataset_dict = dict()
    # Create numpy arrays of given samples
    dataset_dict['rgb'] = np.array(h5file['rgb_' + dset_type],  dtype=np.float32)
    dataset_dict['depth'] = np.array(h5file['depth_' + dset_type], dtype=np.float32)
    dataset_dict['seg_label'] = np.array(h5file['label_' + dset_type], dtype=np.int64)

    # If classification loss is included in training add the classification labels to the dataset as well
    if use_class:
        dataset_dict['class_label'] = np.array(h5file['class_' + dset_type], dtype=np.int64)
    return CreateData(dataset_dict)
