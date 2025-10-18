"""
Script to inspect and visualize the contents of NYU HDF5 dataset
Saves the output information and sample visualizations to datasets/outputs/
"""
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def inspect_h5_file(h5_path, output_dir='datasets/outputs'):
    """
    Inspect HDF5 file and save information and visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open HDF5 file
    print(f"Opening HDF5 file: {h5_path}")
    h5file = h5py.File(h5_path, 'r')
    
    # Create output text file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    info_file = os.path.join(output_dir, f'dataset_info_{timestamp}.txt')
    
    with open(info_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("HDF5 Dataset Inspection Report\n")
        f.write("=" * 80 + "\n")
        f.write(f"File: {h5_path}\n")
        f.write(f"Inspection Time: {datetime.now()}\n")
        f.write("=" * 80 + "\n\n")
        
        # List all keys in the HDF5 file
        f.write("Available Keys in HDF5 file:\n")
        f.write("-" * 80 + "\n")
        for key in h5file.keys():
            f.write(f"  - {key}\n")
        f.write("\n")
        
        # Detailed information for each dataset
        f.write("Detailed Dataset Information:\n")
        f.write("=" * 80 + "\n")
        
        for key in h5file.keys():
            dataset = h5file[key]
            f.write(f"\nDataset: {key}\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Shape: {dataset.shape}\n")
            f.write(f"  Dtype: {dataset.dtype}\n")
            
            # Show first 5 elements for each dataset
            f.write(f"\n  First 5 elements:\n")
            num_to_show = min(5, len(dataset))
            for i in range(num_to_show):
                if len(dataset.shape) == 1:
                    # 1D array (like class labels)
                    f.write(f"    [{i}]: {dataset[i]}\n")
                elif len(dataset.shape) == 3:
                    # 3D array (like segmentation labels)
                    f.write(f"    [{i}] shape: {dataset[i].shape}, "
                           f"min: {dataset[i].min()}, max: {dataset[i].max()}, "
                           f"mean: {dataset[i].mean():.2f}, unique_values: {len(np.unique(dataset[i]))}\n")
                elif len(dataset.shape) == 4:
                    # 4D array (like RGB or depth images)
                    f.write(f"    [{i}] shape: {dataset[i].shape}, "
                           f"min: {dataset[i].min()}, max: {dataset[i].max()}, "
                           f"mean: {dataset[i].mean():.2f}\n")
            f.write("\n")
            f.write(f"  Size: {dataset.size:,} elements\n")
            
            # Calculate memory size
            item_size = dataset.dtype.itemsize
            total_size = dataset.size * item_size
            if total_size < 1024:
                size_str = f"{total_size} bytes"
            elif total_size < 1024**2:
                size_str = f"{total_size/1024:.2f} KB"
            elif total_size < 1024**3:
                size_str = f"{total_size/(1024**2):.2f} MB"
            else:
                size_str = f"{total_size/(1024**3):.2f} GB"
            f.write(f"  Memory Size: {size_str}\n")
            
            # Statistics for numerical data
            if np.issubdtype(dataset.dtype, np.number):
                # Load a sample for statistics (avoid loading huge arrays)
                if dataset.size > 1000000:
                    sample = dataset[:min(100, len(dataset))].flatten()
                    f.write(f"  Min (sample): {np.min(sample):.4f}\n")
                    f.write(f"  Max (sample): {np.max(sample):.4f}\n")
                    f.write(f"  Mean (sample): {np.mean(sample):.4f}\n")
                    f.write(f"  Std (sample): {np.std(sample):.4f}\n")
                else:
                    data = dataset[:]
                    f.write(f"  Min: {np.min(data):.4f}\n")
                    f.write(f"  Max: {np.max(data):.4f}\n")
                    f.write(f"  Mean: {np.mean(data):.4f}\n")
                    f.write(f"  Std: {np.std(data):.4f}\n")
            
            f.write("\n")
        
        # Summary
        f.write("=" * 80 + "\n")
        f.write("Summary:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total number of datasets: {len(h5file.keys())}\n")
        
        # Identify train/test splits
        train_keys = [k for k in h5file.keys() if 'train' in k.lower()]
        test_keys = [k for k in h5file.keys() if 'test' in k.lower()]
        
        f.write(f"Training datasets: {len(train_keys)}\n")
        for k in train_keys:
            f.write(f"  - {k}: {h5file[k].shape}\n")
        
        f.write(f"\nTest datasets: {len(test_keys)}\n")
        for k in test_keys:
            f.write(f"  - {k}: {h5file[k].shape}\n")
    
    print(f"Dataset information saved to: {info_file}")
    
    # Visualize sample data
    visualize_samples(h5file, output_dir, timestamp)
    
    h5file.close()
    print(f"\nInspection complete! Check {output_dir}/ for results")


def visualize_samples(h5file, output_dir, timestamp, num_samples=5):
    """
    Visualize sample RGB, Depth, and Segmentation images separately
    """
    print("\nGenerating sample visualizations (separate images)...")
    
    # Get available datasets
    available_keys = list(h5file.keys())
    
    # Try to find RGB, Depth, and Label data
    rgb_train_key = None
    depth_train_key = None
    label_train_key = None
    
    for key in available_keys:
        if 'rgb' in key.lower() and 'train' in key.lower():
            rgb_train_key = key
        elif 'depth' in key.lower() and 'train' in key.lower():
            depth_train_key = key
        elif 'label' in key.lower() and 'train' in key.lower():
            label_train_key = key
    
    if not all([rgb_train_key, depth_train_key, label_train_key]):
        print("Warning: Could not find all required datasets for visualization")
        print(f"Found - RGB: {rgb_train_key}, Depth: {depth_train_key}, Label: {label_train_key}")
        return
    
    # Load datasets
    rgb_data = h5file[rgb_train_key]
    depth_data = h5file[depth_train_key]
    label_data = h5file[label_train_key]
    
    num_samples = min(num_samples, len(rgb_data))
    
    # Create separate visualizations for each sample - raw images only
    for i in range(num_samples):
        sample_num = i + 1
        
        # 1. Save RGB image separately (raw data)
        rgb_img = rgb_data[i]
        # Convert from CHW to HWC if needed
        if rgb_img.shape[0] == 3:
            rgb_img = np.transpose(rgb_img, (1, 2, 0))
        # Normalize to [0, 1] if needed
        if rgb_img.max() > 1.0:
            rgb_img = rgb_img / 255.0
        
        fig_rgb = plt.figure(frameon=False)
        fig_rgb.set_size_inches(rgb_img.shape[1]/100, rgb_img.shape[0]/100)
        ax = plt.Axes(fig_rgb, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig_rgb.add_axes(ax)
        ax.imshow(rgb_img, aspect='auto')
        output_path = os.path.join(output_dir, f'sample_{sample_num}_rgb_{timestamp}.png')
        plt.savefig(output_path, dpi=100)
        plt.close()
        print(f"  Saved: sample_{sample_num}_rgb_{timestamp}.png")
        
        # 2. Save Depth image separately (raw data)
        depth_img = depth_data[i]
        if len(depth_img.shape) == 3:
            depth_img = depth_img[0]  # Take first channel if multi-channel
        
        fig_depth = plt.figure(frameon=False)
        fig_depth.set_size_inches(depth_img.shape[1]/100, depth_img.shape[0]/100)
        ax = plt.Axes(fig_depth, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig_depth.add_axes(ax)
        ax.imshow(depth_img, cmap='gray', aspect='auto')
        output_path = os.path.join(output_dir, f'sample_{sample_num}_depth_{timestamp}.png')
        plt.savefig(output_path, dpi=100)
        plt.close()
        print(f"  Saved: sample_{sample_num}_depth_{timestamp}.png")
        
        # 3. Save Segmentation label separately (raw data)
        label_img = label_data[i]
        if len(label_img.shape) == 3:
            label_img = label_img[0]  # Take first channel if multi-channel
        
        fig_label = plt.figure(frameon=False)
        fig_label.set_size_inches(label_img.shape[1]/100, label_img.shape[0]/100)
        ax = plt.Axes(fig_label, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig_label.add_axes(ax)
        ax.imshow(label_img, cmap='tab20', aspect='auto')
        output_path = os.path.join(output_dir, f'sample_{sample_num}_label_{timestamp}.png')
        plt.savefig(output_path, dpi=100)
        plt.close()
        print(f"  Saved: sample_{sample_num}_label_{timestamp}.png")
        
        print()  # Empty line between samples
    
    # Create a summary figure showing statistics
    create_statistics_plot(h5file, output_dir, timestamp, label_train_key)


def create_statistics_plot(h5file, output_dir, timestamp, label_key):
    """
    Create plots showing dataset statistics
    """
    print("\nGenerating statistics plots...")
    
    label_data = h5file[label_key]
    
    # Calculate class distribution
    unique_classes, counts = np.unique(label_data[:], return_counts=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Class distribution bar plot
    axes[0].bar(unique_classes, counts)
    axes[0].set_xlabel('Class ID')
    axes[0].set_ylabel('Pixel Count')
    axes[0].set_title('Class Distribution in Training Set')
    axes[0].grid(True, alpha=0.3)
    
    # Class distribution pie chart (top classes)
    if len(unique_classes) > 10:
        # Show top 10 classes
        top_indices = np.argsort(counts)[-10:]
        top_classes = unique_classes[top_indices]
        top_counts = counts[top_indices]
        other_count = np.sum(counts) - np.sum(top_counts)
        
        labels = [f'Class {c}' for c in top_classes] + ['Others']
        sizes = list(top_counts) + [other_count]
    else:
        labels = [f'Class {c}' for c in unique_classes]
        sizes = counts
    
    axes[1].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    axes[1].set_title('Class Distribution (Percentage)')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'statistics_{timestamp}.png')
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved statistics plot: statistics_{timestamp}.png")


if __name__ == '__main__':
    # Path to the HDF5 dataset
    h5_path = './datasets/nyu_class_10_db.h5'
    output_dir = './datasets/outputs'
    
    # Run inspection
    inspect_h5_file(h5_path, output_dir)
