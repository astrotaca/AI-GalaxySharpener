import numpy as np
import tifffile
from astropy.io import fits
from PyQt5.QtGui import QImage
from PyQt5.QtCore import Qt
import cv2
import os

def has_imagecodecs():
    """Check if imagecodecs is available without triggering linter errors"""
    try:
        # Uses importlib.util.find_spec which doesn't trigger linter errors
        import importlib.util
        spec = importlib.util.find_spec("imagecodecs")
        return spec is not None
    except ImportError:
        return False

def load_image(image_path):
    """
    Load an image from file
    Supports FITS and TIFF (including LZW compression if imagecodecs is installed)
    Returns normalized [0,1] float32 array
    """
    _, ext = os.path.splitext(image_path)
    ext = ext.lower()
    
    if ext == '.fits' or ext == '.fit':
        
        with fits.open(image_path) as hdul:
            # Get data from the primary HDU
            data = hdul[0].data
            
            if data is None and len(hdul) > 1:
                data = hdul[1].data
            
            # Convert to float32 and ensure [0,1] range while preserving distribution
            if data is not None:
                if len(data.shape) == 2:
                    data = data.astype(np.float32)
                    if np.max(data) > 1.0:
                        data = data / np.max(data)
                    return data
                elif len(data.shape) == 3:
                    data = data.astype(np.float32)
                    if np.max(data) > 1.0:
                        data = data / np.max(data)
                    
                    if data.shape[0] == 3:
                        data = np.transpose(data, (1, 2, 0))
                    return data
                else:
                    raise ValueError(f"Unsupported FITS data shape: {data.shape}")
            else:
                raise ValueError("No image data found in FITS file")
    elif ext == '.tif' or ext == '.tiff':
        # Use tifffile for TIFF (handles 32-bit float properly)
        try:
            data = tifffile.imread(image_path)
        except Exception as e:
            if "LZW" in str(e) and not has_imagecodecs():
                # LZW compression detected but no imagecodecs
                raise ValueError(
                    "This TIFF file uses LZW compression. Please install the 'imagecodecs' package:\n"
                    "pip install imagecodecs"
                ) from e
            else:
                raise
                
        # Preserve float32 data as is, just ensure [0,1] range
        data = data.astype(np.float32)
        if np.max(data) > 1.0:
            data = data / np.max(data)
        return data
    else:
        raise ValueError(f"Unsupported file format: {ext}. Only TIFF and FITS are supported.")
        
    return None

def load_image_preview(image_path):
    """
    Load an image and convert to 8-bit RGB for preview display
    """
    
    image = load_image(image_path)
    
    # Convert to 8-bit RGB for display
    if image is None:
        return None
    
    if len(image.shape) == 2:
        image = np.stack([image, image, image], axis=2)
    
    if len(image.shape) == 3 and image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    
    image_8bit = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    
    return image_8bit

def save_image(output_path, image_data, compression="lzw"):
    """
    Save image to file with proper format
    
    Parameters:
    -----------
    output_path : str
        Path to save the image
    image_data : ndarray
        Image data to save
    compression : str, optional
        Compression to use for TIFF files. Options: None, "lzw", "zlib", "deflate"
        LZW requires imagecodecs package
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    _, ext = os.path.splitext(output_path)
    ext = ext.lower()
    
    # Save based on format
    if ext == '.tif' or ext == '.tiff':
        if compression == "lzw" and not has_imagecodecs():
            print("Warning: LZW compression requested but imagecodecs not available.")
            print("Falling back to no compression. Install imagecodecs for LZW support.")
            compression = None
            
        # Save as 32-bit TIFF with specified compression
        tifffile.imwrite(
            output_path, 
            image_data.astype(np.float32), 
            compression=compression
        )
    else:
        # For other formats, convert to 8-bit
        image_8bit = (np.clip(image_data, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(output_path, cv2.cvtColor(image_8bit, cv2.COLOR_RGB2BGR))
        
    return True

def array_to_qimage(array):
    """
    Convert a numpy array to QImage for display
    """
    # Make sure we're working with 8-bit RGB
    if array.dtype != np.uint8:
        array = (np.clip(array, 0, 1) * 255).astype(np.uint8)
    
    # Handle different array shapes
    if len(array.shape) == 2:
        # Grayscale
        height, width = array.shape
        # Convert to rgb first
        rgb = np.stack([array, array, array], axis=2)
        return QImage(rgb.tobytes(), width, height, width*3, QImage.Format_RGB888)
    elif len(array.shape) == 3:
        height, width, channels = array.shape
        
        if channels == 3:
            # RGB
            return QImage(array.tobytes(), width, height, width*3, QImage.Format_RGB888)
        elif channels == 4:
            # RGBA
            return QImage(array.tobytes(), width, height, width*4, QImage.Format_RGBA8888)
        elif channels == 1:
            # Single channel - convert to RGB
            rgb = np.repeat(array, 3, axis=2)
            return QImage(rgb.tobytes(), width, height, width*3, QImage.Format_RGB888)
    
    # Fallback
    raise ValueError(f"Unsupported array shape: {array.shape}")

def scale_image_for_display(pixmap, max_size=800):
    """Scale a pixmap to fit within max_size while preserving aspect ratio"""
    if pixmap.width() > max_size or pixmap.height() > max_size:
        if pixmap.width() > pixmap.height():
            pixmap = pixmap.scaledToWidth(max_size, Qt.SmoothTransformation)
        else:
            pixmap = pixmap.scaledToHeight(max_size, Qt.SmoothTransformation)
    return pixmap