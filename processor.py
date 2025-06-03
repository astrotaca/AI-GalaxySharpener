import os
import numpy as np
import tensorflow as tf
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import custom modules for model loading
try:
    from enhanced_loss import AstronomicalLossWrapper, denoise_loss
except ImportError:
    print("Warning: enhanced_loss module not found. Models may not load correctly.")

class AstroSharpener:
    def __init__(self):
        self.models = {}
        self.current_model = None
        self.current_model_name = None
        self.models_path = "models"
        
        # Configure GPU memory growth
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"Enabled memory growth for GPU: {gpu}")
                except:
                    print(f"Failed to enable memory growth for GPU: {gpu}")
    
    def load_available_models(self):
        """Load all available models from the models directory"""
        # Check models directory exists
        if not os.path.exists(self.models_path):
            os.makedirs(self.models_path)
        
        self.models = {}
        
        # Look for model files
        model_files = [f for f in os.listdir(self.models_path) if f.endswith('.h5')]
        
        for model_file in model_files:
            name = os.path.splitext(model_file)[0]
            
            # Determine model type from filename
            model_type = "General"
            if "starless" in name.lower():
                model_type = "Starless Galaxy"
            elif "stars" in name.lower():
                model_type = "Stars"
            elif "nebula" in name.lower():
                model_type = "Nebula"
            
            self.models[name] = {
                'path': os.path.join(self.models_path, model_file),
                'loaded': False,
                'model': None,
                'type': model_type
            }
        
        return len(self.models) > 0
    
    def get_model_types(self):
        """Get a list of unique model types"""
        if not self.models:
            return []
            
        types = set()
        for info in self.models.values():
            types.add(info['type'])
            
        return sorted(list(types))
    
    def load_model(self, model_name):
        """Load a specific model into memory"""
        if model_name not in self.models:
            print(f"Model {model_name} not found")
            return False
            
        if self.models[model_name]['loaded']:
            self.current_model = self.models[model_name]['model']
            self.current_model_name = model_name
            return True
            
        try:
            custom_objects = {
                'AstronomicalLossWrapper': AstronomicalLossWrapper,
                'denoise_loss': denoise_loss
            }
            
            model_path = self.models[model_name]['path']
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
            
            self.models[model_name]['model'] = model
            self.models[model_name]['loaded'] = True
            self.current_model = model
            self.current_model_name = model_name
            return True
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            return False
    
    def is_background_region(self, tile, threshold=0.1, edge_threshold=0.02):
        """
        Determine if a tile is a pure background region that should be preserved
        """
        # Convert to grayscale if needed
        if len(tile.shape) > 2 and tile.shape[2] > 1:
            gray = np.mean(tile, axis=2)
        else:
            gray = np.squeeze(tile)
        
        # Calculate basic stats
        mean_val = np.mean(gray)
        
        if mean_val > threshold:
            return False
            
        edge_mag = cv2.Sobel(gray, cv2.CV_32F, 1, 1, ksize=3)
        edge_mag = np.abs(edge_mag)
        edge_mean = np.mean(edge_mag)
        
        return mean_val < threshold and edge_mean < edge_threshold
    
    def apply_strength_blend(self, original, processed, strength=1.0):
        """Blend between original and processed images based on strength"""
        
        if strength == 1.0:
            return processed
        elif strength > 1.0:
            diff = processed - original
            return np.clip(original + diff * strength, 0, 1)
        else:
            return original * (1 - strength) + processed * strength
    
    def process_image(self, input_path, output_path, strength=1.0, preserve_background=True, 
                     tile_size=128, overlap=64, progress_callback=None, status_callback=None):
        """Process an image with the currently loaded model"""
        if self.current_model is None:
            if status_callback:
                status_callback("No model loaded")
            return False
        
        # Load image
        if status_callback:
            status_callback("Loading image...")
            
        from utils import load_image
        image = load_image(input_path)
        
        if image is None:
            if status_callback:
                status_callback(f"Failed to load image: {input_path}")
            return False
        
        orig_min = np.min(image)
        orig_max = np.max(image)
        orig_mean = np.mean(image)
        orig_std = np.std(image)
        
        if status_callback:
            status_callback(f"Processing image: {image.shape}")
            
        # Ensure 3D with channels, matching training
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=-1)
        
        height, width, channels = image.shape
        
        # output array
        output = np.zeros((height, width, channels), dtype=np.float32)
        weight_map = np.zeros((height, width), dtype=np.float32)
        
        # Pre-compute a full 2D Hann window for better blending (raised cosine)
        y_coords, x_coords = np.meshgrid(np.arange(tile_size), np.arange(tile_size), indexing='ij')
        hann_window = np.sin(np.pi * y_coords / (tile_size - 1)) * np.sin(np.pi * x_coords / (tile_size - 1))
        
        # Calculate steps and total tiles
        y_steps = list(range(0, height - tile_size + 1, tile_size - overlap))
        x_steps = list(range(0, width - tile_size + 1, tile_size - overlap))
        
        # Make sure we include the last tile that covers the edge
        if height > tile_size and (height - tile_size) not in y_steps:
            y_steps.append(height - tile_size)
        if width > tile_size and (width - tile_size) not in x_steps:
            x_steps.append(width - tile_size)
            
        total_tiles = len(y_steps) * len(x_steps)
        
        background_tile_count = 0
        processed_tile_count = 0
        
        # Process tiles
        current_tile = 0
        for y in tqdm(y_steps, desc="Processing rows", disable=progress_callback is None):
            for x in x_steps:
                current_tile += 1
                if progress_callback:
                    progress_percent = int((current_tile / total_tiles) * 100)
                    progress_callback(progress_percent)
                
                end_y = min(y + tile_size, height)
                end_x = min(x + tile_size, width)
                actual_h = end_y - y
                actual_w = end_x - x
                
                if actual_h < tile_size // 2 or actual_w < tile_size // 2:
                    continue
                    
                if actual_h == tile_size and actual_w == tile_size:
                    tile = image[y:end_y, x:end_x].copy()
                else:
                    tile = np.pad(
                        image[y:end_y, x:end_x],
                        ((0, tile_size - actual_h), (0, tile_size - actual_w), (0, 0)),
                        mode='reflect'
                    )
                
                try:
                    # Check if this is a pure background region that should be preserved
                    if preserve_background and self.is_background_region(tile):
                        processed = tile
                        background_tile_count += 1
                    else:
                        tile = tile.astype(np.float32)
                        
                        batch = np.expand_dims(tile, axis=0)
                        
                        # Process tile with model
                        result = self.current_model(batch, training=False).numpy()
                        
                        if strength != 1.0:
                            processed = self.apply_strength_blend(tile, result[0], strength)
                        else:
                            processed = result[0].astype(np.float32)
                            
                        processed_tile_count += 1
                    
                    processed = np.clip(processed, 0, 1)
                    
                    if current_tile == 1:
                        plt.figure(figsize=(12, 6))
                        plt.subplot(1, 2, 1)
                        plt.imshow(np.clip(tile, 0, 1))
                        plt.title("Input Sample")
                        plt.axis('off')
                        
                        plt.subplot(1, 2, 2)
                        plt.imshow(np.clip(processed, 0, 1))
                        plt.title("Model Output" if not self.is_background_region(tile) else "Background (preserved)")
                        plt.axis('off')
                        
                        plt.tight_layout()
                        plt.savefig("sample_process.png")
                    
                    # For partial tiles, use the appropriate slice of the Hann window
                    if actual_h == tile_size and actual_w == tile_size:
                        weights = hann_window.copy()
                    else:
                        weights = hann_window[:actual_h, :actual_w]
                    
                    for c in range(channels):
                        output[y:end_y, x:end_x, c] += processed[:actual_h, :actual_w, c] * weights
                    
                    weight_map[y:end_y, x:end_x] += weights
                    
                except Exception as e:
                    print(f"\nError processing tile at ({x},{y}): {e}")
        
        if status_callback:
            status_callback("Blending tiles...")
            
        mask = weight_map > 0
        for c in range(channels):
            channel = output[:, :, c]
            channel[mask] /= weight_map[mask]
        
        output = np.clip(output, 0, 1)
        
        background_percentage = (background_tile_count / (processed_tile_count + background_tile_count)) * 100
        
        if status_callback:
            status_callback(f"Processed {processed_tile_count} tiles, preserved {background_tile_count} background tiles")
        
        if status_callback:
            status_callback(f"Saving result to {output_path}")
            
        from utils import save_image
        save_image(output_path, output)
        
        # Create histogram comparison
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(image.ravel(), bins=100, alpha=0.7, label='Original', color='blue')
        plt.title("Input Histogram")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        
        plt.subplot(1, 2, 2)
        plt.hist(output.ravel(), bins=100, alpha=0.7, label='Processed', color='green')
        plt.title("Output Histogram")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        
        plt.tight_layout()
        plt.savefig("histogram_comparison.png")
        
        # Print statistics
        processed_stats = {
            'min': float(np.min(output)),
            'max': float(np.max(output)), 
            'mean': float(np.mean(output)),
            'std': float(np.std(output))
        }
        
        if status_callback:
            status_callback(f"Processing complete! Model: {self.current_model_name}, Strength: {strength}")
            
        return True, output_path