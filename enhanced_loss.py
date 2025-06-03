"""
Enhanced loss functions for astronomical image processing with fixed type casting
and proper serialization support
"""

import tensorflow as tf
import numpy as np

def ssim_loss(y_true, y_pred):
    """
    Structural similarity loss
    
    Parameters:
    -----------
    y_true : tf.Tensor
        Ground truth image
    y_pred : tf.Tensor
        Predicted image
        
    Returns:
    --------
    loss : tf.Tensor
        1 - SSIM (convert similarity to a loss)
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

def ms_ssim_loss(y_true, y_pred):
    """
    Multi-scale structural similarity loss
    Uses multiple scales to better capture both fine and coarse details
    
    Parameters:
    -----------
    y_true : tf.Tensor
        Ground truth image
    y_pred : tf.Tensor
        Predicted image
        
    Returns:
    --------
    loss : tf.Tensor
        1 - MS_SSIM (convert similarity to a loss)
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # MS-SSIM requires at least 96x96 images and specific channel counts
    try:
        ms_ssim_value = tf.image.ssim_multiscale(y_true, y_pred, max_val=1.0)
        return 1.0 - tf.reduce_mean(ms_ssim_value)
    except (tf.errors.InvalidArgumentError, ValueError):
        # Fall back to regular SSIM if MS-SSIM fails
        return ssim_loss(y_true, y_pred)

def edge_aware_loss(y_true, y_pred, edge_weight=2.0):
    """
    Edge-aware loss that gives extra weight to high-gradient areas
    Helps preserve fine details in structures
    
    Parameters:
    -----------
    y_true : tf.Tensor
        Ground truth image
    y_pred : tf.Tensor
        Predicted image
    edge_weight : float
        Extra weight for edge areas
        
    Returns:
    --------
    loss : tf.Tensor
        Weighted MSE loss
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    true_gradients = tf.image.sobel_edges(y_true)
    true_gradient_magnitude = tf.sqrt(tf.reduce_sum(tf.square(true_gradients), axis=-1))
    
    edge_mask = tf.clip_by_value(true_gradient_magnitude, 0.0, 1.0)
    
    edge_mask = edge_mask * (edge_weight - 1.0) + 1.0
    
    squared_diff = tf.square(y_true - y_pred)
    
    # Apply weights
    weighted_squared_diff = squared_diff * edge_mask
    
    return tf.reduce_mean(weighted_squared_diff)

def brightness_weighted_mse(y_true, y_pred, bright_weight=2.0, threshold=0.8):
    """
    MSE loss that gives extra weight to bright regions
    Similar to your astronomical_loss but with configurable parameters
    
    Parameters:
    -----------
    y_true : tf.Tensor
        Ground truth image
    y_pred : tf.Tensor
        Predicted image
    bright_weight : float
        Extra weight for bright regions
    threshold : float
        Brightness threshold (0.0 to 1.0)
        
    Returns:
    --------
    loss : tf.Tensor
        Brightness-weighted MSE loss
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    bright_regions = tf.cast(y_true > threshold, tf.float32)
    
    # Base MSE loss
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # Extra weight for bright regions
    bright_loss = tf.reduce_mean(tf.square(y_true - y_pred) * bright_regions) * bright_weight
    
    # Combine losses
    combined_loss = mse + bright_loss
    
    return combined_loss

def hybrid_l1_l2_loss(y_true, y_pred, l1_weight=0.5):
    """
    Hybrid L1-L2 loss combining MAE and MSE
    L1 (MAE) is more robust to outliers and can help preserve edges
    
    Parameters:
    -----------
    y_true : tf.Tensor
        Ground truth image
    y_pred : tf.Tensor
        Predicted image
    l1_weight : float
        Weight for L1 component (0.0 to 1.0)
        
    Returns:
    --------
    loss : tf.Tensor
        Weighted combination of L1 and L2 losses
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    l1_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
    l2_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    
    return l1_weight * l1_loss + (1 - l1_weight) * l2_loss

def combined_astronomical_loss(y_true, y_pred, 
                              ssim_weight=0.3, 
                              edge_weight=0.2,
                              bright_weight=2.0,
                              l1_weight=0.3,
                              threshold=0.8):
    """
    Comprehensive loss function combining multiple components
    
    Parameters:
    -----------
    y_true : tf.Tensor
        Ground truth image
    y_pred : tf.Tensor
        Predicted image
    ssim_weight : float
        Weight for SSIM component
    edge_weight : float
        Weight for edge-aware component
    bright_weight : float
        Extra weight for bright regions
    l1_weight : float
        Weight for L1 component in the hybrid loss
    threshold : float
        Brightness threshold
        
    Returns:
    --------
    loss : tf.Tensor
        Combined loss
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Base loss: hybrid L1-L2 with brightness weighting
    base_l1_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
    base_l2_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    hybrid_base = l1_weight * base_l1_loss + (1 - l1_weight) * base_l2_loss
    
    bright_regions = tf.cast(y_true > threshold, tf.float32)
    bright_loss = tf.reduce_mean(tf.square(y_true - y_pred) * bright_regions) * bright_weight
    
    true_gradients = tf.image.sobel_edges(y_true)
    true_gradient_magnitude = tf.sqrt(tf.reduce_sum(tf.square(true_gradients), axis=-1))
    edge_mask = tf.clip_by_value(true_gradient_magnitude, 0.0, 1.0)
    edge_loss = tf.reduce_mean(tf.square(y_true - y_pred) * edge_mask) * 2.0
    
    # SSIM component
    ssim_value = tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    ssim_component = 1.0 - ssim_value  # Convert to loss
    
    # Combine all components
    combined = hybrid_base + bright_loss + edge_weight * edge_loss + ssim_weight * ssim_component
    
    return combined

def perceptual_loss(y_true, y_pred, feature_extractor=None):
    """
    Perceptual loss using a pre-trained feature extractor
    
    Parameters:
    -----------
    y_true : tf.Tensor
        Ground truth image
    y_pred : tf.Tensor
        Predicted image
    feature_extractor : tf.keras.Model or None
        Pre-trained model for feature extraction
        
    Returns:
    --------
    loss : tf.Tensor
        Perceptual loss
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # If no feature extractor provided, create one from VGG
    if feature_extractor is None:
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        
        selected_layers = ['block1_conv2', 'block2_conv2', 'block3_conv2', 'block4_conv2']
        outputs = [vgg.get_layer(name).output for name in selected_layers]
        
        feature_extractor = tf.keras.Model(inputs=vgg.input, outputs=outputs)
        feature_extractor.trainable = False
    
    def preprocess(x):
        x = x * 255.0
        return tf.keras.applications.vgg19.preprocess_input(x)
    
    y_true_processed = preprocess(y_true)
    y_pred_processed = preprocess(y_pred)
    
    true_features = feature_extractor(y_true_processed)
    pred_features = feature_extractor(y_pred_processed)
    
    loss = 0.0
    for true_feat, pred_feat in zip(true_features, pred_features):
        loss += tf.reduce_mean(tf.square(true_feat - pred_feat))
    
    return loss / len(true_features)

def denoise_loss(y_true, y_pred, noise_mask=None):
    """
    Loss function that penalizes noise amplification in low-signal regions
    """
    if noise_mask is None:
        noise_mask = tf.cast(y_true < 0.1, tf.float32)
    
    noise_diff = tf.square(y_true - y_pred) * noise_mask
    
    return tf.reduce_mean(noise_diff)

class AstronomicalLossWrapper(tf.keras.losses.Loss):
    """
    Configurable wrapper for astronomical loss functions
    Makes it easy to experiment with different loss combinations
    Now inherits from tf.keras.losses.Loss for proper serialization
    """
    def __init__(self, 
                ssim_weight=0.3,
                perceptual_weight=0.0,
                edge_weight=0.2,
                bright_weight=1.2,
                l1_weight=0.3,
                threshold=0.8,
                denoise_weight=0.3,
                feature_extractor=None,
                name='astronomical_loss',
                **kwargs):
        """
        Initialize loss function with configurable weights
        
        Parameters:
        -----------
        ...existing params...
        denoise_weight : float
            Weight for denoise loss component
        ...
        """
        super(AstronomicalLossWrapper, self).__init__(name=name, **kwargs)
        self.ssim_weight = ssim_weight
        self.perceptual_weight = 0.0  # Temporarily disable
        self.edge_weight = edge_weight
        self.bright_weight = bright_weight
        self.l1_weight = l1_weight
        self.threshold = threshold
        self.denoise_weight = denoise_weight
        
        super(AstronomicalLossWrapper, self).__init__(name=name, **kwargs)
        
        self.use_feature_extractor = perceptual_weight > 0 and feature_extractor is None
        self.feature_extractor = None
        
        if self.use_feature_extractor:
            self._setup_feature_extractor()
        elif feature_extractor is not None:
            self.feature_extractor = feature_extractor
    
    def _setup_feature_extractor(self):
        """Create a feature extractor from VGG if needed"""
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        
        selected_layers = ['block1_conv2', 'block2_conv2', 'block3_conv2', 'block4_conv2']
        outputs = [vgg.get_layer(name).output for name in selected_layers]
        
        self.feature_extractor = tf.keras.Model(inputs=vgg.input, outputs=outputs)
        self.feature_extractor.trainable = False
            
    def call(self, y_true, y_pred):
        """
        Calculate loss - renamed from __call__ to standard call method
        
        Parameters:
        -----------
        y_true : tf.Tensor
            Ground truth image
        y_pred : tf.Tensor
            Predicted image
            
        Returns:
        --------
        loss : tf.Tensor
            Combined loss
        """
        # Ensure both inputs are float32
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        base_l1_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
        base_l2_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        hybrid_base = self.l1_weight * base_l1_loss + (1 - self.l1_weight) * base_l2_loss
        
        bright_regions = tf.cast(y_true > self.threshold, tf.float32)

        bright_loss = tf.cast(
            tf.reduce_mean(tf.square(y_true - y_pred) * bright_regions) * self.bright_weight,
            tf.float32
        )
        
        true_gradients = tf.image.sobel_edges(y_true)
        true_gradient_magnitude = tf.sqrt(tf.reduce_sum(tf.square(true_gradients), axis=-1))
        edge_mask = tf.clip_by_value(true_gradient_magnitude, 0.0, 1.0)

        edge_loss = tf.cast(
            tf.reduce_mean(tf.square(y_true - y_pred) * edge_mask) * 2.0,
            tf.float32
        )
        
        ssim_value = tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
        ssim_component = tf.cast(1.0 - ssim_value, tf.float32)  # Convert to loss
        
        perceptual_component = tf.constant(0.0, dtype=tf.float32)
        if self.perceptual_weight > 0 and self.use_feature_extractor:

            if self.feature_extractor is None and self.use_feature_extractor:
                self._setup_feature_extractor()
                
            def preprocess(x):
                x = x * 255.0
                return tf.keras.applications.vgg19.preprocess_input(x)
            
            try:
                y_true_processed = preprocess(y_true)
                y_pred_processed = preprocess(y_pred)
                
                true_features = self.feature_extractor(y_true_processed)
                pred_features = self.feature_extractor(y_pred_processed)
                
                feat_loss = 0.0
                for true_feat, pred_feat in zip(true_features, pred_features):
                    feat_loss += tf.reduce_mean(tf.square(true_feat - pred_feat))
                
                perceptual_component = tf.cast(feat_loss / len(true_features), tf.float32)
            except (ValueError, tf.errors.InvalidArgumentError):
                perceptual_component = tf.constant(0.0, dtype=tf.float32)
        
        denoise_component = tf.constant(0.0, dtype=tf.float32)
        if hasattr(self, 'denoise_weight') and self.denoise_weight > 0:
            noise_mask = tf.cast(y_true < 0.1, tf.float32)
            
            noise_diff = tf.square(y_true - y_pred) * noise_mask
            
            denoise_component = tf.cast(tf.reduce_mean(noise_diff), tf.float32)
        
        components = [
            hybrid_base,
            bright_loss,
            tf.multiply(tf.cast(self.edge_weight, tf.float32), edge_loss),
            tf.multiply(tf.cast(self.ssim_weight, tf.float32), ssim_component),
            tf.multiply(tf.cast(self.perceptual_weight, tf.float32), perceptual_component)
        ]
        
        if hasattr(self, 'denoise_weight') and self.denoise_weight > 0:
            components.append(tf.multiply(tf.cast(self.denoise_weight, tf.float32), denoise_component))
        
        combined = tf.add_n(components)
        combined = tf.where(tf.math.is_finite(combined), combined, 0.5)
        
        return combined
    
    def get_config(self):
        """
        Return configuration for serialization
        """
        config = super(AstronomicalLossWrapper, self).get_config()
        config.update({
            'ssim_weight': self.ssim_weight,
            'perceptual_weight': self.perceptual_weight,
            'edge_weight': self.edge_weight,
            'bright_weight': self.bright_weight,
            'l1_weight': self.l1_weight,
            'threshold': self.threshold,
            'denoise_weight': self.denoise_weight,
            'use_feature_extractor': self.use_feature_extractor
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        """
        Create an instance from configuration
        """
        use_feature_extractor = config.pop('use_feature_extractor', False)
        
        instance = cls(**config)
        instance.use_feature_extractor = use_feature_extractor
        
        if use_feature_extractor:
            instance._setup_feature_extractor()
        
        return instance

# For backward compatibility
def astronomical_loss(y_true, y_pred):
    """
    Custom loss function for astronomical images
    Original function from the previous implementation
    Gives extra weight to regions with stars
    
    Parameters:
    -----------
    y_true : tf.Tensor
        Ground truth image
    y_pred : tf.Tensor
        Predicted image
        
    Returns:
    --------
    loss : tf.Tensor
        Combined loss value
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # Identify stellar regions (high brightness areas)
    stellar_regions = tf.cast(y_true > 0.8, tf.float32)
    
    # Extra weight for stellar regions
    stellar_loss = tf.reduce_mean(tf.square(y_true - y_pred) * stellar_regions) * 2.0
    
    # Combine losses
    combined_loss = mse + stellar_loss
    
    return combined_loss

