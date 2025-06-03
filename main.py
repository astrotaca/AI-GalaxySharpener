import os
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QPushButton, QFileDialog, QSlider, QComboBox, QCheckBox,
                            QGroupBox, QSplitter, QTabWidget, QMessageBox, QProgressBar,
                            QScrollArea, QSizePolicy, QGraphicsView, QGraphicsScene, 
                            QGraphicsPixmapItem, QMenu, QAction)
from PyQt5.QtGui import (QPixmap, QImage, QFont, QBrush, QColor, QPainter, QPen, 
                         QPainterPath, QPolygonF)
from PyQt5.QtCore import Qt, QSize, QPoint, QRectF, QPointF, pyqtSignal
from PyQt5.QtGui import QPainterPath 
import numpy as np
import math
from processor import AstroSharpener
from worker import ProcessingWorker
from utils import array_to_qimage, scale_image_for_display, load_image_preview


class ZoomableImageView(QGraphicsView):
    zoomChanged = pyqtSignal(float, QPointF)  # Emitted when zoom or position changes
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        
        self.pixmap_item = None
        
        self.zoom_factor = 1.0
        self.max_zoom = 8.0 
        
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setOptimizationFlag(QGraphicsView.DontAdjustForAntialiasing, True)
        
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        
        # Set white background
        self.setBackgroundBrush(QBrush(Qt.white))
        
        self.is_syncing = False
        
    def setPixmap(self, pixmap):
        """Set a new pixmap as the image"""
        self.scene.clear()
        
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.pixmap_item.setTransformationMode(Qt.SmoothTransformation)
        
        self.scene.addItem(self.pixmap_item)
        
        rect = self.pixmap_item.boundingRect()
        self.scene.setSceneRect(rect)
        
        self.resetTransform()
        self.fitInView(rect, Qt.KeepAspectRatio) 
        
        # Calculate and store the actual zoom factor after fitting
        transform = self.transform()
        self.zoom_factor = transform.m11() 
        
        self.viewport().update()
        
    def setText(self, text):
        """Display text when no image is loaded"""

        self.scene.clear()
        self.pixmap_item = None
        
        text_item = self.scene.addText(text)
        text_item.setDefaultTextColor(QColor(100, 100, 100))
        
        self.scene.setSceneRect(text_item.boundingRect())
        
    def wheelEvent(self, event):
        """Handle zooming with the mouse wheel"""
        if not self.pixmap_item:
            return super().wheelEvent(event)
            
        old_pos = self.mapToScene(event.pos())
        
        zoom_in = event.angleDelta().y() > 0
        
        old_zoom = self.zoom_factor
        
        if zoom_in and self.zoom_factor < self.max_zoom:
            self.zoom_factor *= 1.25
        elif not zoom_in and self.zoom_factor > 0.1:
            self.zoom_factor /= 1.25
        else:
            return 
            
        power = round(math.log(self.zoom_factor, 1.25))
        self.zoom_factor = 1.25 ** power
        
        self.zoom_factor = max(0.1, min(self.max_zoom, self.zoom_factor))
        
        self.resetTransform()
        self.scale(self.zoom_factor, self.zoom_factor)
        
        new_pos = self.mapToScene(event.pos())
        delta = new_pos - old_pos
        
        # Adjust view position to maintain point under cursor
        self.translate(delta.x(), delta.y())
        
        if not self.is_syncing:
            center = self.mapToScene(self.viewport().rect().center())
            self.zoomChanged.emit(self.zoom_factor, center)
            
        event.accept()

    def devicePixelRatioF(self):
        """Get device pixel ratio, with fallback for older Qt versions"""
        if hasattr(self, 'devicePixelRatio'):
            return self.devicePixelRatio()
        elif hasattr(self.viewport(), 'devicePixelRatioF'):
            return self.viewport().devicePixelRatioF()
        else:
            return 1.0
            
    def mousePressEvent(self, event):
        """Handle mouse press events"""
        if event.button() == Qt.LeftButton and event.modifiers() & Qt.ControlModifier:
            # Reset zoom on Ctrl+click
            if self.pixmap_item:
                self.resetZoom()
                event.accept()
                return
        
        super().mousePressEvent(event)
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release events"""
        result = super().mouseReleaseEvent(event)
        
        if event.button() == Qt.LeftButton and self.pixmap_item and not self.is_syncing:
            center = self.mapToScene(self.viewport().rect().center())
            self.zoomChanged.emit(self.zoom_factor, center)
            
        return result
        
    def contextMenuEvent(self, event):
        """Show context menu for zoom options"""
        if not self.pixmap_item:
            return super().contextMenuEvent(event)
            
        menu = QMenu(self)
        
        zoom_in = QAction("Zoom In", self)
        zoom_in.triggered.connect(lambda: self.zoom(1.25))
        menu.addAction(zoom_in)
        
        zoom_out = QAction("Zoom Out", self)
        zoom_out.triggered.connect(lambda: self.zoom(0.8))
        menu.addAction(zoom_out)
        
        fit = QAction("Fit to View", self)
        fit.triggered.connect(self.fitToView)
        menu.addAction(fit)
        
        reset = QAction("Reset Zoom (100%)", self)
        reset.triggered.connect(self.resetZoom)
        menu.addAction(reset)
        
        menu.addSeparator()
        zoom_percent = QAction(f"Current Zoom: {int(self.zoom_factor * 100)}%", self)
        zoom_percent.setEnabled(False)
        menu.addAction(zoom_percent)
        
        menu.exec_(event.globalPos())
        
    def zoom(self, factor):
        """Zoom by a specific factor"""
        if not self.pixmap_item:
            return
            
        center = self.mapToScene(self.viewport().rect().center())
        
        old_zoom = self.zoom_factor
        
        self.zoom_factor *= factor
        self.zoom_factor = max(0.1, min(self.max_zoom, self.zoom_factor))
        
        power = round(math.log(self.zoom_factor, 1.25))
        self.zoom_factor = 1.25 ** power
        
        self.resetTransform()
        self.scale(self.zoom_factor, self.zoom_factor)
        
        self.centerOn(center)
        
        if not self.is_syncing:
            self.zoomChanged.emit(self.zoom_factor, center)
            
    def resetZoom(self):
        """Reset to 100% zoom (actual size, 1:1 pixel mapping)"""
        if not self.pixmap_item:
            return
            
        pixmap_width = self.pixmap_item.pixmap().width()
        pixmap_height = self.pixmap_item.pixmap().height()
        
        self.resetTransform()
        
        self.zoom_factor = 1.0
        
        viewport_center_x = self.viewport().width() / 2
        viewport_center_y = self.viewport().height() / 2
        
        image_center_x = pixmap_width / 2
        image_center_y = pixmap_height / 2
        
        self.centerOn(image_center_x, image_center_y)
        
        center = self.mapToScene(self.viewport().rect().center())
        
        if not self.is_syncing:
            self.zoomChanged.emit(self.zoom_factor, center)
            
        self.viewport().update()
            
    def fitToView(self):
        """Fit the image to the current view size"""
        if not self.pixmap_item:
            return
        
        self.resetTransform()
        
        self.fitInView(self.pixmap_item.boundingRect(), Qt.KeepAspectRatio)
        
        transform = self.transform()
        self.zoom_factor = transform.m11()
        
        center = self.mapToScene(self.viewport().rect().center())
        
        if not self.is_syncing:
            self.zoomChanged.emit(self.zoom_factor, center)
            
    def getZoomAndPosition(self):
        """Get current zoom factor and view center position"""
        if not self.pixmap_item:
            return (1.0, QPointF(0, 0))
            
        # Get the center point in scene coordinates
        center = self.mapToScene(self.viewport().rect().center())
        return (self.zoom_factor, center)
    
    def setZoomAndPosition(self, zoom_factor, center_pos):
        """Set zoom factor and center position with precision"""
        if not self.pixmap_item:
            return
            
        self.is_syncing = True
        
        exact_center = QPointF(center_pos)
        
        self.zoom_factor = zoom_factor
        self.resetTransform()
        self.scale(self.zoom_factor, self.zoom_factor)
        
        self.centerOn(exact_center)
        
        self.viewport().update()
        
        self.is_syncing = False


class DraggableComparisonView(QGraphicsView):
    """A view that shows before/after comparison with a draggable divider"""
    zoomChanged = pyqtSignal(float, QPointF)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        
        self.original_pixmap = None
        self.processed_pixmap = None
        
        self.divider_position = 0.5
        
        self.dragging = False
        self.divider_width = 4
        self.divider_handle_size = 40
        
        self.zoom_factor = 1.0
        self.max_zoom = 8.0
        
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setOptimizationFlag(QGraphicsView.DontAdjustForAntialiasing, True)
        
        # Scrollbars
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        
        # Background
        self.setBackgroundBrush(QBrush(Qt.white))
        
        self.is_syncing = False
    
    def setImages(self, original, processed):
        """Set both original and processed images"""
        if original is None or processed is None:
            return
            
        self.original_pixmap = original
        self.processed_pixmap = processed
        
        self.scene.clear()
        self.updateScene()
        rect = self.scene.sceneRect()
        self.fitInView(rect, Qt.KeepAspectRatio)
        self.viewport().update()
    
    def updateScene(self):
        """Update the scene with current images and divider position"""
        if not self.original_pixmap or not self.processed_pixmap:
            return
            
        self.scene.clear()
        
        width = self.original_pixmap.width()
        height = self.original_pixmap.height()
        
        self.scene.setSceneRect(0, 0, width, height)
        divider_x = int(width * self.divider_position)
        orig_img = self.original_pixmap.toImage()
        left_part = orig_img.copy(0, 0, divider_x, height)
        left_pixmap = QPixmap.fromImage(left_part)
        self.scene.addPixmap(left_pixmap).setPos(0, 0)
        
        proc_img = self.processed_pixmap.toImage()
        right_part = proc_img.copy(divider_x, 0, width - divider_x, height)
        right_pixmap = QPixmap.fromImage(right_part)
        self.scene.addPixmap(right_pixmap).setPos(divider_x, 0)
        
        divider_rect = QRectF(divider_x - self.divider_width/2, 0, 
                            self.divider_width, height)
        self.scene.addRect(divider_rect, QPen(Qt.white, 2), QBrush(Qt.white))
        
        handle_y = height / 2
        handle_rect = QRectF(divider_x - self.divider_handle_size/2, 
                            handle_y - self.divider_handle_size/2,
                            self.divider_handle_size, self.divider_handle_size)
        self.scene.addEllipse(handle_rect, QPen(Qt.white, 2), QBrush(QColor(40, 93, 134, 180)))
        
        # Add arrow indicators
        arrow_width = 12
        arrow_height = 20
        
        # Left arrow
        left_arrow = QPolygonF()
        left_arrow.append(QPointF(divider_x - arrow_width/2, handle_y))
        left_arrow.append(QPointF(divider_x - arrow_width, handle_y - arrow_height/2))
        left_arrow.append(QPointF(divider_x - arrow_width, handle_y + arrow_height/2))
        self.scene.addPolygon(left_arrow, QPen(Qt.white, 1), QBrush(Qt.white))
        
        # Right arrow
        right_arrow = QPolygonF()
        right_arrow.append(QPointF(divider_x + arrow_width/2, handle_y))
        right_arrow.append(QPointF(divider_x + arrow_width, handle_y - arrow_height/2))
        right_arrow.append(QPointF(divider_x + arrow_width, handle_y + arrow_height/2))
        self.scene.addPolygon(right_arrow, QPen(Qt.white, 1), QBrush(Qt.white))
        
        self.scene.addSimpleText("Original", QFont("Arial", 8)).setPos(10, 10)
        processed_text = self.scene.addSimpleText("Processed", QFont("Arial", 8))
        processed_text.setPos(width - processed_text.boundingRect().width() - 10, 10)
    
    def setText(self, text):
        """Display text when no image is loaded (same interface as ZoomableImageView)"""
        self.scene.clear()
        self.original_pixmap = None
        self.processed_pixmap = None
        
        text_item = self.scene.addText(text)
        text_item.setDefaultTextColor(QColor(100, 100, 100))
        
        self.scene.setSceneRect(text_item.boundingRect())
    
    def mousePressEvent(self, event):
        """Handle mouse press events"""
        if not self.original_pixmap or not self.processed_pixmap:
            return super().mousePressEvent(event)
            
        scene_pos = self.mapToScene(event.pos())
        divider_x = int(self.original_pixmap.width() * self.divider_position)
        handle_y = self.original_pixmap.height() / 2
        
        near_divider = abs(scene_pos.x() - divider_x) < self.divider_width * 2
        near_handle = (abs(scene_pos.x() - divider_x) < self.divider_handle_size/2 and 
                      abs(scene_pos.y() - handle_y) < self.divider_handle_size/2)
        
        if near_divider or near_handle:
            self.dragging = True
            self.setCursor(Qt.SplitHCursor)
            return
            
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """Handle mouse move events"""
        if not self.dragging:
            return super().mouseMoveEvent(event)
            
        scene_pos = self.mapToScene(event.pos())
        width = self.original_pixmap.width()
        
        new_pos = max(0.05, min(0.95, scene_pos.x() / width))
        
        if abs(new_pos - self.divider_position) > 0.001:
            self.divider_position = new_pos
            self.updateScene()
            self.viewport().update()
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release events"""
        if self.dragging:
            self.dragging = False
            self.setCursor(Qt.ArrowCursor)
            return
            
        super().mouseReleaseEvent(event)
    
    def wheelEvent(self, event):
        """Handle zooming with the mouse wheel"""
        if not self.original_pixmap:
            return super().wheelEvent(event)
            
        old_pos = self.mapToScene(event.pos())
        
        zoom_in = event.angleDelta().y() > 0
        
        if zoom_in and self.zoom_factor < self.max_zoom:
            self.zoom_factor *= 1.25
        elif not zoom_in and self.zoom_factor > 0.1:
            self.zoom_factor /= 1.25
        else:
            return  # Zoom limit reached
            
        power = round(math.log(self.zoom_factor, 1.25))
        self.zoom_factor = 1.25 ** power
        
        self.zoom_factor = max(0.1, min(self.max_zoom, self.zoom_factor))
        
        self.resetTransform()
        self.scale(self.zoom_factor, self.zoom_factor)
        
        new_pos = self.mapToScene(event.pos())
        delta = new_pos - old_pos
        
        self.translate(delta.x(), delta.y())
        
        if not self.is_syncing:
            center = self.mapToScene(self.viewport().rect().center())
            self.zoomChanged.emit(self.zoom_factor, center)
            
        event.accept()
    
    def resetZoom(self):
        """Reset to 100% zoom (actual size, 1:1 pixel mapping)"""
        if not self.original_pixmap:
            return
            
        self.zoom_factor = 1.0
        self.resetTransform()
        
        center = self.scene.sceneRect().center()
        self.centerOn(center)
        
        if not self.is_syncing:
            center = self.mapToScene(self.viewport().rect().center())
            self.zoomChanged.emit(self.zoom_factor, center)
    
    def fitToView(self):
        """Fit the image to the current view size"""
        if not self.original_pixmap:
            return
            
        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        
        transform = self.transform()
        self.zoom_factor = transform.m11()
        
        center = self.mapToScene(self.viewport().rect().center())
        
        if not self.is_syncing:
            self.zoomChanged.emit(self.zoom_factor, center)
    
    def getZoomAndPosition(self):
        """Get current zoom factor and view center position"""
        if not self.original_pixmap:
            return (1.0, QPointF(0, 0))
            
        center = self.mapToScene(self.viewport().rect().center())
        return (self.zoom_factor, center)
    
    def setZoomAndPosition(self, zoom_factor, center_pos):
        """Set zoom factor and center position with precision"""
        if not self.original_pixmap:
            return
            
        self.is_syncing = True
        
        exact_center = QPointF(center_pos)
        
        self.zoom_factor = zoom_factor
        self.resetTransform()
        self.scale(self.zoom_factor, self.zoom_factor)
        self.centerOn(exact_center)
        self.viewport().update()
        
        self.is_syncing = False


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AstroSharpener")
        self.setMinimumSize(1200, 800)
        
        # Initialize the sharpener with the startup GPU setting
        self.sharpener = AstroSharpener(use_gpu=True)
        self.input_image_data = None
        self.processed_image_data = None
        
        # Setup UI
        self.setup_ui()
        
        # Load models automatically
        self.refresh_models()
        
        # Initialize status bar
        self.statusBar().setStyleSheet("""
            QStatusBar {
                background-color: #f0f0f0;
                color: #444;
                border-top: 1px solid #ccc;
            }
        """)
        self.statusBar().showMessage("Ready")
        
    def setup_ui(self):
        """Create the main UI layout"""
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        
        # Create a splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel: Controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Model selection group
        model_group = QGroupBox("Model Selection")
        model_layout = QVBoxLayout(model_group)

        self.model_combo = QComboBox()
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        model_layout.addWidget(QLabel("Model Type:"))
        model_layout.addWidget(self.model_combo)

        left_layout.addWidget(model_group)
        
        params_group = QGroupBox("Processing Parameters")
        params_layout = QVBoxLayout(params_group)
        
        strength_group = QGroupBox("Enhancement Strength")
        strength_layout = QVBoxLayout(strength_group)

        value_layout = QHBoxLayout()
        self.strength_value = QLabel("1.0×")
        self.strength_value.setAlignment(Qt.AlignCenter)
        self.strength_value.setStyleSheet("font-size: 16px; font-weight: bold; color: #2d5d86;")
        self.strength_value.setMinimumWidth(60)
        value_layout.addWidget(self.strength_value)
        strength_layout.addLayout(value_layout)

        self.strength_slider = QSlider(Qt.Horizontal)
        self.strength_slider.setRange(10, 200)
        self.strength_slider.setValue(100)
        self.strength_slider.setTickPosition(QSlider.TicksBelow)
        self.strength_slider.setTickInterval(10)
        self.strength_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 6px;
                background: #E0E0E0;
                border-radius: 3px;
                margin: 2px 0;
            }
            QSlider::handle:horizontal {
                background: #2d5d86;
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background: #3a7ab3;
            }
            QSlider::sub-page:horizontal {
                background: #3a7ab3;
                border-radius: 3px;
            }
        """)

        labels_layout = QHBoxLayout()
        min_label = QLabel("0.1×")
        min_label.setStyleSheet("color: #666;")
        max_label = QLabel("2.0×")
        max_label.setStyleSheet("color: #666;")
        labels_layout.addWidget(min_label)
        labels_layout.addStretch()
        labels_layout.addWidget(max_label)

        self.strength_slider.valueChanged.connect(self.update_strength_label)

        strength_layout.addWidget(self.strength_slider)
        strength_layout.addLayout(labels_layout)
        params_layout.addWidget(strength_group)
        
        self.preserve_bg_checkbox = QCheckBox("Preserve Background")
        self.preserve_bg_checkbox.setChecked(True)
        self.preserve_bg_checkbox.setToolTip("Skip processing on pure black sky background regions")
        params_layout.addWidget(self.preserve_bg_checkbox)
        
        advanced_group = QGroupBox("Advanced")
        advanced_layout = QVBoxLayout(advanced_group)

        # GPU/CPU selection
        self.use_gpu_checkbox = QCheckBox("Use GPU Acceleration")
        self.use_gpu_checkbox.setChecked(True)  # Default to GPU
        self.use_gpu_checkbox.setToolTip("Use GPU for faster processing (requires compatible graphics card)\nModel will reload when changed")
        self.use_gpu_checkbox.stateChanged.connect(self.on_gpu_setting_changed)
        advanced_layout.addWidget(self.use_gpu_checkbox)

        # Tile size
        tile_layout = QHBoxLayout()
        tile_layout.addWidget(QLabel("Tile Size:"))
        self.tile_size_combo = QComboBox()
        self.tile_size_combo.addItems(["64", "128", "256", "512"])
        self.tile_size_combo.setCurrentText("256")
        self.tile_size_combo.setToolTip("Size of processing tiles - larger tiles need more memory")
        tile_layout.addWidget(self.tile_size_combo)
        advanced_layout.addLayout(tile_layout)

        # Overlap
        overlap_layout = QHBoxLayout()
        overlap_layout.addWidget(QLabel("Overlap:"))
        self.overlap_combo = QComboBox()
        self.overlap_combo.addItems(["32", "64", "96", "128"])
        self.overlap_combo.setCurrentText("96")
        self.overlap_combo.setToolTip("Overlap between tiles - higher values blend better but slower")
        overlap_layout.addWidget(self.overlap_combo)
        advanced_layout.addLayout(overlap_layout)
        
        params_layout.addWidget(advanced_group)
        
        left_layout.addWidget(params_group)
        
        # File controls
        file_group = QGroupBox("File Controls")
        file_layout = QVBoxLayout(file_group)
        
        self.input_path_label = QLabel("No file selected")
        file_layout.addWidget(self.input_path_label)
        
        load_button = self.create_styled_button("Load Image")
        load_button.clicked.connect(self.load_image)
        file_layout.addWidget(load_button)

        self.process_button = self.create_styled_button("Process Image")
        self.process_button.clicked.connect(self.process_image)
        self.process_button.setEnabled(False)
        file_layout.addWidget(self.process_button)

        save_button = self.create_styled_button("Save Processed Image")
        save_button.clicked.connect(self.save_processed_image)
        save_button.setEnabled(False)
        self.save_button = save_button
        file_layout.addWidget(save_button)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: none;
                border-radius: 3px;
                text-align: center;
                height: 16px;
                background-color: #F0F0F0;
                margin: 6px 0px;
                color: #444;
            }
            QProgressBar::chunk {
                background-color: #3a7ab3;
                border-radius: 3px;
            }
        """)
        file_layout.addWidget(self.progress_bar)
        
        left_layout.addWidget(file_group)
        left_layout.addStretch()
        
        about_button = self.create_styled_button("About")
        about_button.clicked.connect(self.show_about)
        left_layout.addWidget(about_button)
        
        # Right panel: Image display
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        self.tab_widget = QTabWidget()
        
        self.original_view = ZoomableImageView()
        self.original_view.setText("Load an image to begin")
        self.tab_widget.addTab(self.original_view, "Original")

        self.processed_view = ZoomableImageView()
        self.processed_view.setText("Process an image to see results")
        self.tab_widget.addTab(self.processed_view, "Processed")

        self.split_view = DraggableComparisonView()
        self.split_view.setText("Process an image to compare")
        self.tab_widget.addTab(self.split_view, "Before/After")
        
        self.tab_widget.currentChanged.connect(self.sync_views)

        self.original_view.zoomChanged.connect(
            lambda zoom, pos: self.sync_view_from(self.original_view, zoom, pos))
        self.processed_view.zoomChanged.connect(
            lambda zoom, pos: self.sync_view_from(self.processed_view, zoom, pos))
        self.split_view.zoomChanged.connect(
            lambda zoom, pos: self.sync_view_from(self.split_view, zoom, pos))
        
        right_layout.addWidget(self.tab_widget)
        
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        
        splitter.setSizes([300, 900])
        
        main_layout.addWidget(splitter)
        
        self.setCentralWidget(main_widget)

    def on_gpu_setting_changed(self):
        """Handle GPU setting change"""
        new_setting = self.use_gpu_checkbox.isChecked()
        device_name = "GPU" if new_setting else "CPU"
        self.statusBar().showMessage(f"GPU/CPU setting changed to {device_name}. Will apply on next processing.")

    def create_styled_button(self, text, icon_name=None):
        """Create a styled button with optional icon"""
        button = QPushButton(text)
        button.setMinimumHeight(32)
        button.setStyleSheet("""
            QPushButton {
                background-color: #2d5d86;
                color: white;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3a7ab3;
            }
            QPushButton:pressed {
                background-color: #1c4c72;
            }
            QPushButton:disabled {
                background-color: #6c7580;
                color: #d0d0d0;
            }
        """)
        
        return button

        
    def sync_view_from(self, source_view, zoom, center):
        """Synchronize all views to match the source view's zoom and position"""
        if isinstance(source_view, ZoomableImageView) and not source_view.pixmap_item:
            return
        elif isinstance(source_view, DraggableComparisonView) and not source_view.original_pixmap:
            return
                
        current_tab = self.tab_widget.currentIndex()
        if (current_tab == 0 and source_view != self.original_view) or \
        (current_tab == 1 and source_view != self.processed_view) or \
        (current_tab == 2 and source_view != self.split_view):
            return
        
        # Convert center to a QPointF for better precision
        exact_center = QPointF(center)
                
        # Apply zoom and position to other views
        if source_view != self.original_view and hasattr(self.original_view, 'pixmap_item') and self.original_view.pixmap_item:
            self.original_view.setZoomAndPosition(zoom, exact_center)
                
        if source_view != self.processed_view and hasattr(self.processed_view, 'pixmap_item') and self.processed_view.pixmap_item:
            self.processed_view.setZoomAndPosition(zoom, exact_center)
                
        if source_view != self.split_view:
            if isinstance(self.split_view, ZoomableImageView) and self.split_view.pixmap_item:

                if source_view == self.original_view:
                    orig_rect = self.original_view.pixmap_item.boundingRect()
                    split_rect = self.split_view.pixmap_item.boundingRect()
                    
                    x_ratio = center.x() / orig_rect.width()
                    adjusted_center = QPointF(x_ratio * split_rect.width() / 2, center.y())
                    self.split_view.setZoomAndPosition(zoom, adjusted_center)
                else:
                    self.split_view.setZoomAndPosition(zoom, exact_center)
            elif isinstance(self.split_view, DraggableComparisonView) and self.split_view.original_pixmap:
                self.split_view.setZoomAndPosition(zoom, exact_center)

    def sync_views(self):
        """Synchronize zoom and position across all image views"""
        current_tab = self.tab_widget.currentIndex()
        
        if current_tab == 0:
            if not self.original_view.pixmap_item:
                return
                
            zoom, center = self.original_view.getZoomAndPosition()
            
            if self.processed_view.pixmap_item:
                self.processed_view.setZoomAndPosition(zoom, center)
            if isinstance(self.split_view, ZoomableImageView):
                if self.split_view.pixmap_item:
                    self.split_view.setZoomAndPosition(zoom, center)
            elif isinstance(self.split_view, DraggableComparisonView):
                if self.split_view.original_pixmap:
                    self.split_view.setZoomAndPosition(zoom, center)
                
        elif current_tab == 1:  # Processed view is active
            if not self.processed_view.pixmap_item:
                return
                
            zoom, center = self.processed_view.getZoomAndPosition()
            
            if self.original_view.pixmap_item:
                self.original_view.setZoomAndPosition(zoom, center)
            if isinstance(self.split_view, ZoomableImageView):
                if self.split_view.pixmap_item:
                    self.split_view.setZoomAndPosition(zoom, center)
            elif isinstance(self.split_view, DraggableComparisonView):
                if self.split_view.original_pixmap:
                    self.split_view.setZoomAndPosition(zoom, center)
                
        elif current_tab == 2:  # Split view is active
            if isinstance(self.split_view, ZoomableImageView):
                if not self.split_view.pixmap_item:
                    return
            elif isinstance(self.split_view, DraggableComparisonView):
                if not self.split_view.original_pixmap:
                    return
                
            zoom, center = self.split_view.getZoomAndPosition()
            
            if self.original_view.pixmap_item:
                self.original_view.setZoomAndPosition(zoom, center)
            if self.processed_view.pixmap_item:
                self.processed_view.setZoomAndPosition(zoom, center)
        
    def refresh_models(self):
        """Scan for models and update UI"""
        self.sharpener.load_available_models()
        model_types = self.sharpener.get_model_types()
        
        # Update model type dropdown
        current_type = self.model_combo.currentText()
        self.model_combo.clear()
        if model_types:
            self.model_combo.addItems(model_types)
            if current_type in model_types:
                self.model_combo.setCurrentText(current_type)
        
        # If a model is found, load the first one
        if model_types and self.model_combo.count() > 0:
            self.on_model_changed(self.model_combo.currentText())
        
    def update_strength_label(self):
        """Update the strength value label based on slider value"""
        strength = self.strength_slider.value() / 100.0
        self.strength_value.setText(f"{strength:.1f}×")
        
    def on_model_changed(self, model_type):
        """Handle model selection change"""
        if not model_type:
            return
            
        for name, info in self.sharpener.models.items():
            if info['type'] == model_type:
                self.statusBar().showMessage(f"Loading model: {name}...")
                success = self.sharpener.load_model(name)
                if success:
                    self.statusBar().showMessage(f"Model {name} loaded")
                else:
                    self.statusBar().showMessage(f"Failed to load model {name}")
                break
    
    def load_image(self):
        """Open file dialog to select an image"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.tif *.tiff *.fits *.fit)"
        )
        
        if file_path:
            self.input_path = file_path
            self.input_path_label.setText(os.path.basename(file_path))
            
            try:
                preview_image = load_image_preview(file_path)
                self.input_image_data = preview_image
                
                qimage = array_to_qimage(preview_image)
                pixmap = QPixmap.fromImage(qimage)
                
                self.original_view.setPixmap(pixmap)
                
                self.processed_image_data = None
                self.processed_view.setText("Process an image to see results")
                self.split_view.setText("Process an image to compare")
                
                self.save_button.setEnabled(False)
                
                if self.sharpener.current_model:
                    self.process_button.setEnabled(True)
                    
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load image: {str(e)}")
    
    def process_image(self):
        """Process the loaded image with the current settings"""
        if not hasattr(self, 'input_path') or not self.sharpener.current_model:
            return
            
        try:
            strength = self.strength_slider.value() / 100.0
        except (RuntimeError, AttributeError):
            strength = 1.0
            self.statusBar().showMessage("Warning: Slider error, using default strength 1.0")
        
        try:
            preserve_background = self.preserve_bg_checkbox.isChecked()
        except (RuntimeError, AttributeError):
            preserve_background = True
            self.statusBar().showMessage("Warning: Checkbox error, using default preserve background")
        
        try:
            tile_size = int(self.tile_size_combo.currentText())
        except (RuntimeError, ValueError, AttributeError):
            tile_size = 256
            self.statusBar().showMessage("Warning: Tile size error, using default 256")
        
        try:
            overlap = int(self.overlap_combo.currentText())
        except (RuntimeError, ValueError, AttributeError):
            overlap = 96
            self.statusBar().showMessage("Warning: Overlap error, using default 96")
        
        try:
            use_gpu = self.use_gpu_checkbox.isChecked()
        except (RuntimeError, AttributeError):
            use_gpu = True
            self.statusBar().showMessage("Warning: GPU checkbox error, using default GPU")

        params = {
            'strength': strength,
            'preserve_background': preserve_background,
            'tile_size': tile_size,
            'overlap': overlap,
            'use_gpu': use_gpu
        }
        
        # Change button to abort mode
        self.process_button.setText("Abort Processing")
        self.process_button.clicked.disconnect()  # Remove old connection
        self.process_button.clicked.connect(self.abort_processing)
        
        try:
            self.progress_bar.setValue(0)
        except (RuntimeError, AttributeError):
            pass
        
        # Create temporary output path
        output_dir = "temp"
        os.makedirs(output_dir, exist_ok=True)
        temp_output_path = os.path.join(output_dir, "temp_output.tif")
        
        self.worker = ProcessingWorker(
            self.sharpener, self.input_path, temp_output_path, params
        )
        self.worker.progress.connect(self.update_progress)
        self.worker.status.connect(self.update_status)
        self.worker.finished.connect(self.processing_finished)
        self.worker.start()

    def abort_processing(self):
        """Abort the current processing"""
        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.abort()
            self.statusBar().showMessage("Abort requested - please wait...")
        
    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)
        
    def update_status(self, message):
        """Update status bar"""
        self.statusBar().showMessage(message)
        
    def processing_finished(self, success, message, output_path=None):
        """Handle processing completion"""
        # Reset button to normal mode
        self.process_button.setText("Process Image")
        self.process_button.clicked.disconnect()  # Remove abort connection
        self.process_button.clicked.connect(self.process_image)  # Restore normal connection
        self.process_button.setEnabled(True)
        
        self.statusBar().showMessage(message)
        
        if success and output_path and os.path.exists(output_path):
            try:
                processed_image = load_image_preview(output_path)
                self.processed_image_data = processed_image
                
                qimage = array_to_qimage(processed_image)
                pixmap = QPixmap.fromImage(qimage)
                
                self.processed_view.setPixmap(pixmap)
                self.tab_widget.setCurrentIndex(1) # Switch to processed tab
                
                # Create before/after view
                if self.input_image_data is not None:
                    self.create_comparison_view()
                
                # Enable save button
                self.save_button.setEnabled(True)
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to display processed image: {str(e)}")
    
    def create_comparison_view(self):
        """Create a comparison view with draggable divider"""
        if self.input_image_data is None or self.processed_image_data is None:
            return
            
        # Convert to QImage for display
        original_qimage = array_to_qimage(self.input_image_data)
        processed_qimage = array_to_qimage(self.processed_image_data)
        
        original_pixmap = QPixmap.fromImage(original_qimage)
        processed_pixmap = QPixmap.fromImage(processed_qimage)
        
        self.split_view.setImages(original_pixmap, processed_pixmap)
        
        self.sync_views()
    
    def save_processed_image(self):
        """Save the processed image to a file"""
        if self.processed_image_data is None:
            return
                    
        base_name = os.path.splitext(os.path.basename(self.input_path))[0]
        default_output = f"{base_name}_sharpened.tif"
        output_path, _ = QFileDialog.getSaveFileName(
            self, "Save Processed Image", default_output, "TIFF Files (*.tif)"
        )
        
        if not output_path:
            return
                    
        try:
            from utils import save_image
            save_image(output_path, self.processed_image_data, compression="lzw")
            self.statusBar().showMessage(f"Saved to {output_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save image: {str(e)}")
    
    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(self, "About AstroSharpener", 
                        "AstroSharpener v1.0\n\n"
                        "A deep learning-based tool for enhancing astronomical images.\n\n"
                        "Created with TensorFlow and PyQt5.\n\n"
                        "Image Navigation:\n"
                        "• Mouse wheel to zoom in/out at cursor position\n"
                        "• Click and drag to pan the image\n"
                        "• Ctrl+Click to reset zoom\n"
                        "• Right-click for zoom options")

def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    
    app.setStyle('Fusion') 
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()