from PyQt5.QtCore import QThread, pyqtSignal
import os
import traceback

class ProcessingWorker(QThread):
    """Worker thread for image processing to keep UI responsive"""
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(bool, str, str)  # success, message, output_path
    
    def __init__(self, sharpener, input_path, output_path, params):
        super().__init__()
        self.sharpener = sharpener
        self.input_path = input_path
        self.output_path = output_path
        self.params = params
        
    def run(self):
        try:
            self.status.emit("Starting processing...")
            
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(self.output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            # Process the image
            success, _ = self.sharpener.process_image(
                self.input_path, 
                self.output_path,
                strength=self.params['strength'],
                preserve_background=self.params['preserve_background'],
                tile_size=self.params['tile_size'],
                overlap=self.params['overlap'],
                progress_callback=self.progress.emit,
                status_callback=self.status.emit
            )
            
            if success:
                self.finished.emit(True, "Processing complete!", self.output_path)
            else:
                self.finished.emit(False, "Processing failed.", "")
                
        except Exception as e:
            error_trace = traceback.format_exc()
            print(f"Processing error: {error_trace}")
            self.finished.emit(False, f"Error: {str(e)}", "")