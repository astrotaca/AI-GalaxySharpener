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
        self.abort_requested = False
        
    def abort(self):
        """Request processing abort"""
        self.abort_requested = True
        self.status.emit("Aborting processing...")
        
    def run(self):
        try:
            self.status.emit("Starting processing...")
            
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(self.output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            # Process the image with abort checking
            success, _ = self.sharpener.process_image(
                self.input_path, 
                self.output_path,
                strength=self.params['strength'],
                preserve_background=self.params['preserve_background'],
                tile_size=self.params['tile_size'],
                overlap=self.params['overlap'],
                use_gpu=self.params.get('use_gpu', True),
                progress_callback=self.progress_with_abort_check,
                status_callback=self.status_with_abort_check,
                abort_callback=self.check_abort
            )
            
            if self.abort_requested:
                self.finished.emit(False, "Processing aborted by user.", "")
            elif success:
                self.finished.emit(True, "Processing complete!", self.output_path)
            else:
                self.finished.emit(False, "Processing failed.", "")
                
        except Exception as e:
            if self.abort_requested:
                self.finished.emit(False, "Processing aborted.", "")
            else:
                error_trace = traceback.format_exc()
                print(f"Processing error: {error_trace}")
                self.finished.emit(False, f"Error: {str(e)}", "")
    
    def progress_with_abort_check(self, value):
        """Progress callback that checks for abort"""
        if not self.abort_requested:
            self.progress.emit(value)
    
    def status_with_abort_check(self, message):
        """Status callback that checks for abort"""
        if not self.abort_requested:
            self.status.emit(message)
    
    def check_abort(self):
        """Check if abort was requested"""
        return self.abort_requested