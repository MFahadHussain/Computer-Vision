"""Main entry point for YOLO-World V2 Video Detector with Enhanced Multiple Polygon ROI Cropping, Editing, and BoT-SORT Tracking."""

import argparse
import sys
import traceback
from pathlib import Path

from utils.model_checker import check_and_clean_model_files
from utils.logger_config import setup_logger, logger
from detection.detector import YOLOWorldROIDetector


def main():
    """Main entry point with enhanced error handling and model checking."""
    # Step 1: Set up logging
    setup_logger()
    
    # Step 2: Check for corrupted model files
    check_and_clean_model_files()

    try:
        parser = argparse.ArgumentParser(
            description="Enhanced YOLO-World V2 Video Detector with Multiple Polygon ROI Cropping, Editing, and BoT-SORT Tracking",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Basic usage with BoT-SORT tracking
  python main.py -i video.mp4 -o output.mp4
  
  # Without BoT-SORT tracking (use default)
  python main.py -i video.mp4 -o output.mp4 --no-botsort
  
  # With custom classes
  python main.py -i video.mp4 -o output.mp4 -c "person" "car" "dog"
  
  # Without preview window
  python main.py -i video.mp4 -o output.mp4 --no-preview
  
  # Custom confidence threshold
  python main.py -i video.mp4 -o output.mp4 --conf 0.5
            """
        )
        
        parser.add_argument(
            '-i', '--input',
            type=str,
            required=True,
            help='Path to input video file'
        )
        
        parser.add_argument(
            '-o', '--output',
            type=str,
            required=True,
            help='Path for output annotated video'
        )
        
        parser.add_argument(
            '-m', '--model',
            type=str,
            default='yolov8l-worldv2.pt',
            help='Path to YOLO-World model (default: yolov8l-worldv2.pt)'
        )
        
        parser.add_argument(
            '-c', '--classes',
            type=str,
            nargs='+',
            default=None,
            help='Custom classes to detect (e.g., -c "person" "car" "dog")'
        )
        
        parser.add_argument(
            '--conf',
            type=float,
            default=0.3,
            help='Confidence threshold (default: 0.3)'
        )
        
        parser.add_argument(
            '--no-preview',
            action='store_true',
            help='Disable real-time preview window'
        )
        
        parser.add_argument(
            '--no-botsort',
            action='store_true',
            help='Disable BoT-SORT tracking and use default tracking'
        )
        
        args = parser.parse_args()
        
        # Validate input file
        if not Path(args.input).exists():
            print(f"Error: Input file not found: {args.input}")
            return 1
        
        # Create output directory if needed
        output_dir = Path(args.output).parent
        if output_dir and not output_dir.exists():
            output_dir.mkdir(parents=True)
        
        # Initialize detector with BoT-SORT tracking
        try:
            detector = YOLOWorldROIDetector(
                model_path=args.model,
                custom_classes=args.classes,
                confidence_threshold=args.conf,
                use_botsort=not args.no_botsort
            )
        except Exception as e:
            print(f"Error initializing detector: {str(e)}")
            return 1
        
        # Process video with error handling
        return detector.process_video(
            input_path=args.input,
            output_path=args.output,
            show_preview=not args.no_preview,
            custom_classes=args.classes
        )
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        print(f"\nUnexpected error: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())

