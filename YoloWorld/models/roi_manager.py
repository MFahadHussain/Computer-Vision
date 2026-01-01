"""ROI configuration save/load manager."""

import json
from pathlib import Path
from typing import List, Dict, Optional
import logging

from .data_models import PolygonROI, ROIConfig

logger = logging.getLogger(__name__)


class ROIManager:
    """Manager for saving and loading ROI configurations."""
    
    @staticmethod
    def save_roi_config(roi_config: ROIConfig, filepath: str) -> bool:
        """Save ROI configuration to JSON file.
        
        Args:
            roi_config: ROIConfig object to save
            filepath: Path to save the configuration
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert ROI config to dictionary
            config_dict = {
                'polygons': [
                    {
                        'id': poly.id,
                        'points': poly.points,
                        'min_x': poly.min_x,
                        'min_y': poly.min_y,
                        'width': poly.width,
                        'height': poly.height
                    }
                    for poly in roi_config.polygons
                ]
            }
            
            # Create directory if it doesn't exist
            output_path = Path(filepath)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save to JSON
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            logger.info(f"ROI configuration saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving ROI configuration: {str(e)}")
            return False
    
    @staticmethod
    def load_roi_config(filepath: str) -> Optional[ROIConfig]:
        """Load ROI configuration from JSON file.
        
        Args:
            filepath: Path to the configuration file
            
        Returns:
            ROIConfig object if successful, None otherwise
        """
        try:
            if not Path(filepath).exists():
                logger.warning(f"ROI configuration file not found: {filepath}")
                return None
            
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
            
            # Reconstruct ROI config
            roi_config = ROIConfig()
            
            for poly_dict in config_dict.get('polygons', []):
                polygon = PolygonROI(
                    points=[tuple(p) for p in poly_dict['points']],
                    min_x=poly_dict.get('min_x', 0),
                    min_y=poly_dict.get('min_y', 0),
                    width=poly_dict.get('width', 0),
                    height=poly_dict.get('height', 0),
                    id=poly_dict.get('id', 0)
                )
                polygon.compute_bounds()
                roi_config.add_polygon(polygon)
            
            logger.info(f"ROI configuration loaded from {filepath}")
            return roi_config
            
        except Exception as e:
            logger.error(f"Error loading ROI configuration: {str(e)}")
            return None


