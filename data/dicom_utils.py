import pydicom
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class DICOMValidator:
    """Validates DICOM files for pneumonia detection"""
    
    def __init__(self, 
                 allowed_modality: str = "DX",
                 allowed_body_part: str = "CHEST",
                 allowed_positions: list = None):
        self.allowed_modality = allowed_modality
        self.allowed_body_part = allowed_body_part
        self.allowed_positions = allowed_positions or ["PA", "AP"]
    
    def validate(self, dicom_path: Path) -> Tuple[bool, Optional[str]]:
        """
        Validate DICOM file
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            dcm = pydicom.dcmread(str(dicom_path))
            
            # Check modality
            if dcm.Modality != self.allowed_modality:
                return False, f"Invalid modality: {dcm.Modality}, expected {self.allowed_modality}"
            
            # Check body part
            if dcm.BodyPartExamined != self.allowed_body_part:
                return False, f"Invalid body part: {dcm.BodyPartExamined}, expected {self.allowed_body_part}"
            
            # Check patient position
            if dcm.PatientPosition not in self.allowed_positions:
                return False, f"Invalid position: {dcm.PatientPosition}, expected one of {self.allowed_positions}"
            
            return True, None
            
        except Exception as e:
            return False, f"Error reading DICOM: {str(e)}"

def read_dicom(dicom_path: Path, 
               validator: Optional[DICOMValidator] = None) -> Tuple[Optional[np.ndarray], Optional[pydicom.dataset.Dataset]]:
    """
    Read and validate DICOM file
    
    Args:
        dicom_path: Path to DICOM file
        validator: DICOMValidator instance
        
    Returns:
        Tuple of (pixel_array, dicom_dataset) or (None, None) if invalid
    """
    if validator is None:
        validator = DICOMValidator()
    
    is_valid, error_msg = validator.validate(dicom_path)
    
    if not is_valid:
        logger.warning(f"File {dicom_path} contains invalid data: {error_msg}")
        return None, None
    
    dcm = pydicom.dcmread(str(dicom_path))
    return dcm.pixel_array, dcm

def extract_metadata(dcm: pydicom.dataset.Dataset) -> dict:
    """Extract relevant metadata from DICOM"""
    metadata = {
        'patient_id': getattr(dcm, 'PatientID', None),
        'patient_age': getattr(dcm, 'PatientAge', None),
        'patient_gender': getattr(dcm, 'PatientSex', None),
        'view_position': getattr(dcm, 'PatientPosition', None),
        'modality': getattr(dcm, 'Modality', None),
        'body_part': getattr(dcm, 'BodyPartExamined', None),
        'study_description': getattr(dcm, 'StudyDescription', None),
    }
    return metadata
