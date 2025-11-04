import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence
import logging

logger = logging.getLogger(__name__)

class PneumoniaDataset:
    """Dataset handler for pneumonia detection"""
    
    def __init__(self,
                 labels_path: Path,
                 images_dir: Path,
                 preprocessor,
                 augmentor: Optional = None,
                 balance_classes: bool = True):
        """
        Args:
            labels_path: Path to labels CSV
            images_dir: Directory containing images
            preprocessor: ImagePreprocessor instance
            augmentor: DataAugmentor instance (for training)
            balance_classes: Whether to balance pneumonia/non-pneumonia
        """
        self.labels_path = labels_path
        self.images_dir = images_dir
        self.preprocessor = preprocessor
        self.augmentor = augmentor
        self.balance_classes = balance_classes
        
        # Load and process labels
        self.df = self._load_labels()
        self._process_labels()
        
        logger.info(f"Loaded {len(self.df)} samples")
        logger.info(f"Pneumonia cases: {self.df['Pneumonia'].sum()}")
        logger.info(f"Non-pneumonia cases: {(~self.df['Pneumonia']).sum()}")
    
    def _load_labels(self) -> pd.DataFrame:
        """Load labels from CSV"""
        df = pd.read_csv(self.labels_path)
        return df
    
    def _process_labels(self):
        """Process labels to create binary pneumonia labels"""
        # Create binary pneumonia label
        self.df['Pneumonia'] = self.df['Finding Labels'].str.contains('Pneumonia', case=False, na=False).astype(int)
        
        # Extract other useful info
        self.df['Age'] = self.df['Patient Age'].str.extract('(\d+)').astype(float)
        self.df['Gender_Binary'] = (self.df['Patient Gender'] == 'M').astype(int)
        
        # Build image path mapping for efficient lookup
        self._build_image_path_mapping()
        
        # Create image paths using the mapping
        self.df['Image_Path'] = self.df['Image Index'].map(self._image_path_map)
        
        # Remove rows where image path is None (file not found)
        original_count = len(self.df)
        self.df = self.df.dropna(subset=['Image_Path'])
        removed_count = original_count - len(self.df)
        
        if removed_count > 0:
            logger.warning(f"Removed {removed_count} samples with missing image files")
        
        if self.balance_classes:
            self._balance_dataset()
    
    def _build_image_path_mapping(self):
        """Build mapping of image filenames to their full paths"""
        self._image_path_map = {}
        
        # Determine directories to search
        search_dirs = []
        
        if self.images_dir.exists():
            # Flat structure (e.g., all images in data/raw/images/)
            search_dirs.append(self.images_dir)
            logger.info(f"Searching for images in: {self.images_dir}")
        else:
            # Kaggle NIH dataset structure (images_001/images/, images_002/images/, etc.)
            search_dirs = list(self.images_dir.parent.glob('images_*/images'))
            if search_dirs:
                logger.info(f"Found {len(search_dirs)} image subdirectories")
            else:
                logger.warning(f"No images found. Checked: {self.images_dir} and {self.images_dir.parent}/images_*/images")
        
        # Build mapping from all search directories
        for search_dir in search_dirs:
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.dcm']:
                for img_file in search_dir.glob(ext):
                    self._image_path_map[img_file.name] = img_file
        
        logger.info(f"Mapped {len(self._image_path_map)} image files")
    
    def _balance_dataset(self):
        """Balance positive and negative samples"""
        pos_samples = self.df[self.df['Pneumonia'] == 1]
        neg_samples = self.df[self.df['Pneumonia'] == 0]
        
        # Undersample majority class
        n_pos = len(pos_samples)
        n_neg = len(neg_samples)
        
        if n_neg > n_pos:
            # Sample up to 5x positive samples, but no more than available
            n_sample = min(n_neg, n_pos * 5)
            neg_samples = neg_samples.sample(n=n_sample, random_state=42)
            logger.info(f"Balanced dataset: {n_pos} pos, {len(neg_samples)} neg")
        
        self.df = pd.concat([pos_samples, neg_samples]).sample(frac=1, random_state=42)
    
    def split(self, 
              test_size: float = 0.2, 
              val_size: float = 0.1,
              stratify: bool = True) -> Tuple['PneumoniaDataset', 'PneumoniaDataset', 'PneumoniaDataset']:
        """
        Split dataset into train/val/test
        
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        stratify_col = self.df['Pneumonia'] if stratify else None
        
        # First split: train+val / test
        train_val_df, test_df = train_test_split(self.df,
                                                 test_size=test_size,
                                                 stratify=stratify_col,
                                                 random_state=42)
        
        # Second split: train / val
        val_size_adjusted = val_size / (1 - test_size)
        stratify_col_train = train_val_df['Pneumonia'] if stratify else None
        
        train_df, val_df = train_test_split(train_val_df,
                                            test_size=val_size_adjusted,
                                            stratify=stratify_col_train,
                                            random_state=42)
        
        # Create new dataset objects
        train_ds = PneumoniaDataset.__new__(PneumoniaDataset)
        train_ds.df = train_df
        train_ds.preprocessor = self.preprocessor
        train_ds.augmentor = self.augmentor
        
        val_ds = PneumoniaDataset.__new__(PneumoniaDataset)
        val_ds.df = val_df
        val_ds.preprocessor = self.preprocessor
        val_ds.augmentor = None  # No augmentation for validation
        
        test_ds = PneumoniaDataset.__new__(PneumoniaDataset)
        test_ds.df = test_df
        test_ds.preprocessor = self.preprocessor
        test_ds.augmentor = None  # No augmentation for test
        
        logger.info(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_ds, val_ds, test_ds
    
    def get_generator(self, batch_size: int = 32, shuffle: bool = True):
        """Get data generator for training/validation"""
        return PneumoniaGenerator(self.df,
                                  self.preprocessor,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  augmentor=self.augmentor)

class PneumoniaGenerator(Sequence):
    """Keras Sequence generator for pneumonia dataset"""
    
    def __init__(self, 
                 df: pd.DataFrame,
                 preprocessor,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 augmentor: Optional = None):
        self.df = df.reset_index(drop=True)
        self.preprocessor = preprocessor
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augmentor = augmentor
        self.indices = np.arange(len(self.df))
        
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    @property
    def labels(self):
        """Get all labels in the correct order"""
        return self.df.iloc[self.indices]['Pneumonia'].values
    
    def __len__(self):
        """Number of batches per epoch"""
        return int(np.ceil(len(self.df) / self.batch_size))
    
    def __getitem__(self, idx):
        """Get batch at index"""
        batch_indices = self.indices[
            idx * self.batch_size:(idx + 1) * self.batch_size
        ]
        batch_df = self.df.iloc[batch_indices]
        
        # Load and preprocess images
        images = []
        labels = []
        
        for _, row in batch_df.iterrows():
            img = self._load_image(row['Image_Path'])
            if img is not None:
                img = self.preprocessor.preprocess(img)
                images.append(img)
                labels.append(row['Pneumonia'])
        
        X = np.array(images)
        y = np.array(labels)
        
        return X, y
    
    def _load_image(self, path: Path) -> Optional[np.ndarray]:
        """Load image from path"""
        try:
            from skimage import io
            return io.imread(str(path))
        except Exception as e:
            logger.error(f"Error loading image {path}: {e}")
            return None
    
    def on_epoch_end(self):
        """Shuffle indices after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indices)
