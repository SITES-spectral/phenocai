"""Heuristic-based analysis methods for initial classification and quality assessment."""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Any
import cv2


@dataclass
class SnowDetectionResult:
    """Result of snow detection analysis."""
    has_snow: bool
    confidence: float
    snow_percentage: float
    bright_pixels: int
    total_pixels: int
    mean_brightness: float
    mean_saturation: float


@dataclass
class QualityIssue:
    """Detected quality issue in image."""
    issue_type: str
    severity: float  # 0.0 to 1.0
    description: str
    affected_area: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h


class SnowDetector:
    """HSV-based snow detection using heuristic thresholds."""
    
    def __init__(
        self,
        brightness_threshold: int = 180,
        saturation_threshold: int = 30,
        min_snow_percentage: float = 0.1
    ):
        """Initialize snow detector with thresholds.
        
        Args:
            brightness_threshold: Minimum V value for snow pixels (0-255)
            saturation_threshold: Maximum S value for snow pixels (0-255)
            min_snow_percentage: Minimum percentage of image to be snow
        """
        self.brightness_threshold = brightness_threshold
        self.saturation_threshold = saturation_threshold
        self.min_snow_percentage = min_snow_percentage
    
    def detect(self, image: np.ndarray, roi_mask: Optional[np.ndarray] = None) -> SnowDetectionResult:
        """Detect snow in image using HSV thresholds.
        
        Args:
            image: RGB image as numpy array
            roi_mask: Optional binary mask for region of interest
            
        Returns:
            SnowDetectionResult with detection details
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        
        # Apply ROI mask if provided
        if roi_mask is not None:
            s = cv2.bitwise_and(s, s, mask=roi_mask)
            v = cv2.bitwise_and(v, v, mask=roi_mask)
            total_pixels = np.sum(roi_mask > 0)
        else:
            total_pixels = s.shape[0] * s.shape[1]
        
        # Snow detection criteria: high brightness AND low saturation
        snow_mask = (v >= self.brightness_threshold) & (s <= self.saturation_threshold)
        
        # Calculate statistics
        bright_pixels = np.sum(snow_mask)
        snow_percentage = bright_pixels / total_pixels if total_pixels > 0 else 0.0
        
        # Calculate mean values in ROI
        if roi_mask is not None:
            roi_pixels = roi_mask > 0
            mean_brightness = np.mean(v[roi_pixels]) if np.any(roi_pixels) else 0.0
            mean_saturation = np.mean(s[roi_pixels]) if np.any(roi_pixels) else 0.0
        else:
            mean_brightness = np.mean(v)
            mean_saturation = np.mean(s)
        
        # Determine if snow is present
        has_snow = snow_percentage >= self.min_snow_percentage
        
        # Calculate confidence based on how well pixels match snow criteria
        if bright_pixels > 0:
            # Average deviation from ideal snow values (V=255, S=0)
            snow_pixel_indices = np.where(snow_mask)
            v_deviation = 1.0 - (np.mean(v[snow_pixel_indices]) / 255.0)
            s_deviation = np.mean(s[snow_pixel_indices]) / 255.0
            confidence = 1.0 - (v_deviation + s_deviation) / 2.0
        else:
            confidence = 0.0
        
        return SnowDetectionResult(
            has_snow=has_snow,
            confidence=confidence,
            snow_percentage=snow_percentage,
            bright_pixels=int(bright_pixels),
            total_pixels=int(total_pixels),
            mean_brightness=float(mean_brightness),
            mean_saturation=float(mean_saturation)
        )
    
    def detect_with_visualization(
        self, 
        image: np.ndarray, 
        roi_mask: Optional[np.ndarray] = None
    ) -> Tuple[SnowDetectionResult, np.ndarray]:
        """Detect snow and create visualization.
        
        Args:
            image: RGB image as numpy array
            roi_mask: Optional binary mask for region of interest
            
        Returns:
            Tuple of (SnowDetectionResult, visualization_image)
        """
        result = self.detect(image, roi_mask)
        
        # Create visualization
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        
        # Create snow mask
        snow_mask = (v >= self.brightness_threshold) & (s <= self.saturation_threshold)
        if roi_mask is not None:
            snow_mask = snow_mask & (roi_mask > 0)
        
        # Create colored overlay
        overlay = image.copy()
        overlay[snow_mask] = [0, 255, 255]  # Cyan for snow pixels
        
        # Blend with original
        visualization = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
        
        # Add text overlay
        text_color = (0, 255, 0) if result.has_snow else (255, 0, 0)
        cv2.putText(
            visualization,
            f"Snow: {result.snow_percentage:.1%} (Conf: {result.confidence:.2f})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            text_color,
            2
        )
        
        return result, visualization


class QualityAssessment:
    """Assess image quality issues using heuristic methods."""
    
    # Quality issue thresholds
    DARKNESS_THRESHOLD = 50  # Mean brightness below this is too dark
    BRIGHTNESS_THRESHOLD = 200  # Mean brightness above this is too bright
    BLUR_THRESHOLD = 50  # Laplacian variance below this indicates blur
    LOW_CONTRAST_THRESHOLD = 30  # Std dev below this indicates low contrast
    
    def __init__(self):
        """Initialize quality assessment."""
        pass
    
    def assess(self, image: np.ndarray, roi_mask: Optional[np.ndarray] = None) -> List[QualityIssue]:
        """Assess image quality issues.
        
        Args:
            image: RGB image as numpy array
            roi_mask: Optional binary mask for region of interest
            
        Returns:
            List of detected quality issues
        """
        issues = []
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply ROI mask if provided
        if roi_mask is not None:
            analysis_region = gray[roi_mask > 0]
        else:
            analysis_region = gray.flatten()
        
        # Check brightness issues
        mean_brightness = np.mean(analysis_region)
        if mean_brightness < self.DARKNESS_THRESHOLD:
            issues.append(QualityIssue(
                issue_type="darkness",
                severity=1.0 - (mean_brightness / self.DARKNESS_THRESHOLD),
                description=f"Image too dark (mean brightness: {mean_brightness:.1f})"
            ))
        elif mean_brightness > self.BRIGHTNESS_THRESHOLD:
            issues.append(QualityIssue(
                issue_type="high_brightness",
                severity=(mean_brightness - self.BRIGHTNESS_THRESHOLD) / (255 - self.BRIGHTNESS_THRESHOLD),
                description=f"Image too bright (mean brightness: {mean_brightness:.1f})"
            ))
        
        # Check contrast
        std_dev = np.std(analysis_region)
        if std_dev < self.LOW_CONTRAST_THRESHOLD:
            issues.append(QualityIssue(
                issue_type="low_contrast",
                severity=1.0 - (std_dev / self.LOW_CONTRAST_THRESHOLD),
                description=f"Low contrast (std dev: {std_dev:.1f})"
            ))
        
        # Check blur using Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        if roi_mask is not None:
            laplacian_roi = laplacian[roi_mask > 0]
            blur_metric = np.var(laplacian_roi)
        else:
            blur_metric = np.var(laplacian)
        
        if blur_metric < self.BLUR_THRESHOLD:
            issues.append(QualityIssue(
                issue_type="blur",
                severity=1.0 - (blur_metric / self.BLUR_THRESHOLD),
                description=f"Image appears blurry (variance: {blur_metric:.1f})"
            ))
        
        # Check for lens artifacts (simplified)
        # Look for circular dark regions that might indicate lens issues
        edges = cv2.Canny(gray, 50, 150)
        circles = cv2.HoughCircles(
            edges,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=100,
            param1=50,
            param2=30,
            minRadius=50,
            maxRadius=200
        )
        
        if circles is not None and len(circles[0]) > 0:
            for circle in circles[0]:
                x, y, r = map(int, circle)
                # Check if circle region is significantly darker
                mask = np.zeros_like(gray)
                cv2.circle(mask, (x, y), r, 255, -1)
                circle_mean = np.mean(gray[mask > 0])
                
                if circle_mean < mean_brightness * 0.7:  # 30% darker than average
                    issues.append(QualityIssue(
                        issue_type="lens_artifact",
                        severity=0.5,
                        description="Possible lens droplet or obstruction",
                        affected_area=(x-r, y-r, 2*r, 2*r)
                    ))
                    break  # Only report once
        
        return issues
    
    def assess_batch(
        self, 
        images: List[np.ndarray], 
        roi_masks: Optional[List[np.ndarray]] = None
    ) -> List[List[QualityIssue]]:
        """Assess quality issues for batch of images.
        
        Args:
            images: List of RGB images as numpy arrays
            roi_masks: Optional list of binary masks
            
        Returns:
            List of quality issues for each image
        """
        if roi_masks is None:
            roi_masks = [None] * len(images)
        
        return [
            self.assess(img, mask) 
            for img, mask in zip(images, roi_masks)
        ]
    
    def summarize_issues(self, issues: List[QualityIssue]) -> Dict[str, Any]:
        """Summarize quality issues into a simple report.
        
        Args:
            issues: List of quality issues
            
        Returns:
            Dictionary with summary statistics
        """
        if not issues:
            return {
                "has_issues": False,
                "issue_count": 0,
                "severity_score": 0.0,
                "issues": []
            }
        
        issue_types = [issue.issue_type for issue in issues]
        max_severity = max(issue.severity for issue in issues)
        avg_severity = sum(issue.severity for issue in issues) / len(issues)
        
        return {
            "has_issues": True,
            "issue_count": len(issues),
            "severity_score": avg_severity,
            "max_severity": max_severity,
            "issues": [
                {
                    "type": issue.issue_type,
                    "severity": issue.severity,
                    "description": issue.description
                }
                for issue in issues
            ],
            "summary": {
                "darkness": sum(1 for t in issue_types if t == "darkness"),
                "high_brightness": sum(1 for t in issue_types if t == "high_brightness"),
                "blur": sum(1 for t in issue_types if t == "blur"),
                "low_contrast": sum(1 for t in issue_types if t == "low_contrast"),
                "lens_artifact": sum(1 for t in issue_types if t == "lens_artifact")
            }
        }