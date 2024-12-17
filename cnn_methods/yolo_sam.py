from ultralytics import SAM, YOLO
from PIL import Image
import matplotlib.pyplot as plt

def load_train_yolo_model() -> YOLO:
    """
    Load and train a YOLO model for object detection.

    This function loads the YOLO model from a pre-trained checkpoint, trains it 
    using the provided dataset, and evaluates its performance on the validation set.

    Parameters:
        None

    Returns:
        YOLO: The trained YOLO model object.
        
    Example:
        model = load_train_yolo_model()
    """

    
    # Load a model for detection
    model = YOLO("yolo11n.pt", task="detection")

    # Train the model
    train_results = model.train(
        data="coco8.yaml",             # path to dataset YAML
        epochs=100,                    # number of training epochs
        imgsz=640,                     # training image size
        device="0",                    # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    )

    # Evaluate model performance on the validation set
    model.val()
    return model

def test_yolo_model(model: YOLO, image_path: str) -> None:
    """
    Test the YOLO model by performing object detection on a given image.

    This function uses the provided YOLO model to detect objects in the specified image 
    and displays the results by showing the image with bounding boxes around detected objects.

    Parameters:
        model (YOLO): The trained YOLO model used for object detection.
        image_path (str): The file path to the image on which object detection will be performed.

    Returns:
        None: This function only displays the image with detection results and does not return any value.

    Example:
        test_yolo_model(model, "path/to/image.jpg")
    """
    # Perform object detection on an image

    results = model(image_path)
    results[0].show()

def display_images(image_path: str, segmented_image_path: str) -> None:
    """
    Display the original and segmented images side by side.

    This function loads two images (the original and the segmented one), 
    and displays them in a side-by-side layout for comparison. The images 
    are shown without axes, and each image is labeled accordingly.

    Parameters:
        image_path (str): The file path to the original image.
        segmented_image_path (str): The file path to the segmented image.

    Returns:
        None: This function only displays the images and does not return any value.

    Example:
        display_images("path/to/original/image.jpg", "path/to/segmented/image.jpg")
    """

    # Load the images
    original_image = Image.open(image_path)
    segmented_image = Image.open(segmented_image_path)
    
    # Create a side-by-side plot
    _, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Display the original image
    ax[0].imshow(original_image)
    ax[0].axis('off')  # Hide the axes
    ax[0].set_title('Original Image')  # Set title

    # Display the segmented image
    ax[1].imshow(segmented_image)
    ax[1].axis('off')  # Hide the axes
    ax[1].set_title('Segmented Image')  # Set title

    # Adjust layout for better appearance
    plt.tight_layout()
    plt.show()

def test_segmentation_sam(image_path: str, image_name: str) -> None:
    """
    Perform segmentation using the SAM model and display the original and segmented images.

    This function loads the SAM (Segment Anything Model) model, performs segmentation 
    on the provided image, and then displays the original and segmented images side by side 
    for comparison. The segmented image is saved, and the function also visualizes both images.

    Parameters:
        image_path (str): The file path to the image to be segmented.
        image_name (str): The name to save the segmented image under.

    Returns:
        None: This function performs segmentation and displays the images but does not return any value.

    Example:
        test_segmentation_sam("path/to/image.jpg", "segmented_image.jpg")
    """
    # Load the SAM model
    sam_model = "sam_b.pt"
    sam_model = SAM(sam_model)

    # Perform segmentation
    sam_results = sam_model(image_path, verbose=True, save=True, device="0", show_boxes=False)

    # Get the path of the saved segmented image
    segmented_image_path = f"{sam_results[0].save_dir}/{image_name}"

    # Display the original and segmented images side by side
    display_images(image_path, segmented_image_path)

def test_segmentation_yolo_sam(image_path: str, image_name: str) -> None:
    """
    Perform segmentation using the YOLO and SAM models, and display the original and segmented images.

    This function loads the SAM (Segment Anything Model) model, performs segmentation 
    on the provided image, and then displays the original and segmented images side by side 
    for comparison. The segmented image is saved, and the function also visualizes both images.

    Parameters:
        image_path (str): The file path to the image to be segmented.
        image_name (str): The name to save the segmented image under.

    Returns:
        None: This function performs segmentation and displays the images but does not return any value.

    Example:
        test_segmentation_yolo_sam("path/to/image.jpg", "segmented_image.jpg")
    """
    # Load the SAM model
    sam_model = "sam_b.pt"
    sam_model = SAM(sam_model)

    # Perform segmentation
    sam_results = sam_model(image_path, verbose=True, save=True, device="0", show_boxes=True)

    # Get the path of the saved segmented image
    segmented_image_path = f"{sam_results[0].save_dir}/{image_name}"

    # Display the original and segmented images side by side
    display_images(image_path, segmented_image_path)

def main():
    # Load the YOLO model
    yolo_model = load_train_yolo_model()

    # Test the YOLO model
    image_path = "images/coches.jpg"
    test_yolo_model(yolo_model, image_path)

    # Test the SAM model
    image_name = "coches.jpg"
    test_segmentation_sam(image_path, image_name)

    # Test the YOLO and SAM models
    test_segmentation_yolo_sam(image_path, image_name)

if __name__ == "__main__":
    main()