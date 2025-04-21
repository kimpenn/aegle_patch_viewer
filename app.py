import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import time
import os

Image.MAX_IMAGE_PIXELS = None

# Usage:
# conda activate aegle_patch_viewer
# streamlit run app.py --server.headless true

mock_flg = False
image_path = "/Users/kuangda/Developer/1-projects/4-codex-analysis/0-phenocycler-penntmc-pipeline/aegle_patch_viewer/data/extended_extracted_channel_image.png"
# image_path = "/Users/kuangda/Developer/1-projects/4-codex-analysis/0-phenocycler-penntmc-pipeline/aegle_patch_viewer/NW_1_Scan1_rgb.png"
# image_path = "/Users/kuangda/Developer/1-projects/4-codex-analysis/0-phenocycler-penntmc-pipeline/aegle_patch_viewer/NW_1_Scan1_dev_rgb.png"

# Key parameters
patch_height = 1440
patch_width = 1920
overlap = 0.1
# patch_height = 5000
# patch_width = 5000
# patch_height = 10000
# patch_width = 10000
# patch_height = 20000
# patch_width = 20000    
# overlap = 0.0

def extend_image(image, patch_height, patch_width, step_height, step_width):
    """
    # Extend the image to ensure full coverage when cropping patches.

    Args:
        image (np.ndarray): The image to extend.
        patch_height (int): Height of each patch.
        patch_width (int): Width of each patch.
        step_height (int): Step size in height.
        step_width (int): Step size in width.

    Returns:
        np.ndarray: Extended image.
    """
    img_height, img_width, _ = image.shape
    pad_height = (
        patch_height - (img_height - patch_height) % step_height
    ) % patch_height
    pad_width = (patch_width - (img_width - patch_width) % step_width) % patch_width

    extended_image = np.pad(
        image,
        ((0, pad_height), (0, pad_width), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    print(f"Extended image shape: {extended_image.shape}")
    return extended_image


# Function to create a mock image
def create_mock_image(width=500, height=500, color=(255, 255, 255)):
    # Create a blank white image
    img = Image.new("RGB", (width, height), color=color)
    # Draw some shapes or patterns (optional)
    draw = ImageDraw.Draw(img)
    for i in range(0, width, 50):
        draw.line((i, 0, i, height), fill=(200, 200, 200))
    for j in range(0, height, 50):
        draw.line((0, j, width, j), fill=(200, 200, 200))
    return img


# Function to generate mock patch mapping
def generate_mock_patches(image_size, patch_size=100, overlap=0):

    width, height = image_size
    patches = {}
    index = 1
    step = patch_size - overlap
    for y in range(0, height, step):
        for x in range(0, width, step):
            x1 = x
            y1 = y
            x2 = min(x + patch_size, width)
            y2 = min(y + patch_size, height)
            patches[index] = (x1, y1, x2, y2)
            index += 1
    return patches


@st.cache_data
def read_image(
    image_path="NW_1_Scan1_rgb.png",
    patch_height=1440,
    patch_width=1920,
    step_height=100,
    step_width=100,
):
    """
    Profile the image reading and processing time.
    On my linux desktop, the time to read a 28800x50400 image is about 140 seconds.
    ```
    image.size: (28800, 50400)
    Time to open image: 0.00 seconds
    image.shape: (50400, 28800, 3)
    Time to transform image to numpy array: 27.86 seconds
    Extended image shape: (50832, 29760, 3)
    (50832, 29760, 3)
    Time to extend image: 137.57 seconds
    Time to convert back to PIL Image: 140.83 seconds
    Total time for read_image: 140.83 seconds
    
    On my macbook, the time to read a 28800x50400 image is about 34 seconds.
    ```
    image.size: (28800, 50400)
    Time to open image: 0.01 seconds
    image.shape: (50400, 28800, 3)
    Time to transform image to numpy array: 32.16 seconds
    Extended image shape: (50832, 29760, 3)
    (50832, 29760, 3)
    Time to extend image: 32.95 seconds
    Time to convert back to PIL Image: 34.72 seconds
    Total time for read_image: 34.72 seconds
    ```
    """

    # Check if the image exists
    if not os.path.exists(image_path):
        print(f"Image not found at: {image_path}")
        return None
    
    start_time = time.time()
    # Step 1: Open the image
    image = Image.open(image_path)
    
    # chop the image to 1/2 along vertical axis
    # image = image.crop((0, 0, image.width, image.height // 2))
    print(f"image.size: {image.size}")
    step_time = time.time()
    print(f"Time to open image: {step_time - start_time:.2f} seconds")

    # transform image to numpy array
    # Step 2: Transform image to numpy array
    image = np.asarray(image)
    print(f"image.shape: {image.shape}")
    step_time = time.time()
    print(
        f"Time to transform image to numpy array: {step_time - start_time:.2f} seconds"
    )

    # Step 3: Extend the image to ensure full coverage
    extended_image_arr = extend_image(
        image, patch_height, patch_width, step_height, step_width
    )
    print(extended_image_arr.shape)
    step_time = time.time()
    print(f"Time to extend image: {step_time - start_time:.2f} seconds")

    # Step 4: Convert back to PIL Image
    extended_image = Image.fromarray(extended_image_arr)
    step_time = time.time()
    print(f"Time to convert back to PIL Image: {step_time - start_time:.2f} seconds")

    total_time = time.time()
    print(f"Total time for read_image: {total_time - start_time:.2f} seconds")

    return extended_image


def generate_patches(
    img_height, img_width, patch_height, patch_width, step_height, step_width
):
    index = 0
    patches = {}
    for y in range(0, img_height - patch_height + 1, step_height):
        for x in range(0, img_width - patch_width + 1, step_width):
            x1 = x
            y1 = y
            x2 = min(x + patch_width, img_width)
            y2 = min(y + patch_height, img_height)
            patches[index] = (x1, y1, x2, y2)
            index += 1
    return patches


def main():
    print("---------- Starting Streamlit app...")
    st.title("Image Patch Visualizer")

    # Define patch size and overlap
    overlap_height = int(patch_height * overlap)
    overlap_width = int(patch_width * overlap)

    # Calculate step size for cropping
    step_height = patch_height - overlap_height
    step_width = patch_width - overlap_width

    # Read and process the image
    if mock_flg:
        original_image = create_mock_image()
        patch_mapping = generate_mock_patches(original_image.size)
    else:
        original_image = read_image(
            image_path, patch_height, patch_width, step_height, step_width
        )
        img_width, img_height = original_image.size
        patch_mapping = generate_patches(
            img_height, img_width, patch_height, patch_width, step_height, step_width
        )

    # Get original image dimensions
    original_width, original_height = original_image.size

    # Scale the image for display purposes
    display_width = 800  # Adjust as needed
    scale_factor = display_width / original_width
    display_height = int(original_height * scale_factor)
    display_image = original_image.resize(
        (display_width, display_height), Image.LANCZOS
    )

    # Update the label to include the range of indices
    min_index = min(patch_mapping.keys())
    max_index = max(patch_mapping.keys())

    # Initialize session state for showing all patches
    if "show_all_patches" not in st.session_state:
        st.session_state.show_all_patches = False
    
    # Initialize session state for clearing selections
    if "clear_selections" not in st.session_state:
        st.session_state.clear_selections = False

    # Create buttons side by side using columns
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Show All Patches"):
            st.session_state.show_all_patches = True
            st.session_state.clear_selections = False
            st.rerun()
    
    with col2:
        if st.button("Reset to Default"):
            st.session_state.clear_selections = True
            st.session_state.show_all_patches = False
            st.rerun()

    # Determine default selection based on session state
    if st.session_state.clear_selections:
        default_selection = [min_index]
    elif st.session_state.show_all_patches:
        default_selection = list(patch_mapping.keys())
    else:
        default_selection = [min_index]

    # Allow the user to select multiple patch indices
    selected_indices = st.multiselect(
        f"Select Patch Indices ({min_index} - {max_index}):",
        options=list(patch_mapping.keys()),
        default=default_selection,
        key="multiselect"
    )

    # Reset the session states if user manually changes selection
    if selected_indices and st.session_state.clear_selections:
        st.session_state.clear_selections = False
    elif selected_indices != list(patch_mapping.keys()) and st.session_state.show_all_patches:
        st.session_state.show_all_patches = False

    if selected_indices:
        # Create a copy of the scaled image to draw on
        image_with_bboxes = display_image.copy()
        draw = ImageDraw.Draw(image_with_bboxes)

        # Prepare a list of colors to cycle through
        colors = ["red", "blue", "yellow", "purple"]

        # Collect patches to display later
        patches_to_display = []

        # Loop over selected indices and draw bounding boxes
        for idx_num, idx in enumerate(selected_indices):
            if idx in patch_mapping:
                bbox = patch_mapping[idx]

                # Scale bbox coordinates for the display image
                scaled_bbox = [int(coord * scale_factor) for coord in bbox]

                # Select color by cycling through the colors list
                color = colors[idx_num % len(colors)]
                draw.rectangle(
                    scaled_bbox, outline=color, width=2
                )  # Reduced width for scaled image

                # Store the patch and its color to display later
                patch = original_image.crop(bbox)
                patches_to_display.append((patch, idx, color))
            else:
                st.error(f"Patch index {idx} not found.")

        # Display the image with all bounding boxes
        st.image(
            image_with_bboxes,
            caption="Selected Patches Location",
            use_column_width=True,
        )

        # Display the patches below the main image in columns
        st.subheader("Selected Patches")
        num_columns = 3  # Number of patches per row
        columns = st.columns(num_columns)

        for idx, (patch, idx_num, color) in enumerate(patches_to_display):
            col = columns[idx % num_columns]
            with col:
                st.image(
                    patch,
                    caption=f"Patch {idx_num} (Color: {color})",
                    width=200,  # Fixed width for each patch
                )
    else:
        st.warning("No patches selected.")


if __name__ == "__main__":
    main()
