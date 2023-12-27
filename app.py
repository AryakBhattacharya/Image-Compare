import streamlit as st
import cv2
import numpy as np

def upload_images():
    st.write("Upload the first image:")
    uploaded_file1 = st.file_uploader("Choose a file")

    st.write("Upload the second image:")
    uploaded_file2 = st.file_uploader("Choose a file")

    return uploaded_file1, uploaded_file2


def compare_images(image1, image2):
    if image1 is None or image2 is None:
        st.error("Please upload both images.")
        return None

    # Read the images in color
    img1_color = cv2.imdecode(np.frombuffer(image1.read(), np.uint8), -1)
    img2_color = cv2.imdecode(np.frombuffer(image2.read(), np.uint8), -1)

    # Convert images to grayscale
    img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)

    # Resize images to the same dimensions (adjust as needed)
    img1 = cv2.resize(img1, (300, 300))
    img2 = cv2.resize(img2, (300, 300))

    # Check if images have the same shape
    if img1.shape != img2.shape:
        st.error("Error: Images must have the same dimensions for comparison.")
        return None

    # Compare grayscale images
    difference = cv2.absdiff(img1, img2)
    similarity = np.sum(difference == 0) / np.prod(img1.shape)

    return similarity

def main():
    st.title("Image Similarity Comparison")
    uploaded_file1, uploaded_file2 = upload_images()

    if st.button("Compare Images"):
        similarity = compare_images(uploaded_file1, uploaded_file2)

        if similarity is not None:
            st.success(f"Similarity between the two images: {similarity:.2%}")
        else:
            st.error("Image comparison failed.")

if __name__ == "__main__":
    main()
