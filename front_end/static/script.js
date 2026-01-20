 const imageInput = document.getElementById("imageInput");
  const preview = document.getElementById("preview");
  const result = document.getElementById("result");

  // Show preview image
  imageInput.addEventListener("change", () => {
    const file = imageInput.files[0];
    if (!file) return;

    preview.src = URL.createObjectURL(file);
    preview.classList.remove("d-none");
  });

  async function sendImage(endpoint) {
    const file = imageInput.files[0];
    if (!file) {
      alert("Please select an image first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      //Request to the backend
      const response = await fetch(endpoint, {
        method: "POST",
        body: formData
      });

      if (!response.ok) {
        throw new Error("Server error");
      }

      const blob = await response.blob();
      const imageURL = URL.createObjectURL(blob);

      // Show result image
      result.src = imageURL;
      result.classList.remove("d-none");

      // Download image
      const a = document.createElement("a");
      a.href = imageURL;
      a.download = `transformed_image_${Date.now()}.jpg`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);

    } catch (error) {
      alert("Error processing image");
      console.error(error);
    }
  }