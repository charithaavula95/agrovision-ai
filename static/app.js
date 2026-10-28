const fileInput = document.querySelector("#leaf-upload");
const dropZone = document.querySelector("#drop-zone");
const preview = document.querySelector("#image-preview");
const fileName = document.querySelector("#file-name");
const message = document.querySelector("#form-message");
const analyzeButton = document.querySelector("#analyze-button");
const buttonLabel = document.querySelector("#button-label");
const resultCard = document.querySelector("#result-card");

let selectedFile = null;
let previewUrl = null;

function showMessage(text, kind) {
  message.textContent = text;
  message.className = `form-message ${kind}`;
  message.hidden = false;
}

function setSelectedFile(file) {
  if (!file) return;
  const isSupported = ["image/jpeg", "image/png"].includes(file.type);
  if (!isSupported) {
    showMessage("Please choose a JPG or PNG image.", "error");
    return;
  }
  if (file.size > 8 * 1024 * 1024) {
    showMessage("This image is larger than 8 MB. Choose a smaller photo.", "error");
    return;
  }

  selectedFile = file;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  preview.src = previewUrl;
  preview.hidden = false;
  dropZone.classList.add("has-image");
  fileName.textContent = file.name;
  message.hidden = true;
  resultCard.hidden = true;
  analyzeButton.disabled = false;
  buttonLabel.textContent = "Check this leaf";
}

fileInput.addEventListener("change", () => setSelectedFile(fileInput.files[0]));

dropZone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    fileInput.click();
  }
});

for (const eventName of ["dragenter", "dragover"]) {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("is-dragging");
  });
}

for (const eventName of ["dragleave", "drop"]) {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("is-dragging");
  });
}

dropZone.addEventListener("drop", (event) => {
  setSelectedFile(event.dataTransfer.files[0]);
});

function renderResult(data) {
  const prediction = data.prediction;
  const content = resultCard.querySelector(".result-content");
  content.replaceChildren();

  const classification = document.createElement("span");
  classification.className = `result-tag ${prediction.uncertain ? "is-uncertain" : ""}`;
  classification.textContent = prediction.uncertain
    ? "Needs a closer look"
    : prediction.classification === "healthy"
      ? "Looks healthy to the model"
      : "Possible issue detected";

  const title = document.createElement("h3");
  title.className = "result-title";
  title.textContent = prediction.class_name.replaceAll("___", " — ").replaceAll("_", " ");

  const confidence = document.createElement("p");
  confidence.className = "result-confidence";
  confidence.textContent = `${prediction.confidence_percent}% model confidence`;

  const bar = document.createElement("div");
  bar.className = "confidence-track";
  const fill = document.createElement("span");
  fill.style.width = `${Math.max(0, Math.min(100, prediction.confidence_percent))}%`;
  bar.append(fill);

  content.append(classification, title, confidence, bar);

  if (data.guide) {
    const guideTitle = document.createElement("h4");
    guideTitle.className = "guide-heading";
    guideTitle.textContent = "What to look for";
    const symptoms = document.createElement("p");
    symptoms.className = "guide-copy";
    symptoms.textContent = data.guide.symptoms;
    const nextStepsTitle = document.createElement("h4");
    nextStepsTitle.className = "guide-heading";
    nextStepsTitle.textContent = "A possible next step";
    const nextSteps = document.createElement("p");
    nextSteps.className = "guide-copy";
    nextSteps.textContent = data.guide.next_steps;
    content.append(guideTitle, symptoms, nextStepsTitle, nextSteps);
  } else {
    const note = document.createElement("p");
    note.className = "guide-copy";
    note.textContent = "The model was not confident enough to show a disease guide. Consider taking another clear photo and asking a local expert.";
    content.append(note);
  }

  resultCard.hidden = false;
}

analyzeButton.addEventListener("click", async () => {
  if (!selectedFile || analyzeButton.disabled) return;

  const formData = new FormData();
  formData.append("image", selectedFile);
  analyzeButton.disabled = true;
  analyzeButton.classList.add("is-loading");
  buttonLabel.textContent = "Checking your photo…";
  message.hidden = true;

  try {
    const response = await fetch("/api/predict", { method: "POST", body: formData });
    const data = await response.json();
    if (!response.ok) {
      const errorText = response.status === 503
        ? "A prediction isn’t available right now. Please try again later."
        : data.error || "We couldn’t check that photo. Please try again.";
      showMessage(errorText, "error");
      return;
    }
    renderResult(data);
    resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch {
    showMessage("Couldn’t reach the API. Check that the local server is running and try again.", "error");
  } finally {
    analyzeButton.disabled = false;
    analyzeButton.classList.remove("is-loading");
    buttonLabel.textContent = "Check this leaf";
  }
});
