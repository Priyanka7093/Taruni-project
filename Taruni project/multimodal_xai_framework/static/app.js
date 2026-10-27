document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const presetButtonsContainer = document.getElementById("presetButtons");
    const diagnosticForm = document.getElementById("diagnosticForm");
    const mriInput = document.getElementById("mriInput");
    const dropzone = document.getElementById("dropzone");
    const dropzoneContent = document.getElementById("dropzoneContent");
    const imagePreview = document.getElementById("imagePreview");
    const submitBtn = document.getElementById("submitBtn");

    const emptyState = document.getElementById("emptyState");
    const loadingState = document.getElementById("loadingState");
    const resultsContent = document.getElementById("resultsContent");

    const diagBadge = document.getElementById("diagBadge");
    const confidenceText = document.getElementById("confidenceText");
    const confidenceFill = document.getElementById("confidenceFill");
    const probDistribution = document.getElementById("probDistribution");

    const originalImgDisplay = document.getElementById("originalImgDisplay");
    const heatmapImgDisplay = document.getElementById("heatmapImgDisplay");
    const heatmapToggle = document.getElementById("heatmapToggle");
    const shapList = document.getElementById("shapList");

    let selectedPresetIdx = null;

    // 1. Fetch & Populate Sample Patient Presets
    async function loadPresets() {
        try {
            const res = await fetch("/api/sample_presets");
            const data = await res.json();

            presetButtonsContainer.innerHTML = "";
            data.presets.forEach((preset, index) => {
                const btn = document.createElement("button");
                btn.type = "button";
                btn.className = "preset-btn";
                btn.innerText = preset.name;
                btn.addEventListener("click", () => applyPreset(preset, index));
                presetButtonsContainer.appendChild(btn);
            });
        } catch (err) {
            console.error("Error loading sample presets:", err);
        }
    }

    function applyPreset(preset, idx) {
        selectedPresetIdx = preset.class_idx;

        // Populate inputs
        document.getElementById("age").value = preset.clinical.Age;
        document.getElementById("gender").value = preset.clinical.Gender;
        document.getElementById("mmse").value = preset.clinical.MMSE;
        document.getElementById("cdr").value = preset.clinical.CDR;
        document.getElementById("educ").value = preset.clinical.Educ;
        document.getElementById("faq").value = preset.clinical.FAQ;

        // Reset image input preview
        mriInput.value = "";
        imagePreview.classList.add("hidden");
        dropzoneContent.classList.remove("hidden");

        // Submit form automatically for seamless quick preview
        diagnosticForm.dispatchEvent(new Event("submit"));
    }

    // 2. Drag & Drop File Upload Handling
    mriInput.addEventListener("change", (e) => {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (event) => {
                imagePreview.src = event.target.result;
                imagePreview.classList.remove("hidden");
                dropzoneContent.classList.add("hidden");
            };
            reader.readAsDataURL(file);
            selectedPresetIdx = null;
        }
    });

    // 3. Form Submit Handler
    diagnosticForm.addEventListener("submit", async (e) => {
        e.preventDefault();

        emptyState.classList.add("hidden");
        resultsContent.classList.add("hidden");
        loadingState.classList.remove("hidden");

        const formData = new FormData();

        if (mriInput.files[0]) {
            formData.append("file", mriInput.files[0]);
        }
        if (selectedPresetIdx !== null) {
            formData.append("class_preset", selectedPresetIdx);
        }

        formData.append("age", document.getElementById("age").value);
        formData.append("gender", document.getElementById("gender").value);
        formData.append("mmse", document.getElementById("mmse").value);
        formData.append("cdr", document.getElementById("cdr").value);
        formData.append("educ", document.getElementById("educ").value);
        formData.append("faq", document.getElementById("faq").value);

        try {
            const response = await fetch("/api/predict_and_explain", {
                method: "POST",
                body: formData
            });

            const data = await response.json();

            if (data.success) {
                renderResults(data);
            } else {
                alert("Error running diagnosis: " + data.detail);
            }
        } catch (err) {
            console.error("API call failed:", err);
            alert("Failed to connect to API backend.");
        } finally {
            loadingState.classList.add("hidden");
        }
    });

    // 4. Render Results Function
    function renderResults(data) {
        resultsContent.classList.remove("hidden");

        // Set Diagnosis Badge
        diagBadge.innerText = data.diagnosis;
        diagBadge.className = "diag-badge";

        if (data.diagnosis.includes("Normal")) {
            diagBadge.classList.add("cn");
        } else {
            diagBadge.classList.add("ad");
        }

        // Confidence
        confidenceText.innerText = `${data.confidence}%`;
        confidenceFill.style.width = `${data.confidence}%`;

        // Class Probabilities
        probDistribution.innerHTML = "";
        for (const [className, prob] of Object.entries(data.probabilities)) {
            const probPct = (prob * 100).toFixed(1);
            const probDiv = document.createElement("div");
            probDiv.className = "prob-item";
            probDiv.innerHTML = `
                <span>${className.split(' ')[0]}</span>
                <strong>${probPct}%</strong>
            `;
            probDistribution.appendChild(probDiv);
        }

        // Images (Grad-CAM)
        originalImgDisplay.src = data.original_image;
        heatmapImgDisplay.src = data.heatmap_image;

        // SHAP List
        shapList.innerHTML = "";
        data.attributions.forEach((attr) => {
            const item = document.createElement("div");
            item.className = "shap-item";

            const isPositive = attr.impact >= 0;
            const barWidth = Math.min(Math.abs(attr.impact) * 300, 100);
            const dirLabel = isPositive ? "+ Impact" : "- Impact";

            item.innerHTML = `
                <div class="shap-row-header">
                    <span><strong>${attr.feature}</strong> = ${attr.value}</span>
                    <span style="color: ${isPositive ? 'var(--color-ad)' : 'var(--color-normal)'}">
                        ${dirLabel} (${attr.impact > 0 ? '+' : ''}${attr.impact})
                    </span>
                </div>
                <div class="shap-bar-track">
                    <div class="shap-bar-fill ${isPositive ? 'pos' : 'neg'}" style="width: ${barWidth}%;"></div>
                </div>
                <div class="shap-desc">${attr.description}</div>
            `;
            shapList.appendChild(item);
        });
    }

    // Heatmap Toggle Listener
    heatmapToggle.addEventListener("change", (e) => {
        if (e.target.checked) {
            heatmapImgDisplay.style.opacity = "1";
        } else {
            heatmapImgDisplay.style.opacity = "0";
        }
    });

    // Initialize
    loadPresets();
});