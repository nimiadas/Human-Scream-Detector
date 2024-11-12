let audioBlob = null;

document.getElementById("load-button").addEventListener("click", function() {
    document.getElementById("file-input").click();
});

document.getElementById("file-input").addEventListener("change", function(event) {
    audioBlob = event.target.files[0];
});

document.getElementById("record-button").addEventListener("click", async function() {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mediaRecorder = new MediaRecorder(stream);
    let audioChunks = [];
    
    mediaRecorder.ondataavailable = event => audioChunks.push(event.data);
    mediaRecorder.onstop = () => audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
    
    mediaRecorder.start();
    setTimeout(() => mediaRecorder.stop(), 3000);  // Record for 3 seconds
});

document.getElementById("predict-button").addEventListener("click", function() {
    if (!audioBlob) {
        alert("Please load or record an audio file first.");
        return;
    }

    const formData = new FormData();
    formData.append("file", audioBlob);

    fetch("/predict", { method: "POST", body: formData })
        .then(response => response.json())
        .then(data => {
            document.getElementById("result-message").textContent = data.result;
        })
        .catch(error => {
            console.error("Error:", error);
            document.getElementById("result-message").textContent = "An error occurred during prediction.";
        });
});
