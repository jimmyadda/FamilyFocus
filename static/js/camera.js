const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const statusText = document.getElementById("status");
const startButton = document.getElementById("startCamera");
const recognizeButton = document.getElementById("startRecognition");
const stopButton = document.getElementById("stopCamera");

let stream = null;
let recognitionTimer = null;
let isRecognizing = false;
let isRequestInFlight = false;

async function startCamera() {
    try {
        statusText.textContent = "Requesting camera permission...";

        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: "environment",
                width: { ideal: 640 },
                height: { ideal: 480 }
            },
            audio: false
        });

        video.srcObject = stream;
        await video.play();

        resizeOverlay();
        window.addEventListener("resize", resizeOverlay);

        statusText.textContent = "Camera is on. Press Start Recognition.";
    } catch (error) {
        console.error(error);
        statusText.textContent = "Could not start camera. Allow camera access.";
    }
}

function resizeOverlay() {
    overlay.width = video.videoWidth || 1280;
    overlay.height = video.videoHeight || 720;
}

function stopCamera() {
    stopRecognition();

    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }

    clearOverlay();
    statusText.textContent = "Camera is off.";
}

function clearOverlay() {
    const ctx = overlay.getContext("2d");
    ctx.clearRect(0, 0, overlay.width, overlay.height);
}

function startRecognition() {
    if (!stream) {
        statusText.textContent = "Start camera first.";
        return;
    }


    isRecognizing = true;
    statusText.textContent = "Recognition is running...";

    if (recognitionTimer) {
        clearInterval(recognitionTimer);
    }

    recognitionTimer = setInterval(sendFrameToServer, 1500);
}

function stopRecognition() {
    isRecognizing = false;

    if (recognitionTimer) {
        clearInterval(recognitionTimer);
        recognitionTimer = null;
    }

    isRequestInFlight = false;
}

function captureFrameAsJpeg() {
    const tempCanvas = document.createElement("canvas");

    const videoWidth = video.videoWidth;
    const videoHeight = video.videoHeight;

    if (!videoWidth || !videoHeight) {
        return null;
    }

    const maxWidth = 640;
    const scale = Math.min(1, maxWidth / videoWidth);

    tempCanvas.width = Math.round(videoWidth * scale);
    tempCanvas.height = Math.round(videoHeight * scale);

    const ctx = tempCanvas.getContext("2d");
    ctx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);

    return {
        image: tempCanvas.toDataURL("image/jpeg", 0.75),
        sentWidth: tempCanvas.width,
        sentHeight: tempCanvas.height
    };
}

async function sendFrameToServer() {
    if (!isRecognizing || isRequestInFlight) {
        return;
    }

    const frame = captureFrameAsJpeg();

    if (!frame) {
        return;
    }

    isRequestInFlight = true;

    try {
        const response = await fetch("/api/recognize-frame", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({                
                image: frame.image
            })
        });

        const result = await response.json();

        if (!response.ok || !result.ok) {
            clearOverlay();
            console.error(result);
            statusText.textContent = result.error || "Recognition failed.";
            return;
        }

        drawRecognitionResults(
            result.matches || [],
            frame.sentWidth,
            frame.sentHeight
        );

        const found = (result.matches || []).some(face => face.match);

        if (found) {
            statusText.textContent = `Family member found. Faces detected: ${result.faces_found}`;
        } else {
            statusText.textContent = `Searching... Faces detected: ${result.faces_found}`;
        }

    } catch (error) {
        console.error(error);
        statusText.textContent = "Server recognition error. Check Flask terminal.";
    } finally {
        isRequestInFlight = false;
    }
}

function drawRecognitionResults(matches, sentWidth, sentHeight) {
    clearOverlay();

    const ctx = overlay.getContext("2d");

    const scaleX = overlay.width / sentWidth;
    const scaleY = overlay.height / sentHeight;

    for (const face of matches) {
        const box = face.box;

        const x = box.x * scaleX;
        const y = box.y * scaleY;
        const width = box.width * scaleX;
        const height = box.height * scaleY;

        const isMatch = face.match === true;

        // BLUE = selected family member
        // GREEN = face detected, but not selected member
        ctx.strokeStyle = isMatch ? "#0066ff" : "#00cc44";
        ctx.fillStyle = isMatch ? "#0066ff" : "#00cc44";
        ctx.lineWidth = isMatch ? 6 : 4;

        ctx.strokeRect(x, y, width, height);

        const label = isMatch
            ? `${face.name} ${face.distance}`
            : "Unknown";

        ctx.font = "24px Arial";
        ctx.fillText(label, x, Math.max(25, y - 10));
    }
}

startButton.addEventListener("click", startCamera);
recognizeButton.addEventListener("click", startRecognition);
stopButton.addEventListener("click", stopCamera);