document.addEventListener('DOMContentLoaded', () => {
    const video = document.getElementById('webcam-video');
    const captureButton = document.getElementById('capture-button');
    const statusMessage = document.getElementById('status-message');
    const canvas = document.getElementById('canvas');
    const fusedEmotionSpan = document.getElementById('fused-emotion');
    const facialEmotionSpan = document.getElementById('facial-emotion');
    const speechEmotionSpan = document.getElementById('speech-emotion');
    const replyMessageDiv = document.getElementById('reply-message');

    let mediaRecorder;
    const audioChunks = [];

    // --- Phase 2.2: Capture Webcam & Audio ---
    async function setupWebcamAndMic() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
            video.srcObject = stream;
            
            mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.ondataavailable = event => {
                audioChunks.push(event.data);
            };
            
            mediaRecorder.onstop = async () => {
                const audioBlob = new Blob(audioChunks, { 'type' : 'audio/wav' });
                await sendDataToBackend(audioBlob);
            };
            statusMessage.textContent = "Webcam and microphone are active. Click Capture!";

        } catch (err) {
            console.error("Error accessing media devices:", err);
            statusMessage.textContent = "Error: Could not access webcam or microphone.";
        }
    }

    // --- Phase 2.3: Send Data to Backend ---
    async function sendDataToBackend(audioBlob) {
        statusMessage.textContent = "🧠 Analyzing emotion... Please wait.";

        // Capture a still frame from the video feed
        const context = canvas.getContext('2d');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        const imageData = canvas.toDataURL('image/jpeg');

        const reader = new FileReader();
        reader.readAsDataURL(audioBlob);
        reader.onloadend = async () => {
            const base64Audio = reader.result;
            
            try {
                const response = await fetch('http://127.0.0.1:5001/predict_emotion', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        audio_data: base64Audio,
                        image_data: imageData
                    })
                });

                if (!response.ok) throw new Error('Network response was not ok');
                
                const data = await response.json();
                
                // Update the UI with predictions
                fusedEmotionSpan.textContent = data.fused_emotion;
                facialEmotionSpan.textContent = data.facial_emotion;
                speechEmotionSpan.textContent = data.speech_emotion;
                
                // Get chatbot reply
                const replyResponse = await fetch('http://127.0.0.1:5001/chatbot_reply', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ emotion: data.fused_emotion })
                });

                const replyData = await replyResponse.json();
                replyMessageDiv.textContent = replyData.reply;

                statusMessage.textContent = "✅ Analysis complete! Click Capture again!";
                document.getElementById('loading').style.display = 'none';
                captureButton.disabled = false;

            } catch (error) {
                console.error("Failed to fetch from backend:", error);
                statusMessage.textContent = "❌ Error: Failed to connect to backend.";
                document.getElementById('loading').style.display = 'none';
                captureButton.disabled = false;
            }
        };
    }

    // --- Event Listeners ---
    captureButton.addEventListener('click', () => {
        audioChunks.length = 0;
        mediaRecorder.start();
        captureButton.disabled = true;
        statusMessage.textContent = "🎤 Recording for 5 seconds...";
        document.getElementById('loading').style.display = 'block';

        setTimeout(() => {
            if (mediaRecorder.state === 'recording') {
                mediaRecorder.stop();
                statusMessage.textContent = "🔄 Processing your emotions...";
            }
        }, 5000);
    });

    // Start the process
    setupWebcamAndMic();
});