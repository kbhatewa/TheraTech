// script.js 

document.addEventListener("DOMContentLoaded", function () {
    const analyzeButton = document.getElementById("analyze-emotion");
    const emotionResponse = document.getElementById("emotion-response");

    analyzeButton.addEventListener("click", function () {
        fetch("/analyze_emotion")
            .then(response => response.json())
            .then(data => {
                emotionResponse.textContent = `${data.emotion}`;
            })
            .catch(error => {
                emotionResponse.textContent = "Error analyzing emotion.";
                console.error("Error:", error);
            });
    });
});

