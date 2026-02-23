// Main JavaScript file for Cloud UBA Dashboard

// document.addEventListener('DOMContentLoaded', function() {
//     // Initialize tooltips
//     var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
//     var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
//         return new bootstrap.Tooltip(tooltipTriggerEl)
//     });

//     // Form submission handling
//     const predictionForm = document.getElementById('prediction-form');
//     if (predictionForm) {
//         predictionForm.addEventListener('submit', function(e) {
//             e.preventDefault();
            
//             const formData = new FormData(predictionForm);
//             const jsonData = {};
            
//             formData.forEach((value, key) => {
//                 jsonData[key] = value;
//             });
            
//             fetch('/api/predict', {
//                 method: 'POST',
//                 headers: {
//                     'Content-Type': 'application/json',
//                     'X-API-Key': document.getElementById('api-key').value
//                 },
//                 body: JSON.stringify(jsonData)
//             })
//             .then(response => response.json())
//             .then(data => {
//                 document.getElementById('prediction-result').innerHTML = 
//                     `<div class="alert ${data.score > 0.8 ? 'alert-high' : data.score > 0.5 ? 'alert-medium' : 'alert-low'}">
//                         <h4>Prediction Result</h4>
//                         <p>Anomaly Score: ${data.score.toFixed(4)}</p>
//                         <p>Explanation: ${data.explanation}</p>
//                     </div>`;
//             })
//             .catch(error => {
//                 document.getElementById('prediction-result').innerHTML = 
//                     `<div class="alert alert-danger">
//                         <h4>Error</h4>
//                         <p>${error.message}</p>
//                     </div>`;
//             });
//         });
//     }
// });

// Find the form submission event handler in your JavaScript
document.addEventListener('DOMContentLoaded', function() {
    const predictionForm = document.getElementById('prediction-form');
    if (predictionForm) {
        predictionForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const formData = new FormData(predictionForm);
            const jsonData = {};
            
            formData.forEach((value, key) => {
                jsonData[key] = value;
            });
            
            fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': document.getElementById('api-key').value
                },
                body: JSON.stringify(jsonData)
            })
            .then(response => response.json())
            .then(data => {
                console.log('API Response:', data);
                // Check if score exists before using toFixed
                const score = data.score !== undefined ? data.score.toFixed(4) : 'N/A';
                const scoreClass = data.score > 0.8 ? 'alert-high' : 
                                  data.score > 0.5 ? 'alert-medium' : 'alert-low';
                
                document.getElementById('prediction-result').innerHTML = 
                    `<div class="alert ${scoreClass}">
                        <h4>Prediction Result</h4>
                        <p>Anomaly Score: ${score}</p>
                        <p>Explanation: ${data.explanation || 'No explanation available'}</p>
                    </div>`;
            })
            .catch(error => {
                document.getElementById('prediction-result').innerHTML = 
                    `<div class="alert alert-danger">
                        <h4>Error</h4>
                        <p>${error.message}</p>
                    </div>`;
            });
        });
    }
});