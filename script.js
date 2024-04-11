document.getElementById('submitSymptoms').addEventListener('click', function() {
    const selectedSymptoms = [];
    document.querySelectorAll('.symptom-checkbox:checked').forEach(item => {
        selectedSymptoms.push(item.value);
    });

    fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({symptoms: selectedSymptoms}),
    })
    
    .then(response => response.json())
    .then(data => {
        document.getElementById('output').innerText = `Recommendations: ${data.recommendations}`;
    })
    .catch((error) => {
        console.error('Error:', error);
    });
});
    document.getElementById('clearSymptoms').addEventListener('click', function() {
        document.getElementById('output').innerText = '';

        let selectedSymptoms = document.querySelectorAll('.symptom-checkbox');
        for (let i = 0; i < selectedSymptoms.length; i++)
        {
            selectedSymptoms[i].checked = false;
        }
    });
