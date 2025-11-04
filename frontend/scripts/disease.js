
document.addEventListener('DOMContentLoaded', () => {
    loadDiseases();
});

async function loadDiseases() {
    
    const container = document.getElementById('disease-container');
    
    if (!container) {
        console.error("Error: Element with ID 'disease-container' not found in HTML.");
        return; 
    }

    try {
    
        const response = await fetch('data/diseases.json');
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const diseases = await response.json();

        diseases.forEach(disease => {
        
            const diseaseElement = document.createElement('div');
            diseaseElement.classList.add('disease-item'); 

            diseaseElement.innerHTML = `
                <div class="disease-text">
                    <h2>${disease.id}. ${disease.title}</h2>
                    <p>${disease.description}</p>
                </div>
                <div class="disease-image">
                    <img src="${disease.image_url}" alt="${disease.title}">
                </div>
            `;

            container.appendChild(diseaseElement);
        });

    } catch (error) {
        
        console.error("An error occurred while loading diseases:", error);
        container.innerHTML = "<p>Data could not be loaded. Please try again later.</p>";
    }
}