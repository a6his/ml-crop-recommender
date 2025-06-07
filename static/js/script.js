// JavaScript for enhanced user experience

// Example data for quick fill
const examples = {
    rice: {
        nitrogen: 90,
        phosphorus: 45,
        potassium: 40,
        temperature: 25,
        humidity: 80,
        ph: 6.5,
        rainfall: 250
    },
    coffee: {
        nitrogen: 100,
        phosphorus: 30,
        potassium: 30,
        temperature: 20,
        humidity: 65,
        ph: 7.0,
        rainfall: 150
    },
    banana: {
        nitrogen: 120,
        phosphorus: 60,
        potassium: 60,
        temperature: 30,
        humidity: 70,
        ph: 6.8,
        rainfall: 80
    }
};

// Fill form with example data
function fillExample(cropType) {
    const data = examples[cropType];
    if (data) {
        Object.keys(data).forEach(key => {
            const input = document.getElementById(key);
            if (input) {
                input.value = data[key];
                // Add visual feedback
                input.style.backgroundColor = '#e8f5e8';
                setTimeout(() => {
                    input.style.backgroundColor = '';
                }, 1000);
            }
        });
    }
}

// Form validation and submission
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('cropForm');
    
    if (form) {
        form.addEventListener('submit', function(e) {
            // Add loading state
            const submitBtn = form.querySelector('button[type="submit"]');
            const originalText = submitBtn.innerHTML;
            
            submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Processing...';
            submitBtn.disabled = true;
            
            // Add loading class to form
            form.classList.add('loading');
            
            // Form will submit normally, but with visual feedback
        });
        
        // Real-time validation
        const inputs = form.querySelectorAll('input[type="number"]');
        inputs.forEach(input => {
            input.addEventListener('input', function() {
                validateInput(this);
            });
            
            input.addEventListener('blur', function() {
                validateInput(this);
            });
        });
    }
});

// Input validation function
function validateInput(input) {
    const value = parseFloat(input.value);
    const min = parseFloat(input.min);
    const max = parseFloat(input.max);
    
    // Remove existing validation classes
    input.classList.remove('is-valid', 'is-invalid');
    
    if (input.value === '') {
        return; // Don't validate empty inputs
    }
    
    if (isNaN(value) || value < min || value > max) {
        input.classList.add('is-invalid');
        showTooltip(input, `Value must be between ${min} and ${max}`);
    } else {
        input.classList.add('is-valid');
        hideTooltip(input);
    }
}

// Tooltip functions
function showTooltip(element, message) {
    hideTooltip(element); // Remove existing tooltip
    
    const tooltip = document.createElement('div');
    tooltip.className = 'tooltip-validation';
    tooltip.innerHTML = message;
    tooltip.style.cssText = `
        position: absolute;
        background: #dc3545;
        color: white;
        padding: 5px 10px;
        border-radius: 4px;
        font-size: 12px;
        z-index: 1000;
        white-space: nowrap;
        margin-top: 5px;
    `;
    
    element.parentNode.style.position = 'relative';
    element.parentNode.appendChild(tooltip);
}

function hideTooltip(element) {
    const tooltip = element.parentNode.querySelector('.tooltip-validation');
    if (tooltip) {
        tooltip.remove();
    }
}

// API call function for programmatic access
async function predictCrop(data) {
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('Error making prediction:', error);
        throw error;
    }
}

// Copy result to clipboard
function copyResult() {
    const result = document.querySelector('.alert-success h3').textContent;
    navigator.clipboard.writeText(result).then(() => {
        alert('Result copied to clipboard!');
    });
}

// Auto-resize textareas
function autoResize(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = textarea.scrollHeight + 'px';
}