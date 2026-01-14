let currentModelType = 'mnist'; // 'mnist' or 'cifar'
const CLASSES = {
    'mnist': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
    'cifar': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
};

document.addEventListener('DOMContentLoaded', () => {
    fetchModels();
    setupCanvas();
});

// --- Model Management ---
async function fetchModels() {
    try {
        const res = await fetch('/models');
        const models = await res.json();
        
        const selector = document.getElementById('model-selector');
        selector.innerHTML = '';
        
        if (models.length === 0) {
            const opt = document.createElement('option');
            opt.text = "No models found";
            selector.add(opt);
            return;
        }

        models.forEach(model => {
            const opt = document.createElement('option');
            opt.value = model.filename;
            opt.text = model.name; // e.g., "MNIST (123456)"
            opt.dataset.type = model.type;
            selector.add(opt);
        });

        loadModelConfig();
    } catch (err) {
        console.error("Failed to load models:", err);
    }
}

function loadModelConfig() {
    const selector = document.getElementById('model-selector');
    const selectedOpt = selector.options[selector.selectedIndex];
    
    if (!selectedOpt || !selectedOpt.value) return;

    currentModelType = selectedOpt.dataset.type;
    document.getElementById('model-type-title').innerText = currentModelType.toUpperCase();

    // Update UI based on model type
    const drawTab = document.getElementById('draw-tab');
    
    if (currentModelType === 'cifar') {
        // CIFAR only supports upload
        switchTab('upload');
        drawTab.style.opacity = '0.5';
        drawTab.style.pointerEvents = 'none';
        document.getElementById('confidence').innerText = "Upload an image (32x32 preferred)";
    } else {
        // MNIST supports both
        drawTab.style.opacity = '1';
        drawTab.style.pointerEvents = 'all';
        switchTab('draw'); // Default back to draw for MNIST
        document.getElementById('confidence').innerText = "Draw a digit";
    }

    // Tell backend to load this model
    fetch(`/load_model?model=${selectedOpt.value}`);
}

// --- UI Tabs ---
function switchTab(mode) {
    const drawWrapper = document.getElementById('canvas-wrapper');
    const uploadWrapper = document.getElementById('upload-wrapper');
    const drawTab = document.getElementById('draw-tab');
    const uploadTab = document.getElementById('upload-tab');

    if (mode === 'draw') {
        drawWrapper.style.display = 'flex';
        uploadWrapper.style.display = 'none';
        drawTab.classList.add('active');
        uploadTab.classList.remove('active');
    } else {
        drawWrapper.style.display = 'none';
        uploadWrapper.style.display = 'flex';
        drawTab.classList.remove('active');
        uploadTab.classList.add('active');
    }
}

// --- Canvas Logic ---
let canvas, ctx, isDrawing = false;

function setupCanvas() {
    canvas = document.getElementById('canvas');
    ctx = canvas.getContext('2d');
    
    // Black ink on White paper
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 15;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);
}

function startDrawing(e) { isDrawing = true; draw(e); }
function stopDrawing() { isDrawing = false; ctx.beginPath(); }
function draw(e) {
    if (!isDrawing) return;
    const rect = canvas.getBoundingClientRect();
    ctx.lineTo(e.clientX - rect.left, e.clientY - rect.top);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(e.clientX - rect.left, e.clientY - rect.top);
}

function clearCanvas() {
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    resetResults();
}

// --- Upload Logic ---
function previewImage(input) {
    if (input.files && input.files[0]) {
        const reader = new FileReader();
        reader.onload = function(e) {
            const img = document.getElementById('image-preview');
            img.src = e.target.result;
            img.style.display = 'block';
            document.querySelector('.drop-zone p').style.display = 'none';
        }
        reader.readAsDataURL(input.files[0]);
    }
}

// --- Prediction ---
async function predictCanvas() {
    canvas.toBlob(blob => sendPrediction(blob));
}

async function predictUpload() {
    const input = document.getElementById('file-input');
    if (input.files[0]) {
        sendPrediction(input.files[0]);
    } else {
        alert("Please select an image first.");
    }
}

async function sendPrediction(blob) {
    const loader = document.getElementById('loader');
    const predDiv = document.getElementById('prediction');
    const confDiv = document.getElementById('confidence');
    const classesList = document.getElementById('classes-list');

    // Reset UI
    predDiv.style.display = 'none';
    loader.style.display = 'block';
    classesList.innerHTML = '';

    const formData = new FormData();
    formData.append('file', blob, 'image.png');

    try {
        const res = await fetch('/predict', { method: 'POST', body: formData });
        const data = await res.json();

        loader.style.display = 'none';
        predDiv.style.display = 'block';

        if (data.error) {
            alert(data.error);
            return;
        }

        // Get class name
        const classNames = CLASSES[currentModelType] || [];
        const label = classNames[data.prediction] || data.prediction;

        predDiv.innerText = label;
        confDiv.innerText = `Confidence: ${(data.confidence * 100).toFixed(1)}%`;
        
        // Show all probabilities
        if (data.probabilities) {
            data.probabilities.forEach((prob, idx) => {
                const className = classNames[idx] || idx;
                const div = document.createElement('div');
                div.className = `class-item ${idx === data.prediction ? 'top' : ''}`;
                div.innerHTML = `<span>${className}</span><span>${(prob * 100).toFixed(1)}%</span>`;
                classesList.appendChild(div);
            });
        }

    } catch (err) {
        console.error(err);
        loader.style.display = 'none';
        confDiv.innerText = "Error occurred";
    }
}

function resetResults() {
    document.getElementById('prediction').innerText = "-";
    document.getElementById('confidence').innerText = "Ready";
    document.getElementById('classes-list').innerHTML = "";
}
