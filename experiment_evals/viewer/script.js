let runFolders = [];
let leftData = null;
let rightData = null;
let boardCounter = 0;

// Initialize the application
async function init() {
    try {
        await loadRunFolders();
        setupEventListeners();
    } catch (error) {
        console.error('Initialization error:', error);
        showError('leftContent', 'Failed to initialize application');
        showError('rightContent', 'Failed to initialize application');
    }
}

// Load available run folders
async function loadRunFolders() {
    try {
        const response = await fetch('./src');
        if (!response.ok) throw new Error('Failed to fetch run folders');
        
        const text = await response.text();
        const parser = new DOMParser();
        const doc = parser.parseFromString(text, 'text/html');
        const links = doc.querySelectorAll('a[href]');
        
        runFolders = Array.from(links)
            .map(link => link.getAttribute('href'))
            .filter(href => href && href !== '../' && !href.includes('.'))
            .map(href => href.replace('/', ''));

        populateRunFolderSelects();
    } catch (error) {
        console.error('Error loading run folders:', error);
        showError('leftContent', 'Failed to load run folders. Make sure to serve this page over HTTP.');
        showError('rightContent', 'Failed to load run folders. Make sure to serve this page over HTTP.');
    }
}

// Populate run folder select dropdowns
function populateRunFolderSelects() {
    const selects = [document.getElementById('runFolder1'), document.getElementById('runFolder2')];
    selects.forEach(select => {
        select.innerHTML = '<option value="">Select a run folder...</option>';
        runFolders.forEach(folder => {
            const option = document.createElement('option');
            option.value = folder;
            option.textContent = folder;
            select.appendChild(option);
        });
    });
}

// Load files for a specific run folder
async function loadFiles(runFolder) {
    try {
        const response = await fetch(`./src/${runFolder}`);
        if (!response.ok) throw new Error('Failed to fetch files');
        
        const text = await response.text();
        const parser = new DOMParser();
        const doc = parser.parseFromString(text, 'text/html');
        const links = doc.querySelectorAll('a[href]');
        
        return Array.from(links)
            .map(link => link.getAttribute('href'))
            .filter(href => href && href.endsWith('.json'))
            .map(href => href.replace('./', ''));
    } catch (error) {
        console.error('Error loading files:', error);
        throw error;
    }
}

// Populate file select dropdown
async function populateFileSelect(selectId, runFolder) {
    const select = document.getElementById(selectId);
    if (!runFolder) {
        select.innerHTML = '<option value="">Select a file...</option>';
        select.disabled = true;
        return;
    }

    select.innerHTML = '<option value="">Loading files...</option>';
    select.disabled = true;

    try {
        const files = await loadFiles(runFolder);
        select.innerHTML = '<option value="">Select a file...</option>';
        files.forEach(file => {
            const option = document.createElement('option');
            option.value = file;
            option.textContent = file;
            select.appendChild(option);
        });
        select.disabled = false;
    } catch (error) {
        select.innerHTML = '<option value="">Error loading files</option>';
        console.error('Error populating file select:', error);
    }
}

// Load and parse JSON data
async function loadData(runFolder, fileName) {
    try {
        const response = await fetch(`./src/${runFolder}/${fileName}`);
        if (!response.ok) throw new Error('Failed to fetch data');
        return await response.json();
    } catch (error) {
        console.error('Error loading data:', error);
        throw error;
    }
}

// Parse prompt to extract system and user parts
function parsePrompt(prompt) {
    if (!prompt) return { system: '', user: '' };
    
    // Check for new format with <|start_header_id|> and <|end_header_id|>
    const headerIdSystemMatch = prompt.match(/<\|start_header_id\|>system<\|end_header_id\|>(.*?)<\|eot_id\|><\|start_header_id\|>user<\|end_header_id\|>/s);
    const headerIdUserMatch = prompt.match(/<\|start_header_id\|>user<\|end_header_id\|>(.*?)(<\|eot_id\|>|$)/s);
    
    if (headerIdSystemMatch || headerIdUserMatch) {
        return {
            system: headerIdSystemMatch ? headerIdSystemMatch[1].trim() : '',
            user: headerIdUserMatch ? headerIdUserMatch[1].trim() : ''
        };
    }
    
    // Check for original format with <|im_start|> and <|im_end|>
    const systemMatch = prompt.match(/<\|im_start\|>system<\|im_end\|>(.*?)<\|im_start\|>user<\|im_end\|>/s);
    const userMatch = prompt.match(/<\|im_start\|>user<\|im_end\|>(.*?)(<\|im_start\|>assistant<\|im_end\|>|$)/s);
    
    if (systemMatch || userMatch) {
        return {
            system: systemMatch ? systemMatch[1].trim() : '',
            user: userMatch ? userMatch[1].trim() : ''
        };
    }
    
    // Fallback: treat entire prompt as user prompt
    return {
        system: '',
        user: prompt
    };
}

// Display a random sample from the data
function displayRandomSample(data, contentId, runFolder, fileName) {
    const contentEl = document.getElementById(contentId);
    
    if (!data || !Array.isArray(data) || data.length === 0) {
        showError(contentId, 'No data available');
        return;
    }

    const randomIndex = Math.floor(Math.random() * data.length);
    const sample = data[randomIndex];
    const { system, user } = parsePrompt(sample.prompt);
    
    // Parse info data
    let boardFen = '';
    let infoData = null;
    
    if (sample.info) {
        // Direct info structure
        boardFen = sample.info.board || '';
        infoData = sample.info;
    } else if (sample.ground_truth && sample.ground_truth.info) {
        // Info nested under ground_truth
        boardFen = sample.ground_truth.info.board || '';
        infoData = sample.ground_truth.info;
    } else if (sample.ground_truth) {
        // Old format - use existing structure as fallback
        infoData = sample.ground_truth;
        boardFen = '';
    }
    
    // Generate chess board HTML if we have a FEN
    let boardImageHtml = '';
    let boardId = '';
    if (boardFen) {
        boardId = `board-${contentId}-${boardCounter++}`;
        boardImageHtml = `
            <div class="section">
                <h4>♟️ Chess Board</h4>
                <div class="chess-board">
                    <div id="${boardId}" style="width: 200px; margin: 0 auto;"></div>
                </div>
            </div>
        `;
    }
    
    contentEl.innerHTML = `
        <div class="sample-info">
            <h3>Sample ${randomIndex + 1} of ${data.length}</h3>
        </div>

        ${boardImageHtml}

        ${system ? `
        <div class="section">
            <details>
                <summary style="cursor: pointer; font-weight: bold; padding: 5px 0;">🎯 System Prompt</summary>
                <div class="prompt-text" style="margin-top: 10px;">${escapeHtml(system)}</div>
            </details>
        </div>
        ` : ''}

        <div class="section">
            <details>
                <summary style="cursor: pointer; font-weight: bold; padding: 5px 0;">👤 User Prompt</summary>
                <div class="prompt-text" style="margin-top: 10px;">${escapeHtml(user)}</div>
            </details>
        </div>

        <div class="section">
            <h4>🤖 Model Response</h4>
            <div class="response-text">${escapeHtml(sample.model_response || 'No response available')}</div>
        </div>

        ${sample.parsed_response ? `
        <div class="section">
            <details open>
                <summary style="cursor: pointer; font-weight: bold; padding: 5px 0;">🔍 Parsed Response</summary>
                <div class="parsed-response-text" style="margin-top: 10px;">
                    <pre>${JSON.stringify(sample.parsed_response, null, 2)}</pre>
                </div>
            </details>
        </div>
        ` : ''}

        <div class="section">
            <details${sample.parsed_response ? '' : ' open'}>
                <summary style="cursor: pointer; font-weight: bold; padding: 5px 0;">ℹ️ Info</summary>
                <div class="info-text" style="margin-top: 10px;">
                    ${infoData ? `<pre>${JSON.stringify(infoData, null, 2)}</pre>` : 'No info data available'}
                </div>
            </details>
        </div>
    `;
    
    // Render chess board if we have a FEN and board ID
    if (boardFen && boardId) {
        setTimeout(() => {
            try {
                Chessboard(boardId, {
                    position: boardFen,
                    showNotation: false,
                    pieceTheme: 'https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png'
                });
            } catch (error) {
                console.error('Error rendering chess board:', error);
                // Fallback to text display
                document.getElementById(boardId).innerHTML = `<div style="font-size: 10px; text-align: center; padding: 10px; background: #f0f0f0; border-radius: 4px;">FEN: ${boardFen}</div>`;
            }
        }, 100);
    }
}

// Show error message
function showError(contentId, message) {
    document.getElementById(contentId).innerHTML = `<div class="error">${escapeHtml(message)}</div>`;
}

// Show loading message
function showLoading(contentId, message = 'Loading...') {
    document.getElementById(contentId).innerHTML = `<div class="loading">${escapeHtml(message)}</div>`;
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Update button states
function updateButtonStates() {
    const randomizeBtn = document.getElementById('randomizeBtn');
    const randomizeLeftBtn = document.getElementById('randomizeLeftBtn');
    const randomizeRightBtn = document.getElementById('randomizeRightBtn');

    randomizeBtn.disabled = !leftData || !rightData;
    randomizeLeftBtn.disabled = !leftData;
    randomizeRightBtn.disabled = !rightData;
}

// Setup event listeners
function setupEventListeners() {
    // Run folder selection
    document.getElementById('runFolder1').addEventListener('change', (e) => {
        populateFileSelect('fileSelect1', e.target.value);
        leftData = null;
        document.getElementById('leftContent').innerHTML = '<div class="loading">Select a file to view data</div>';
        updateButtonStates();
    });

    document.getElementById('runFolder2').addEventListener('change', (e) => {
        populateFileSelect('fileSelect2', e.target.value);
        rightData = null;
        document.getElementById('rightContent').innerHTML = '<div class="loading">Select a file to view data</div>';
        updateButtonStates();
    });

    // File selection
    document.getElementById('fileSelect1').addEventListener('change', async (e) => {
        const runFolder = document.getElementById('runFolder1').value;
        const fileName = e.target.value;
        
        if (!runFolder || !fileName) {
            leftData = null;
            updateButtonStates();
            return;
        }

        showLoading('leftContent', 'Loading data...');
        try {
            leftData = await loadData(runFolder, fileName);
            displayRandomSample(leftData, 'leftContent', runFolder, fileName);
        } catch (error) {
            showError('leftContent', 'Failed to load data');
            leftData = null;
        }
        updateButtonStates();
    });

    document.getElementById('fileSelect2').addEventListener('change', async (e) => {
        const runFolder = document.getElementById('runFolder2').value;
        const fileName = e.target.value;
        
        if (!runFolder || !fileName) {
            rightData = null;
            updateButtonStates();
            return;
        }

        showLoading('rightContent', 'Loading data...');
        try {
            rightData = await loadData(runFolder, fileName);
            displayRandomSample(rightData, 'rightContent', runFolder, fileName);
        } catch (error) {
            showError('rightContent', 'Failed to load data');
            rightData = null;
        }
        updateButtonStates();
    });

    // Randomize buttons
    document.getElementById('randomizeBtn').addEventListener('click', () => {
        if (leftData) {
            const runFolder1 = document.getElementById('runFolder1').value;
            const fileName1 = document.getElementById('fileSelect1').value;
            displayRandomSample(leftData, 'leftContent', runFolder1, fileName1);
        }
        if (rightData) {
            const runFolder2 = document.getElementById('runFolder2').value;
            const fileName2 = document.getElementById('fileSelect2').value;
            displayRandomSample(rightData, 'rightContent', runFolder2, fileName2);
        }
    });

    document.getElementById('randomizeLeftBtn').addEventListener('click', () => {
        if (leftData) {
            const runFolder1 = document.getElementById('runFolder1').value;
            const fileName1 = document.getElementById('fileSelect1').value;
            displayRandomSample(leftData, 'leftContent', runFolder1, fileName1);
        }
    });

    document.getElementById('randomizeRightBtn').addEventListener('click', () => {
        if (rightData) {
            const runFolder2 = document.getElementById('runFolder2').value;
            const fileName2 = document.getElementById('fileSelect2').value;
            displayRandomSample(rightData, 'rightContent', runFolder2, fileName2);
        }
    });
}

// Start the application when DOM is loaded
document.addEventListener('DOMContentLoaded', init); 