// DOM Elements
const storyForm = document.getElementById('story-form');
const generateBtn = document.getElementById('generate-btn');
const logContainer = document.getElementById('log-container');
const currentChapterEl = document.getElementById('current-chapter');
const currentSceneEl = document.getElementById('current-scene');
const chaptersLeftEl = document.getElementById('chapters-left');
const scenesLeftEl = document.getElementById('scenes-left');
const charactersListEl = document.getElementById('characters-list');
const chaptersListEl = document.getElementById('chapters-list');
const scenesListEl = document.getElementById('scenes-list');

// WebSocket connection
let ws;
const connectWebSocket = () => {
    // Connect to the WebSocket server
    ws = new WebSocket('ws://' + window.location.host + '/ws/logs');
    
    ws.onopen = () => {
        console.log('WebSocket connected');
        addLogEntry('Connected to story generation system', 'info');
    };
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        // Handle different types of messages
        if (data.type === 'log') {
            addLogEntry(data.message, data.level || 'info');
        } else if (data.type === 'status') {
            updateStatus(data.status);
        } else if (data.type === 'database') {
            updateDatabaseDisplay(data.data);
        }
    };
    
    ws.onclose = () => {
        console.log('WebSocket disconnected');
        addLogEntry('Connection to story generation system lost. Reconnecting...', 'warning');
        
        // Attempt to reconnect after a delay
        setTimeout(connectWebSocket, 3000);
    };
    
    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        addLogEntry('WebSocket error occurred', 'error');
    };
};

// Add log entry to the log container
const addLogEntry = (message, level = 'info') => {
    const logEntry = document.createElement('div');
    logEntry.className = `log-entry ${level}`;
    logEntry.textContent = message;
    
    // Add timestamp
    const timestamp = new Date().toLocaleTimeString();
    logEntry.innerHTML = `<span class="timestamp">${timestamp}</span> ${message}`;
    
    logContainer.appendChild(logEntry);
    logContainer.scrollTop = logContainer.scrollHeight;
};

// Update status display
const updateStatus = (status) => {
    if (status.current_chapter !== undefined) {
        currentChapterEl.textContent = status.current_chapter;
    }
    
    if (status.current_scene !== undefined) {
        currentSceneEl.textContent = status.current_scene;
    }
    
    if (status.chapters_left !== undefined) {
        chaptersLeftEl.textContent = status.chapters_left;
    }
    
    if (status.scenes_left !== undefined) {
        scenesLeftEl.textContent = status.scenes_left;
    }
};

// Update database display
const updateDatabaseDisplay = (data) => {
    // Update characters list
    if (data.characters) {
        charactersListEl.innerHTML = '';
        if (data.characters.length === 0) {
            const emptyItem = document.createElement('div');
            emptyItem.className = 'list-item empty';
            emptyItem.textContent = 'No characters in database';
            charactersListEl.appendChild(emptyItem);
        } else {
            data.characters.forEach(character => {
                const item = document.createElement('div');
                item.className = 'list-item';
                item.innerHTML = `
                    <strong>${character.name}</strong>
                    <div class="meta">
                        <span>Age: ${character.age || 'N/A'}</span>
                        <span>Role: ${character.role || 'N/A'}</span>
                        <span>Species: ${character.species || 'N/A'}</span>
                    </div>
                `;
                charactersListEl.appendChild(item);
            });
        }
    }
    
    // Update chapters list
    if (data.chapters) {
        chaptersListEl.innerHTML = '';
        if (data.chapters.length === 0) {
            const emptyItem = document.createElement('div');
            emptyItem.className = 'list-item empty';
            emptyItem.textContent = 'No chapters in database';
            chaptersListEl.appendChild(emptyItem);
        } else {
            data.chapters.forEach(chapter => {
                const item = document.createElement('div');
                item.className = 'list-item';
                item.innerHTML = `
                    <strong>${chapter.title}</strong>
                    <div class="meta">
                        <span>Order: ${chapter.order_index || 'N/A'}</span>
                        <span>Scene Count: ${chapter.scene_count || 0}</span>
                        <span>Created: ${chapter.created_at ? new Date(chapter.created_at).toLocaleDateString() : 'N/A'}</span>
                    </div>
                `;
                chaptersListEl.appendChild(item);
            });
        }
    }
    
    // Update scenes list
    if (data.scenes) {
        scenesListEl.innerHTML = '';
        if (data.scenes.length === 0) {
            const emptyItem = document.createElement('div');
            emptyItem.className = 'list-item empty';
            emptyItem.textContent = 'No scenes in database';
            scenesListEl.appendChild(emptyItem);
        } else {
            data.scenes.forEach(scene => {
                const item = document.createElement('div');
                item.className = 'list-item';
                item.innerHTML = `
                    <strong>${scene.title}</strong>
                    <div class="meta">
                        <span>Scene #: ${scene.scene_number || 'N/A'}</span>
                        <span>Characters: ${scene.character_count || 0}</span>
                        <span>Created: ${scene.created_at ? new Date(scene.created_at).toLocaleDateString() : 'N/A'}</span>
                    </div>
                `;
                scenesListEl.appendChild(item);
            });
        }
    }
};

// Fetch database content
const fetchDatabaseContent = async () => {
    try {
        const response = await fetch('/api/database');
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Update the database display
        updateDatabaseDisplay(data);
        
        // Add log entry
        addLogEntry(`Loaded database content: ${data.characters.length} characters, ${data.chapters.length} chapters, ${data.scenes.length} scenes`, 'info');
        
    } catch (error) {
        addLogEntry(`Error fetching database content: ${error.message}`, 'error');
    }
};

// Handle form submission
storyForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const storyIdea = document.getElementById('story-idea').value.trim();
    
    if (!storyIdea) {
        addLogEntry('Please enter a story idea', 'error');
        return;
    }
    
    // Disable the button during generation
    generateBtn.disabled = true;
    generateBtn.textContent = 'Generating...';
    
    try {
        // Send the story idea to the server
        const response = await fetch('/api/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ idea: storyIdea })
        });
        
        const result = await response.json();
        
        if (response.ok) {
            addLogEntry(`Story generation started with ID: ${result.story_id}`, 'success');
            // Reset form
            storyForm.reset();
            
            // Fetch updated database content after generation starts
            fetchDatabaseContent();
        } else {
            addLogEntry(`Error: ${result.message || 'Unknown error'}`, 'error');
        }
    } catch (error) {
        addLogEntry(`Network error: ${error.message}`, 'error');
    } finally {
        // Re-enable the button
        generateBtn.disabled = false;
        generateBtn.textContent = 'Generate';
    }
});

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    // Connect to WebSocket
    connectWebSocket();
    
    // Add initial log entry
    addLogEntry('Web interface loaded. Waiting for connection to story generation system...');
    
    // Fetch initial database data
    fetch('/api/story-ideas')
        .then(response => response.json())
        .then(data => {
            if (data.story_ideas && data.story_ideas.length > 0) {
                addLogEntry(`Loaded ${data.story_ideas.length} story ideas from database`, 'info');
            }
        })
        .catch(error => {
            addLogEntry(`Error loading story ideas: ${error.message}`, 'error');
        });
    
    // Fetch initial database content
    fetchDatabaseContent();
});