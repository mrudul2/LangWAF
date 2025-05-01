/**
 * LangWAF Client-side Functionality
 */

// Global variable to track polling state
let pollingActive = false;
let pollingInterval = null;

// Add fade-in animation to cards on page load
document.addEventListener('DOMContentLoaded', function() {
    // Apply fade-in class to cards with a slight delay for each
    const cards = document.querySelectorAll('.card');
    cards.forEach((card, index) => {
        setTimeout(() => {
            card.classList.add('fade-in');
        }, 100 * index);
    });

    // Initialize tooltips if Bootstrap is loaded
    if (typeof bootstrap !== 'undefined' && bootstrap.Tooltip) {
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }

    // Start the status polling
    startPolling();
});

// Function to manage the polling for status updates
function startPolling() {
    if (pollingActive) return; // Don't start multiple polling instances

    pollingActive = true;

    // Initial poll immediately
    pollAllStatus();

    // Set up regular polling interval (every few seconds)
    pollingInterval = setInterval(pollAllStatus, 3000);

    console.log("Status polling started");
}

// Function to poll all model statuses and update UI
function pollAllStatus() {
    const models = ['model1', 'model2', 'model3', 'pipeline'];

    // First fetch all status data
    Promise.all(models.map(model => 
        fetch(`/status/${model}`)
            .then(response => {
                if (!response.ok) throw new Error(`Status fetch failed for ${model}`);
                return response.json();
            })
    ))
    .then(statusResults => {
        // Update UI with status data
        models.forEach((model, index) => {
            updateStatusUI(model, statusResults[index]);
        });

        // Check if any model is running
        const anyRunning = statusResults.some(status => status.status === 'running');

        // Then fetch results data - we do this regardless if models are running
        // to ensure we catch completed executions
        return fetch('/results')
            .then(response => {
                if (!response.ok) throw new Error('Results fetch failed');
                return response.json();
            })
            .then(resultsData => {
                // Update both our UI components
                updateResultsUI(resultsData);

                // Also update the chart and history in index.html
                if (typeof updatePerformanceChart === 'function') {
                    updatePerformanceChart(resultsData);
                }
                if (typeof updateExecutionHistory === 'function') {
                    updateExecutionHistory(resultsData);
                }
            });
    })
    .catch(error => {
        console.error('Error in status polling:', error);
    });
}

// Function to update status UI elements
function updateStatusUI(model, statusData) {
    if (!statusData) return;

    const statusElem = document.getElementById(`${model}-status`);
    const progressElem = document.getElementById(`${model}-progress`);
    const messageElem = document.getElementById(`${model}-message`);

    if (!statusElem || !progressElem || !messageElem) {
        console.warn(`UI elements not found for ${model}`);
        return;
    }

    // Update status text with appropriate styling
    statusElem.textContent = statusData.status.charAt(0).toUpperCase() + statusData.status.slice(1);
    statusElem.className = 'fw-bold';

    // Add style based on status
    if (statusData.status === 'running') {
        statusElem.classList.add('text-primary');
        progressElem.classList.add('progress-bar-animated', 'progress-bar-striped');
    } else if (statusData.status === 'completed') {
        statusElem.classList.add('text-success');
        progressElem.classList.remove('progress-bar-animated', 'progress-bar-striped');
    } else if (statusData.status === 'error') {
        statusElem.classList.add('text-danger');
        progressElem.classList.remove('progress-bar-animated', 'progress-bar-striped');
    } else {
        statusElem.classList.add('text-secondary');
        progressElem.classList.remove('progress-bar-animated', 'progress-bar-striped');
    }

    // Update progress bar
    progressElem.style.width = `${statusData.progress}%`;
    progressElem.setAttribute('aria-valuenow', statusData.progress);

    // Update message
    if (statusData.message) {
        messageElem.textContent = statusData.message;
    }
}

// Function to update the results UI elements
function updateResultsUI(results) {
    // Make this function globally available for the index.html script
    window.updateResultsUI = function(data) {
        updateResultsUI(data);
    };

    // Update performance chart if it exists
    if (window.performanceChart) {
        // Ensure each model has valid data
        const chartData = [
            results.model1 && results.model1.execution_time ? results.model1.execution_time : 0,
            results.model2 && results.model2.execution_time ? results.model2.execution_time : 0,
            results.model3 && results.model3.execution_time ? results.model3.execution_time : 0,
            results.pipeline && results.pipeline.execution_time ? results.pipeline.execution_time : 0
        ];

        window.performanceChart.data.datasets[0].data = chartData;
        window.performanceChart.update();
    }

    // Update execution times in model cards
    ['model1', 'model2', 'model3', 'pipeline'].forEach(model => {
        const timeElem = document.getElementById(`${model}-time`);
        if (timeElem && results[model] && results[model].execution_time) {
            timeElem.textContent = `Last run: ${formatTime(results[model].execution_time)}`;
        }
    });

    // Update execution history
    updateExecutionHistory(results);
}

// Function to update execution history section
function updateExecutionHistory(results) {
    const historyElem = document.getElementById('execution-history');
    if (!historyElem) return;

    // Filter models that have been executed
    const executedModels = ['model1', 'model2', 'model3', 'pipeline'].filter(
        model => results[model] && results[model].completed_at && results[model].completed_at !== ''
    );

    if (executedModels.length === 0) {
        historyElem.innerHTML = `
            <div class="text-center py-5">
                <i class="fas fa-hourglass text-muted mb-3" style="font-size: 2rem;"></i>
                <p class="text-muted">No executions yet</p>
            </div>
        `;
        return;
    }

    // Sort models by most recent execution
    executedModels.sort((a, b) => {
        const dateA = new Date(results[a].completed_at);
        const dateB = new Date(results[b].completed_at);
        return dateB - dateA;
    });

    // Generate HTML for execution history
    let html = '<ul class="list-group list-group-flush">';

    executedModels.forEach(model => {
        const data = results[model];
        if (!data.execution_time) return;

        const modelName = 
            model === 'model1' ? 'Base Classification' :
            model === 'model2' ? 'Language Detection' :
            model === 'model3' ? 'SVM Classification' : 'Full Pipeline';

        const icon = 
            model === 'model1' ? 'robot' :
            model === 'model2' ? 'language' :
            model === 'model3' ? 'brain' : 'project-diagram';

        const bgClass = 
            model === 'model1' ? 'bg-primary' :
            model === 'model2' ? 'bg-success' :
            model === 'model3' ? 'bg-info' : 'bg-dark';

        const statusClass = data.status === 'completed' ? 'text-success' : 
                          data.status === 'error' ? 'text-danger' : 'text-muted';

        html += `
            <li class="list-group-item d-flex justify-content-between align-items-center py-3">
                <div>
                    <span class="badge ${bgClass} me-2">
                        <i class="fas fa-${icon}"></i>
                    </span>
                    <strong>${modelName}</strong>
                    <div class="text-muted small">${data.completed_at}</div>
                </div>
                <div class="text-end">
                    <span class="badge bg-light text-dark execution-badge">
                        ${formatTime(data.execution_time)}
                    </span>
                    <div class="small ${statusClass}">
                        ${data.status.charAt(0).toUpperCase() + data.status.slice(1)}
                    </div>
                </div>
            </li>
        `;
    });

    html += '</ul>';
    historyElem.innerHTML = html;
}

// Function to run a model via API request
function runModel(model) {
    console.log(`Running model: ${model}`);

    // Update UI immediately to show processing
    const statusElem = document.getElementById(`${model}-status`);
    const messageElem = document.getElementById(`${model}-message`);
    const progressElem = document.getElementById(`${model}-progress`);

    if (statusElem && messageElem && progressElem) {
        statusElem.textContent = "Starting...";
        statusElem.className = 'fw-bold text-primary';
        messageElem.textContent = `Starting ${model} execution...`;

        // Add animation to progress bar
        progressElem.style.width = '10%';
        progressElem.setAttribute('aria-valuenow', 10);
        progressElem.classList.add('progress-bar-animated', 'progress-bar-striped');
    }

    // Make API request to run the model
    fetch(`/run/${model}`)
        .then(response => {
            if (!response.ok) {
                throw new Error('Failed to start model');
            }
            return response.json();
        })
        .then(data => {
            console.log(`Server response for ${model}:`, data);

            if (data.status === 'started') {
                // Ensure polling is active to update status
                startPolling();

                // Show success notification
                if (window.Swal) {
                    Swal.fire({
                        title: 'Success!',
                        text: `${model} execution started successfully`,
                        icon: 'success',
                        timer: 2000,
                        showConfirmButton: false
                    });
                }
            } else if (data.status === 'already_running') {
                // Show info notification
                if (window.Swal) {
                    Swal.fire({
                        title: 'Already Running',
                        text: data.message,
                        icon: 'info'
                    });
                }
            } else {
                // Show error notification
                if (window.Swal) {
                    Swal.fire({
                        title: 'Error',
                        text: data.message,
                        icon: 'error'
                    });
                }

                // Reset status UI on error
                if (statusElem && messageElem && progressElem) {
                    statusElem.textContent = "Idle";
                    statusElem.className = 'fw-bold text-secondary';
                    messageElem.textContent = "Ready to execute";
                    progressElem.style.width = '0%';
                    progressElem.setAttribute('aria-valuenow', 0);
                    progressElem.classList.remove('progress-bar-animated', 'progress-bar-striped');
                }
            }
        })
        .catch(error => {
            console.error('Error starting model:', error);

            // Show error notification
            if (window.Swal) {
                Swal.fire({
                    title: 'Error',
                    text: 'Failed to start model execution',
                    icon: 'error'
                });
            }

            // Reset status UI on error
            if (statusElem && messageElem && progressElem) {
                statusElem.textContent = "Idle";
                statusElem.className = 'fw-bold text-secondary';
                messageElem.textContent = "Ready to execute";
                progressElem.style.width = '0%';
                progressElem.setAttribute('aria-valuenow', 0);
                progressElem.classList.remove('progress-bar-animated', 'progress-bar-striped');
            }
        });
}

// Function to format time in a human-readable format
function formatTime(seconds) {
    if (!seconds) return '0 seconds';

    seconds = parseFloat(seconds);
    if (isNaN(seconds)) return '0 seconds';

    if (seconds < 60) {
        return `${seconds.toFixed(2)} seconds`;
    } else if (seconds < 3600) {
        const minutes = Math.floor(seconds / 60);
        const sec = (seconds % 60).toFixed(2);
        return `${minutes}m ${sec}s`;
    } else {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const sec = ((seconds % 3600) % 60).toFixed(2);
        return `${hours}h ${minutes}m ${sec}s`;
    }
}

// Function to view file content
function viewFile(path) {
    const fileName = path.split('\\').pop().split('/').pop();

    if (window.Swal) {
        Swal.fire({
            title: `File: ${fileName}`,
            text: 'Loading file content...',
            didOpen: () => {
                Swal.showLoading();

                // Extract file extension
                const fileExt = fileName.split('.').pop().toLowerCase();

                // Make request to get file content
                fetch(`/file/${encodeURIComponent(fileName)}`)
                    .then(response => {
                        if (!response.ok) throw new Error('File not found');
                        return fileExt === 'json' ? response.json() : response.text();
                    })
                    .then(data => {
                        let content;

                        if (fileExt === 'json') {
                            // Format JSON with syntax highlighting
                            content = `<pre style="text-align: left; max-height: 70vh; overflow-y: auto;">${JSON.stringify(data, null, 2)}</pre>`;
                        } else if (fileExt === 'csv') {
                            // Format CSV as table
                            const lines = data.split('\n');
                            if (lines.length > 0) {
                                const headers = lines[0].split(',');
                                content = '<div style="max-height: 70vh; overflow-y: auto;"><table class="table table-striped table-sm"><thead><tr>';

                                // Add headers
                                headers.forEach(header => {
                                    content += `<th>${header}</th>`;
                                });

                                content += '</tr></thead><tbody>';

                                // Add rows (limit to first 100 rows)
                                const rowLimit = Math.min(lines.length, 101);
                                for (let i = 1; i < rowLimit; i++) {
                                    if (lines[i].trim()) {
                                        content += '<tr>';
                                        lines[i].split(',').forEach(cell => {
                                            content += `<td>${cell}</td>`;
                                        });
                                        content += '</tr>';
                                    }
                                }

                                content += '</tbody></table>';

                                if (lines.length > 101) {
                                    content += `<div class="alert alert-info">Showing first 100 rows of ${lines.length-1} total</div>`;
                                }

                                content += '</div>';
                            } else {
                                content = '<div class="alert alert-warning">Empty file</div>';
                            }
                        } else {
                            // Other file types
                            content = `<pre style="text-align: left; max-height: 70vh; overflow-y: auto;">${data}</pre>`;
                        }

                        Swal.fire({
                            title: fileName,
                            html: content,
                            width: '80%',
                            confirmButtonText: 'Close'
                        });
                    })
                    .catch(error => {
                        console.error('Error loading file:', error);
                        Swal.fire({
                            icon: 'error',
                            title: 'Error',
                            text: `Failed to load file: ${error.message}`
                        });
                    });
            }
        });
    } else {
        alert(`File viewing requires SweetAlert2 library which is not loaded.`);
    }
}
