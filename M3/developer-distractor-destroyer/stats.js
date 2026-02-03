document.addEventListener('DOMContentLoaded', () => {
    displayStats();
});

function displayStats() {
    document.title = 'Time Statistics - Developer Distractor Destroyer';

    const timeStatsList = document.getElementById('statsList');
    const timeChartCanvas = document.getElementById('timeChart').getContext('2d');
    const clearTimeStatsBtn = document.getElementById('clearTimeStats');
    const periodSelect = document.getElementById('periodSelect');
    const exportBtn = document.getElementById('exportData');
    const importInput = document.getElementById('importFile');
    let timeChart = null;

    const gotchaStatsList = document.getElementById('gotchaList');
    const gotchaChartCanvas = document.getElementById('gotchaChart').getContext('2d');
    const clearGotchaStatsBtn = document.getElementById('clearGotchaStats');
    let gotchaChart = null;

    let intervalId = null;

    function formatTime(seconds) {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = seconds % 60;
        return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }

    // Helper function to get today's date key in YYYY-MM-DD format
    function getTodayKey() {
        const now = new Date();
        return `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`;
    }

    // Get date range based on selected period
    function getDateRange(period) {
        const today = new Date();
        today.setHours(0, 0, 0, 0);

        let startDate = null;
        const endDate = new Date(today);
        endDate.setHours(23, 59, 59, 999);

        switch (period) {
            case 'today':
                startDate = new Date(today);
                break;
            case '7days':
                startDate = new Date(today);
                startDate.setDate(startDate.getDate() - 6);
                break;
            case '30days':
                startDate = new Date(today);
                startDate.setDate(startDate.getDate() - 29);
                break;
            case 'all':
            default:
                return null; // No filter
        }

        return { start: startDate, end: endDate };
    }

    // Format date to YYYY-MM-DD
    function formatDateKey(date) {
        return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}`;
    }

    // Check if a date key is within the range
    function isDateInRange(dateKey, range) {
        if (!range) return true;
        const date = new Date(dateKey);
        return date >= range.start && date <= range.end;
    }

    // Aggregate data based on period
    function aggregateData(data, period) {
        const range = getDateRange(period);
        const aggregated = {};

        for (const domain in data) {
            const domainData = data[domain];

            // Handle both old flat structure (number) and new per-day structure (object)
            if (typeof domainData === 'number') {
                // Old flat structure - include only if showing all time
                if (period === 'all') {
                    aggregated[domain] = domainData;
                }
            } else if (typeof domainData === 'object') {
                // New per-day structure
                let total = 0;
                for (const dateKey in domainData) {
                    if (isDateInRange(dateKey, range)) {
                        total += domainData[dateKey];
                    }
                }
                if (total > 0) {
                    aggregated[domain] = total;
                }
            }
        }

        return aggregated;
    }

    function updateStats() {
        const selectedPeriod = periodSelect.value;

        chrome.storage.local.get(['timeData', 'gotchaStats'], (result) => {
            // Time Stats
            timeStatsList.innerHTML = '';
            const rawTimeData = result.timeData || {};
            const timeData = aggregateData(rawTimeData, selectedPeriod);
            const sortedTimeSites = Object.entries(timeData).sort((a, b) => b[1] - a[1]);
    
            if (sortedTimeSites.length === 0) {
                timeStatsList.innerHTML = '<div class="stat-item">No time tracking data yet.</div>';
                document.getElementById('timeChart').style.display = 'none';
            } else {
                document.getElementById('timeChart').style.display = 'block';
                sortedTimeSites.forEach(([site, time]) => {
                    const statItem = createStatItem(site, formatTime(time), timeChart, timeStatsList);
                    timeStatsList.appendChild(statItem);
                });
                renderPieChart(sortedTimeSites);
            }

            // Gotcha Stats
            gotchaStatsList.innerHTML = '';
            const rawGotchaData = result.gotchaStats || {};
            const gotchaData = aggregateData(rawGotchaData, selectedPeriod);
            const sortedGotchaSites = Object.entries(gotchaData).sort((a, b) => b[1] - a[1]);

            if (sortedGotchaSites.length === 0) {
                gotchaStatsList.innerHTML = '<div class="stat-item">No "gotcha" data yet.</div>';
                document.getElementById('gotchaChart').style.display = 'none';
            } else {
                document.getElementById('gotchaChart').style.display = 'block';
                sortedGotchaSites.forEach(([site, count]) => {
                    const statItem = createStatItem(site, `${count} times`, gotchaChart, gotchaStatsList);
                    gotchaStatsList.appendChild(statItem);
                });
                renderGotchaChart(sortedGotchaSites);
            }
        });
    }

    function removeStatEntry(statType, siteToRemove) {
        chrome.storage.local.get([statType], (result) => {
            const stats = result[statType];
            if (stats && stats[siteToRemove]) {
                delete stats[siteToRemove];
                let dataToSet = {};
                dataToSet[statType] = stats;
                chrome.storage.local.set(dataToSet, () => {
                    updateStats();
                });
            }
        });
    }

    function createStatItem(site, value, chart, listElement) {
        const statItem = document.createElement('div');
        statItem.className = 'stat-item';
        statItem.dataset.site = site;

        if (chart) {
            const index = chart.data.labels.indexOf(site);
            if (index !== -1 && !chart.getDataVisibility(index)) {
                statItem.classList.add('disabled');
            }
        }

        const siteText = document.createElement('span');
        siteText.textContent = site;

        const valueContainer = document.createElement('div');
        valueContainer.className = 'value-container';

        const valueText = document.createElement('span');
        valueText.textContent = value;

        const deleteBtn = document.createElement('span');
        deleteBtn.className = 'delete-stat-btn';
        deleteBtn.textContent = '❌';

        deleteBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            const statType = listElement.id === 'statsList' ? 'timeData' : 'gotchaStats';
            if (confirm(`Are you sure you want to delete stats for "${site}"?`)) {
                removeStatEntry(statType, site);
            }
        });

        valueContainer.appendChild(valueText);
        valueContainer.appendChild(deleteBtn);

        statItem.appendChild(siteText);
        statItem.appendChild(valueContainer);

        statItem.addEventListener('click', () => {
            if (!chart) return;
            const index = chart.data.labels.indexOf(site);
            if (index !== -1) {
                chart.toggleDataVisibility(index);
                chart.update();
                statItem.classList.toggle('disabled', !chart.getDataVisibility(index));
            }
        });

        statItem.addEventListener('mouseover', () => {
            if (!chart) return;
            const index = chart.data.labels.indexOf(site);
            if (index !== -1) {
                chart.setActiveElements([{ datasetIndex: 0, index: index }]);
                chart.update();
            }
        });

        statItem.addEventListener('mouseout', () => {
            if (!chart) return;
            chart.setActiveElements([]);
            chart.update();
        });

        return statItem;
    }

    function renderPieChart(data) {
        const labels = data.map(item => item[0]);
        const values = data.map(item => item[1]);

        if (timeChart) {
            timeChart.data.labels = labels;
            timeChart.data.datasets[0].data = values;
            timeChart.update();
            return;
        }

        timeChart = new Chart(timeChartCanvas, {
            type: 'pie',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Time Spent (seconds)',
                    data: values,
                    backgroundColor: [
                        'rgba(255, 99, 132, 0.7)',
                        'rgba(54, 162, 235, 0.7)',
                        'rgba(255, 206, 86, 0.7)',
                        'rgba(75, 192, 192, 0.7)',
                        'rgba(153, 102, 255, 0.7)',
                        'rgba(255, 159, 64, 0.7)'
                    ],
                    borderColor: [
                        'rgba(255, 99, 132, 1)',
                        'rgba(54, 162, 235, 1)',
                        'rgba(255, 206, 86, 1)',
                        'rgba(75, 192, 192, 1)',
                        'rgba(153, 102, 255, 1)',
                        'rgba(255, 159, 64, 1)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            color: 'white'
                        },
                        onClick: (e, legendItem, legend) => {
                            const index = legendItem.index;
                            const ci = legend.chart;
                            
                            ci.toggleDataVisibility(index);
                            ci.update();

                            const isVisible = ci.getDataVisibility(index);
                            const statItem = timeStatsList.querySelector(`.stat-item[data-site="${legendItem.text}"]`);
                            if (statItem) {
                                statItem.classList.toggle('disabled', !isVisible);
                            }
                        },
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                let label = context.dataset.label || '';
                                if (label) {
                                    label += ': ';
                                }
                                if (context.parsed !== null) {
                                    label += formatTime(context.parsed);
                                }
                                return label;
                            }
                        }
                    }
                }
            }
        });
    }

    function renderGotchaChart(data) {
        const labels = data.map(item => item[0]);
        const values = data.map(item => item[1]);

        if (gotchaChart) {
            gotchaChart.data.labels = labels;
            gotchaChart.data.datasets[0].data = values;
            gotchaChart.update();
            return;
        }

        gotchaChart = new Chart(gotchaChartCanvas, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: '"Gotcha" Count',
                    data: values,
                    backgroundColor: 'rgba(255, 99, 132, 0.7)',
                    borderColor: 'rgba(255, 99, 132, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    x: {
                        ticks: {
                            color: 'white'
                        }
                    },
                    y: {
                        ticks: {
                            color: 'white'
                        }
                    }
                }
            }
        });
    }

    clearTimeStatsBtn.addEventListener('click', () => {
        if (confirm('Are you sure you want to clear all time statistics? This cannot be undone.')) {
            chrome.storage.local.set({ timeData: {}, currentSessionTime: 0 }, () => {
                if (timeChart) {
                    timeChart.destroy();
                    timeChart = null;
                }
                updateStats();
            });
        }
    });

    clearGotchaStatsBtn.addEventListener('click', () => {
        if (confirm('Are you sure you want to clear all "gotcha" statistics? This cannot be undone.')) {
            chrome.storage.local.set({ gotchaStats: {} }, () => {
                if (gotchaChart) {
                    gotchaChart.destroy();
                    gotchaChart = null;
                }
                updateStats();
            });
        }
    });

    // Period selector change handler
    periodSelect.addEventListener('change', () => {
        // Reset charts to force redraw with new data
        if (timeChart) {
            timeChart.destroy();
            timeChart = null;
        }
        if (gotchaChart) {
            gotchaChart.destroy();
            gotchaChart = null;
        }
        updateStats();
    });

    // Export data function
    function exportData() {
        chrome.storage.local.get(['timeData', 'gotchaStats', 'blockedWebsites', 'isBlocking', 'storageVersion'], (result) => {
            const exportObj = {
                timeData: result.timeData || {},
                gotchaStats: result.gotchaStats || {},
                blockedWebsites: result.blockedWebsites || [],
                isBlocking: result.isBlocking !== undefined ? result.isBlocking : true,
                storageVersion: result.storageVersion || 1,
                exportDate: new Date().toISOString()
            };

            const blob = new Blob([JSON.stringify(exportObj, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `ddd-backup-${getTodayKey()}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        });
    }

    // Import data function
    function importData(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            try {
                const importObj = JSON.parse(e.target.result);

                // Validate structure
                if (typeof importObj !== 'object' || importObj === null) {
                    alert('Invalid file format: not a valid JSON object.');
                    return;
                }

                // Check for required fields
                if (!importObj.timeData && !importObj.gotchaStats) {
                    alert('Invalid file format: missing timeData or gotchaStats.');
                    return;
                }

                if (!confirm('This will replace all your current data. Are you sure you want to continue?')) {
                    return;
                }

                const dataToSet = {
                    timeData: importObj.timeData || {},
                    gotchaStats: importObj.gotchaStats || {},
                    storageVersion: importObj.storageVersion || 1
                };

                // Only import these if they exist in the file
                if (importObj.blockedWebsites !== undefined) {
                    dataToSet.blockedWebsites = importObj.blockedWebsites;
                }
                if (importObj.isBlocking !== undefined) {
                    dataToSet.isBlocking = importObj.isBlocking;
                }

                chrome.storage.local.set(dataToSet, () => {
                    // Reset charts and update
                    if (timeChart) {
                        timeChart.destroy();
                        timeChart = null;
                    }
                    if (gotchaChart) {
                        gotchaChart.destroy();
                        gotchaChart = null;
                    }
                    updateStats();
                    alert('Data imported successfully!');
                });
            } catch (error) {
                alert('Error parsing file: ' + error.message);
            }
        };
        reader.readAsText(file);
    }

    // Export button click handler
    exportBtn.addEventListener('click', exportData);

    // Import file change handler
    importInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            importData(e.target.files[0]);
            e.target.value = ''; // Reset input
        }
    });

    // Initial update
    updateStats();

    // Set up auto-refresh
    intervalId = setInterval(updateStats, 5000);

    // Clean up the interval when the page is hidden
    document.addEventListener('visibilitychange', () => {
        if (document.hidden) {
            clearInterval(intervalId);
        } else {
            intervalId = setInterval(updateStats, 5000);
        }
    });
} 