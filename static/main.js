
function showUploadPopup() {
    document.getElementById('uploadPopup').style.display = 'flex';
}

function hideUploadPopup() {
    document.getElementById('uploadPopup').style.display = 'none';
}


function toggleDuplicatesTable() {
    const container = document.getElementById("duplicatesContainer");
    const button = document.getElementById("toggleButton");
    container.classList.toggle("expanded");
    
    // Update button text
    if (container.classList.contains("expanded")) {
        button.textContent = "Hide Duplicate Rows";
    } else {
        button.textContent = "Show Duplicate Rows";
    }
}